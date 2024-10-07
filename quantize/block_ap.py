import torch
import torch.nn as nn
import quantize.int_linear_fake as int_linear_fake
import quantize.int_linear_real as int_linear_real
from quantize.recon_loss import get_recon_loss

from torch.optim.lr_scheduler import CosineAnnealingLR
import math
import gc
from utils.quant_utils import (
    quant_parameters,weight_parameters,trainable_parameters,
    set_quant_state,quant_inplace,set_quant_parameters,
    set_weight_parameters,trainable_parameters_num,get_named_linears,set_op_by_name,
    mse_init)
import time
from utils.train_utils import NativeScalerWithGradNormCount
from utils.data_utils import BlockTrainDataset, copy_block_dataset
from contextlib import nullcontext
from utils.model_utils import get_kv_cache, mv_kv_cache


@torch.no_grad()
def update_dataset(layer, source_dataset, target_dataset, dev, attention_mask, position_ids, prefixed_key_values):
    with torch.cuda.amp.autocast():
        for index, inps in enumerate(source_dataset):
            inps = inps.to(dev)
            if len(inps.shape)==2:
                inps = inps.unsqueeze(0)
            new_data = layer(inps, attention_mask=attention_mask,position_ids=position_ids,past_key_value=get_kv_cache(prefixed_key_values, bs=source_dataset.batch_size))[0].to('cpu')
            target_dataset.update_data(index,new_data)
            
def train_one_epoch(qlayer, prefixed_key_values, attention_mask, position_ids,
                      loss_scaler, loss_func, lr_schedule, optimizer, dev, traincast,
                      quant_inps, fp_inps_with_fp, fp_inps_with_quant, training_target):
    loss_list = []
    norm_list = []
    for index in range(len(quant_inps)):
        with traincast():
            input = quant_inps[index].to(dev)
            past_key_value = get_kv_cache(prefixed_key_values, bs=input.shape[0])
            quant_out = qlayer(input, attention_mask=attention_mask,position_ids=position_ids,
                                past_key_value=past_key_value)[0]
            if training_target == 'fp_input':
                label = fp_inps_with_fp[index].to(dev)
                loss = loss_func(quant_out, label)
            elif training_target == 'quant_input':
                label = fp_inps_with_quant[index].to(dev)
                loss = loss_func(quant_out, label)
            elif training_target == 'both':
                label_1 = fp_inps_with_quant[index].to(dev)
                loss_1 = loss_func(quant_out, label_1)
                label_2 = fp_inps_with_fp[index].to(dev)
                loss_2 = loss_func(quant_out, label_2)
                loss = 1/2 * (loss_1 + loss_2)
        if not math.isfinite(loss.item()):
            print("Loss is NAN, stopping training")
        loss_list.append(loss.detach().cpu())
        optimizer.zero_grad()
        norm = loss_scaler(loss, optimizer,parameters=trainable_parameters(qlayer)).cpu()
        norm_list.append(norm.data)
        lr_schedule.step(optimizer)
    loss_mean = torch.stack(loss_list).mean()
    norm_mean = torch.stack(norm_list).mean()
    return loss_mean, norm_mean

@torch.no_grad()
def eval_one_epoch(qlayer, prefixed_key_values, attention_mask, position_ids,
                      loss_func, dev, traincast,
                      quant_inps, fp_inps_with_fp, fp_inps_with_quant, training_target):
    loss_list = []
    for index in range(len(quant_inps)):
        with traincast():
            input = quant_inps[index].to(dev)
            past_key_value = get_kv_cache(prefixed_key_values, bs=input.shape[0])
            quant_out = qlayer(input, attention_mask=attention_mask,position_ids=position_ids,
                                past_key_value=past_key_value)[0]
            if training_target == 'fp_input':
                label = fp_inps_with_fp[index].to(dev)
                loss = loss_func(quant_out, label)
            elif training_target == 'quant_input':
                label = fp_inps_with_quant[index].to(dev)
                loss = loss_func(quant_out, label)
            elif training_target == 'both':
                label_1 = fp_inps_with_quant[index].to(dev)
                loss_1 = loss_func(quant_out, label_1)
                label_2 = fp_inps_with_fp[index].to(dev)
                loss_2 = loss_func(quant_out, label_2)
                loss = 1/2 * (loss_1 + loss_2)
        loss_list.append(loss.detach().cpu())
    loss_mean = torch.stack(loss_list).mean()
    return loss_mean

class CustomLRSchedule(object):
    def __init__(self, args, total_iter) -> None:
        param_group_index = 0
        if args.quant_lr > 0:
            empty_optimizer_1 = torch.optim.AdamW([torch.tensor(0)], lr=args.quant_lr)
            self.quant_scheduler = CosineAnnealingLR(empty_optimizer_1, T_max=total_iter, eta_min=args.quant_lr/args.min_lr_factor)
            self.quant_index = param_group_index
            param_group_index += 1
        else:
            self.quant_scheduler = None
        if args.weight_lr > 0:
            empty_optimizer_2 = torch.optim.AdamW([torch.tensor(0)], lr=args.weight_lr)
            self.weight_scheduler = CosineAnnealingLR(empty_optimizer_2, T_max=total_iter, eta_min=args.weight_lr/args.min_lr_factor)
            self.weight_index = param_group_index
            param_group_index += 1  
        else:
            self.weight_scheduler = None
    def step(self, optimizer):
        if self.quant_scheduler is not None:
            self.quant_scheduler.step()
            optimizer.param_groups[self.quant_index]['lr'] = self.quant_scheduler.get_lr()[0]
        if self.weight_scheduler is not None:
            self.weight_scheduler.step()
            optimizer.param_groups[self.weight_index]['lr'] = self.weight_scheduler.get_lr()[0]
        
             
def block_ap(
    model,
    prefixed_key_values,
    args,
    trainloader,
    valloader,
    logger=None,
):
    logger.info("Starting ...")
    if args.off_load_to_disk:
        logger.info("offload the training dataset to disk, saving CPU memory, but may slowdown the training due to additional I/O...")
    
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prefixed_key_values = mv_kv_cache(prefixed_key_values, dev=dev)
    use_cache = model.config.use_cache
    model.config.use_cache = True
    
    # step 1: move embedding layer and first layer to target device, only suppress llama models now
    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    if hasattr(model.model, 'rotary_emb'):
        # for llama-3.1
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)
    dtype = torch.float16 if not args.use_fp32 else torch.float32
    traincast = torch.cuda.amp.autocast if not args.use_fp32 else nullcontext

    # step 2: init dataset
    fp_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen, 
                                model.config.hidden_size, args.batch_size, dtype, cache_path=args.cache_dir, off_load_to_disk=args.off_load_to_disk)
    fp_val_inps = BlockTrainDataset(args.val_size, args.training_seqlen, 
                                model.config.hidden_size, args.batch_size, dtype, cache_path=args.cache_dir,off_load_to_disk=args.off_load_to_disk)
    
    # step 3: catch the input of thefirst layer 
    class Catcher(nn.Module):
        def __init__(self, module, dataset):
            super().__init__()
            self.module = module
            self.dataset = dataset
            self.index = 0
            self.attention_mask = None
            self.position_ids = None

        def forward(self, inp, **kwargs):
            self.dataset.update_data(self.index, inp.squeeze(0).to('cpu'))
            self.index += 1
            if self.attention_mask is None:
                self.attention_mask = kwargs["attention_mask"]
            if self.position_ids is None:
                self.position_ids = kwargs["position_ids"]
            raise ValueError
    
    # step 3.1: catch the input of training set
    layers[0] = Catcher(layers[0],fp_train_inps)
    iters = len(trainloader)//args.batch_size
    with torch.no_grad():
        for i in range(iters):
            data = torch.cat([trainloader[j][0] for j in range(i*args.batch_size,(i+1)*args.batch_size)],dim=0)
            try:
                model(data.to(dev),past_key_values=get_kv_cache(prefixed_key_values, bs=args.batch_size))
            except ValueError:
                pass
    position_ids = layers[0].position_ids
    attention_mask = layers[0].attention_mask
    attention_mask = attention_mask.to(dtype) if attention_mask is not None else None
    layers[0] = layers[0].module
    # step 3.2: catch the input of validation set
    layers[0] = Catcher(layers[0],fp_val_inps)
    iters = len(valloader)//args.batch_size
    with torch.no_grad():
        for i in range(iters):
            data = torch.cat([valloader[j][0] for j in range(i*args.batch_size,(i+1)*args.batch_size)],dim=0)
            try:
                model(data.to(dev),past_key_values=get_kv_cache(prefixed_key_values, bs=args.batch_size))
            except ValueError:
                pass
    layers[0] = layers[0].module
    
    # step 4: move embedding layer and first layer to cpu
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    if hasattr(model.model, 'rotary_emb'):
        # for llama-3.1
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()


    # step 5: copy fp input as the quant input, they are same at the first layer
    quant_train_inps = copy_block_dataset(fp_train_inps)
    quant_val_inps = copy_block_dataset(fp_val_inps)
    if args.training_target == 'fp_input':
        fp_train_inps_with_fp = fp_train_inps
        fp_val_inps_with_fp = fp_val_inps
        fp_train_inps_with_quant = None
        fp_val_inps_with_quant = None
    elif args.training_target == 'quant_input':
        fp_train_inps_with_fp = None
        fp_val_inps_with_fp = None
        fp_train_inps_with_quant = fp_train_inps
        fp_val_inps_with_quant = fp_val_inps
    elif args.training_target == 'both':
        fp_train_inps_with_fp = fp_train_inps
        fp_val_inps_with_fp = fp_val_inps
        fp_train_inps_with_quant =  copy_block_dataset(fp_train_inps)
        fp_val_inps_with_quant = copy_block_dataset(fp_val_inps)
    else:
        raise NotImplementedError
       
    
    
    # step 6: start training    
    loss_func = get_recon_loss(args.loss_type) 
    for block_index in range(len(layers)):
        logger.info(f"=== Start quantize blocks {block_index}===")
        qlayer = layers[block_index].to(dev)
        
        qlayer.to(dev)
        # obtain output of full-precision model
        if args.epochs > 0 or args.mse_init:
            set_quant_state(qlayer,weight_quant=False,act_quant=False)
            if args.training_target == 'fp_input':
                update_dataset(qlayer,fp_train_inps_with_fp, fp_train_inps_with_fp,dev,attention_mask,position_ids,prefixed_key_values)
                update_dataset(qlayer,fp_val_inps_with_fp, fp_val_inps_with_fp,dev,attention_mask,position_ids,prefixed_key_values)
            elif args.training_target == 'quant_input':
                update_dataset(qlayer,quant_train_inps, fp_train_inps_with_quant,dev,attention_mask,position_ids,prefixed_key_values)
                update_dataset(qlayer,quant_val_inps, fp_val_inps_with_quant,dev,attention_mask,position_ids,prefixed_key_values)
            elif args.training_target == 'both':
                update_dataset(qlayer,fp_train_inps_with_fp, fp_train_inps_with_fp,dev,attention_mask,position_ids,prefixed_key_values)
                update_dataset(qlayer,fp_val_inps_with_fp, fp_val_inps_with_fp,dev,attention_mask,position_ids,prefixed_key_values)
                update_dataset(qlayer,quant_train_inps, fp_train_inps_with_quant,dev,attention_mask,position_ids,prefixed_key_values)
                update_dataset(qlayer,quant_val_inps, fp_val_inps_with_quant,dev,attention_mask,position_ids,prefixed_key_values)

        # serarch for the optimal initialization for per-tensor static activation quantization
        if args.mse_init:
            logger.info("MSE init start")
            sub_train_input = quant_train_inps.get_subset(args.mse_init_size).to(dev,torch.float16) 
            one_attention_mask = None if attention_mask is None else attention_mask[0:1]
            mse_init(qlayer,prefixed_key_values, dev, sub_train_input, one_attention_mask, position_ids, logger, args)
            # mse_init(qlayer,prefixed_key_values, dev, sub_train_input, position_ids, logger, args, sub_train_gt)
            logger.info("MSE init end")

       
        
        
        # activate quantization
        set_quant_state(qlayer,weight_quant=True,act_quant=True)  
        total_training_iteration = args.epochs * args.train_size / args.batch_size
        if args.epochs > 0:
            with torch.no_grad():
                qlayer.float()      # fp32 is also required for AMP training
            # create optimizer
            assert args.quant_lr > 0 or args.weight_lr > 0
            set_quant_parameters(qlayer,args.quant_lr > 0)
            set_weight_parameters(qlayer,args.weight_lr > 0)
            param = []
            if args.quant_lr > 0:
                param.append({"params":quant_parameters(qlayer),"lr":args.quant_lr})
            if args.weight_lr > 0:
                param.append({"params":weight_parameters(qlayer),"lr":args.weight_lr})
                
                        
            lr_schedule = CustomLRSchedule(args, total_training_iteration)
            optimizer = torch.optim.AdamW(param, weight_decay=args.wd)

            loss_scaler = NativeScalerWithGradNormCount()
            trainable_number = trainable_parameters_num(qlayer)
            logger.info(f"trainable parameter number: {trainable_number/1e6}M")

            best_val_loss = 1e6
            early_stop_flag = 0
            for epoch in range(args.epochs):
                start_time = time.time()
                train_loss, gradient_norm = train_one_epoch(qlayer, prefixed_key_values, attention_mask, position_ids,
                      loss_scaler, loss_func, lr_schedule, optimizer, dev, traincast,
                      quant_train_inps, fp_train_inps_with_fp, fp_train_inps_with_quant, args.training_target)
                val_loss = eval_one_epoch(qlayer, prefixed_key_values, attention_mask, position_ids,
                      loss_func, dev, traincast,
                      quant_val_inps, fp_val_inps_with_fp, fp_val_inps_with_quant, args.training_target)
                logger.info(f"blocks {block_index} epoch {epoch} train_loss:{train_loss} val_loss:{val_loss}  norm:{gradient_norm:.8f} max memory_allocated {torch.cuda.max_memory_allocated(dev) / 1024**2} time {time.time()-start_time} ")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                else:
                    early_stop_flag += 1
                    if args.early_stop > 0 and early_stop_flag >=args.early_stop:
                        break
            optimizer.zero_grad()
            del optimizer

        # real smooth and quantization
        qlayer.half()
        quant_inplace(qlayer)
        set_quant_state(qlayer,weight_quant=False, act_quant=True)  # weight has been quantized inplace, activation should be quantized on-line
        if args.epochs>0 or args.mse_init:
            # update inputs of quantization model
            update_dataset(qlayer,quant_train_inps, quant_train_inps,dev,attention_mask,position_ids,prefixed_key_values)
            update_dataset(qlayer,quant_val_inps, quant_val_inps,dev,attention_mask,position_ids,prefixed_key_values)
        
        # move to cpu
        layers[block_index] = qlayer.to("cpu")

        # pack quantized weights, note that this is slow on poor CPU or busy CPU
        if args.real_quant:
            assert args.input_bits>=16 and args.k_bits>=16 and args.v_bits>=16, "only supprot for weight-only quantization"
            named_linears = get_named_linears(qlayer, int_linear_fake.QuantLinear)
            for name, module in named_linears.items():
                scales = module.weight_quantizer.scale.clamp(1e-4,1e4).detach()
                zeros = module.weight_quantizer.zero_point.detach().cuda().round().cpu()
                group_size = module.weight_quantizer.group_size
                dim0 = module.weight.shape[0]
                scales = scales.view(dim0,-1).transpose(0,1).contiguous()
                zeros = zeros.view(dim0,-1).transpose(0,1).contiguous()
                q_linear = int_linear_real.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                q_linear.pack(module.cpu(),  scales.float().cpu(), zeros.float().cpu())
                set_op_by_name(qlayer, name, q_linear)       
                logger.info(f"pack quantized {name} finished")
                del module        
        torch.cuda.empty_cache()

    # delete cached dataset
    if args.off_load_to_disk:
        for dataset in [fp_train_inps, fp_val_inps, quant_train_inps, quant_val_inps]:
            if dataset is not None:
                dataset.clear_cache(())

    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model

