
import torch
import numpy as np
from collections import Counter

@torch.no_grad()
def stat_layer_wise_magnitude_input(dataloader, activation_dict, model, layer_name,prefixed_tokens):
    '''
    top-1/2/3
    median
    '''
    stats = []
    data_num = len(dataloader)
    for i in range(data_num):
        data = dataloader[i][0]
        if prefixed_tokens is not None:
            data = torch.cat([torch.tensor([prefixed_tokens]),data],dim=1)
        with torch.no_grad():
            model(data.to('cuda'))
        num_layers = len(model.model.layers)
        seq_np = np.zeros((5, num_layers))
        for block_index in range(num_layers):
            if layer_name == 'hidden_state':
                entire_name = f'model.layers.{block_index}'
            elif layer_name == 'down_proj':
                entire_name = f'model.layers.{block_index}.mlp.down_proj'
            elif layer_name == 'up_proj':
                entire_name = f'model.layers.{block_index}.mlp.up_proj'
            elif layer_name == 'q_proj':
                entire_name = f'model.layers.{block_index}.self_attn.q_proj'
            elif layer_name == 'o_proj':
                entire_name = f'model.layers.{block_index}.self_attn.o_proj'
            else:
                raise NotImplementedError
            activation_abs = activation_dict[entire_name].abs()
            activation_abs = activation_abs.max(dim=-1).values
            sort_res = torch.sort(activation_abs.flatten(), descending=True)
            seq_np[:3, block_index] = sort_res.values[:3].cpu() # top 1, top 2, top 3
            seq_np[3, block_index] = torch.median(activation_abs).cpu() # median
            seq_np[4, block_index] = torch.min(activation_abs).cpu() # minimum
        stats.append(seq_np)
    return stats

@torch.no_grad()
def stat_layer_wise_magnitude_output(dataloader, activation_dict, model, layer_name,prefixed_tokens):
    '''
    min-1/2/3
    median
    top-1
    '''
    stats = []
    data_num = len(dataloader)
    for i in range(data_num):
        data = dataloader[i][0]
        if prefixed_tokens is not None:
            data = torch.cat([torch.tensor([prefixed_tokens]),data],dim=1)
        with torch.no_grad():
            model(data.to('cuda'))
        num_layers = len(model.model.layers)
        seq_np = np.zeros((5, num_layers))
        for block_index in range(num_layers):
            if layer_name == 'hidden_state':
                entire_name = f'model.layers.{block_index}'
            elif layer_name == 'down_proj':
                entire_name = f'model.layers.{block_index}.mlp.down_proj'
            elif layer_name == 'up_proj':
                entire_name = f'model.layers.{block_index}.mlp.up_proj'
            elif layer_name == 'q_proj':
                entire_name = f'model.layers.{block_index}.self_attn.q_proj'
            elif layer_name == 'k_proj':
                entire_name = f'model.layers.{block_index}.self_attn.k_proj'
            elif layer_name == 'v_proj':
                entire_name = f'model.layers.{block_index}.self_attn.v_proj'
            elif layer_name == 'o_proj':
                entire_name = f'model.layers.{block_index}.self_attn.o_proj'
            elif 'apply_rotary_pos_emb_qk_rotation_wrapper' in layer_name:
                entire_name = f'model.layers.{block_index}.self_attn.{layer_name}'
            else:
                raise NotImplementedError
            activation_abs = activation_dict[entire_name].abs()
            activation_abs = activation_abs.max(dim=-1).values
            sort_res = torch.sort(activation_abs.flatten(), descending=False)
            seq_np[:3, block_index] = sort_res.values[:3].cpu() # min 1, min 2, min 3
            seq_np[3, block_index] = torch.median(activation_abs).cpu() # median
            seq_np[4, block_index] = torch.max(activation_abs).cpu() # maximum
        stats.append(seq_np)
    return stats


def stat_layer_wise_outlier_token_number(dataloader, output_activation, model, outlier_threshold=50, outlier_object='hidden_state'):
    stats = []
    data_num = len(dataloader)
    for i in range(data_num):
        data = dataloader[i][0]
        with torch.no_grad():
            model(data.to('cuda'))
        num_layers = len(model.model.layers)
        seq_np = np.zeros((1, num_layers))
        for block_index in range(num_layers):
            if outlier_object == 'hidden_state':
                entire_name = f'model.layers.{block_index}'
            elif outlier_object == 'down_proj':
                entire_name = f'model.layers.{block_index}.mlp.down_proj'
            else:
                raise NotImplementedError
            activation_abs = output_activation[entire_name].abs()
            activation_abs = activation_abs.max(dim=-1).values
            sort_res = torch.sort(activation_abs.flatten(), descending=True)
            ratio = sort_res.values / sort_res.values.median()
            num = (ratio > outlier_threshold).sum()
            seq_np[0, block_index] = num.cpu()
        stats.append(seq_np)
    return stats

        
def stat_outlier_token_position(dataloader, output_activation, model, prefixed_tokens=None, outlier_threshold=20, outlier_object='hidden_state'):
    stats = []
    data_num = len(dataloader)
    for i in range(data_num):
        data = dataloader[i][0]
        if prefixed_tokens is not None:
            data = torch.cat([torch.tensor([prefixed_tokens]),data],dim=1)
        with torch.no_grad():
            model(data.to('cuda'))
        num_layers = len(model.model.layers)
        for block_index in range(num_layers):
            if outlier_object == 'hidden_state':
                entire_name = f'model.layers.{block_index}'
            elif outlier_object == 'down_proj':
                entire_name = f'model.layers.{block_index}.mlp.down_proj'
            else:
                raise NotImplementedError
            activation_abs = output_activation[entire_name].abs()
            activation_abs = activation_abs.max(dim=-1).values
            sort_res = torch.sort(activation_abs.flatten(), descending=True)
            ratio = sort_res.values / sort_res.values.median()
            num = (ratio > outlier_threshold).sum()
            if num > 0:
                stats += sort_res.indices[:num].tolist()
    return stats
    

def stat_outlier_token(dataloader, output_activation, model, tokenizer=None, decode=False, outlier_threshold=20, outlier_object='hidden_state'):
    stats = []
    data_num = len(dataloader)
    for i in range(data_num):
        data = dataloader[i][0]
        with torch.no_grad():
            model(data.to('cuda'))
        num_layers = len(model.model.layers)
        for block_index in range(num_layers):
            if outlier_object == 'hidden_state':
                entire_name = f'model.layers.{block_index}'
            elif outlier_object == 'down_proj':
                entire_name = f'model.layers.{block_index}.mlp.down_proj'
            else:
                raise NotImplementedError
            activation_abs = output_activation[entire_name].abs()
            activation_abs = activation_abs.max(dim=-1).values
            sort_res = torch.sort(activation_abs.flatten(), descending=True)
            ratio = sort_res.values / sort_res.values.median()
            num = (ratio > outlier_threshold).sum()
            if num > 0:
                selected_token_indexs = sort_res.indices[:num]
                for token_index in selected_token_indexs:
                    if token_index == 0: 
                        continue
                    else:
                        if decode:
                            content = tokenizer.decode(data[0][token_index])
                            if content =='\n':
                                content = '\\n'  
                            stats.append(f"'{content}'")
                        else:
                            stats.append(data[0][token_index].item())
    return stats
    

def get_activation_hook_2(layer_name, activation_dict, is_input=True):
    def hook(model, input, output):
        if is_input:
            activation_dict[layer_name] = input[0]
        else:
            activation_dict[layer_name] = output[0]
    return hook

def get_nrom_and_decoder_class(model_family, model):
    if model_family == 'llama':
        from transformers.models.llama.modeling_llama import LlamaRMSNorm,LlamaDecoderLayer
        norm_class = LlamaRMSNorm
        decoder_class = LlamaDecoderLayer
    elif model_family == 'qwen':
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm,Qwen2DecoderLayer
        norm_class = Qwen2RMSNorm
        decoder_class = Qwen2DecoderLayer
    elif model_family == 'mistral':
        from transformers.models.mistral.modeling_mistral import MistralRMSNorm,MistralDecoderLayer
        norm_class = MistralRMSNorm
        decoder_class = MistralDecoderLayer
    elif model_family == 'gemma':
        from transformers.models.gemma2.modeling_gemma2 import Gemma2RMSNorm,Gemma2DecoderLayer
        norm_class = Gemma2RMSNorm
        decoder_class = Gemma2DecoderLayer
    elif model_family == 'internlm':
        norm_class = model.model.layers[0].attention_norm.__class__
        decoder_class = model.model.layers[0].__class__
    elif model_family == 'phi':
        norm_class = model.model.layers[0].input_layernorm.__class__
        decoder_class = model.model.layers[0].__class__
    else:
        raise NotImplementedError
    return norm_class, decoder_class

def get_down_proj_name(model_family):
    if model_family in ['llama', 'qwen', 'mistral', 'phi']:
        name = 'down_proj'
    elif model_family == 'internlm':
        name = 'w2'
    else:
        raise NotImplementedError
    return name


def get_prefixed_tokens(dataloader, model, tokenizer, model_name, outlier_threshold=64, activation_type='down_proj'):
    activation_dict = {}
    model_family = model_name.split('-')[0].lower()
    norm_class, decoder_class = get_nrom_and_decoder_class(model_family, model)
    down_proj_name = get_down_proj_name(model_family)
    if activation_type == 'hidden_state':
        is_input = False
        target_class = decoder_class
    elif activation_type == 'down_proj':
        is_input = True
        from quantize.int_linear_fake import QuantLinear
        target_class = (torch.nn.Linear, QuantLinear)
    else:
        raise NotImplementedError
    
    hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, target_class):
            if isinstance(layer, torch.nn.Linear) and not down_proj_name in name:
                continue
            hooks.append(layer.register_forward_hook(get_activation_hook_2(name, activation_dict, is_input)))
    stats = stat_layer_wise_outlier_token_number(dataloader, activation_dict, model, outlier_threshold, activation_type)
    outlier_num = int(np.ceil(np.mean(stats,axis=0).max()))
    assert outlier_num > 0
    # find high-frequency outlier token
    stats = stat_outlier_token(dataloader, activation_dict, model, None,  False, outlier_threshold, activation_type)
    token_dict = Counter(stats)
    prefixed_tokens = []

    # corner case, filtering no necessary outlier token
    if len(token_dict)==1 and token_dict.most_common()[0][1]<0.1*len(dataloader):
        outlier_num -= 1
        token_dict = {}
    # for i in range(outlier_num-1):
    for i in range(outlier_num):
        if i > len(token_dict)-1:
            break
        prefixed_tokens.append(token_dict.most_common()[i][0])
    if 'qwen' in (model_name).lower():
        # start_token = tokenizer.encode('\n')     # qwen donot include bos token
        start_token = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)     # qwen has same [BOS] and [EOS]
        print('qwen model has same [BOS] and [EOS] token.')
    else:
        start_token = tokenizer.encode(tokenizer.bos_token, add_special_tokens=False)
    prefixed_tokens += start_token
    for h in hooks:
        h.remove()
    return prefixed_tokens


def get_input_modify_hook(modified_index, prefixed_tokens_num=0):
    def hook(model, input):
        input = input[0]
        # exclude pefixed tokens
        input_max = input[:, prefixed_tokens_num:].abs().max(dim=-1)
        input_max_sort =  torch.sort(input_max.values.flatten(), descending=True)
        outlier_threshold_value = input_max_sort.values[9]
        
        ratio = input_max_sort.values / outlier_threshold_value
        outlier_num = (ratio > 20).sum()


        sorted_indice = input_max_sort.indices[:outlier_num].sort().values
        if outlier_num - 1 >= modified_index:
            deal_index = sorted_indice[modified_index]
            # print(sorted_indice)
            deal_index += prefixed_tokens_num
            input[0][deal_index][input[0][deal_index].abs()>outlier_threshold_value] = 0
            # deal_index = sorted_indice[0]
            print(input_max_sort.values[:10])
            print(sorted_indice)

            
        return input
    return hook



def set_outlier_token_zero(model, model_name, modified_index=0, prefixed_tokens_num=0):
    model_family = model_name.split('-')[0].lower()
    down_proj_name = get_down_proj_name(model_family)
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Linear) and down_proj_name in name:
            layer.register_forward_pre_hook(get_input_modify_hook(modified_index,prefixed_tokens_num))