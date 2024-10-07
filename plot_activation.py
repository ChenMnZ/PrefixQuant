import torch
import torch.nn as nn
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from utils.data_utils import get_loaders
import argparse
import utils.hadamard_utils as hadamard_utils
from quantize.int_linear_fake import QuantLinear
from accelerate import infer_auto_device_map, dispatch_model
import utils.model_utils as model_utils
import utils.rotation_utils as rotation_utils
from utils.plot_utils import (plot_3D_tensor, plot_layer_ax_input,plot_layer_ax_output, plot_layer_outlier_token_num,
                            plot_outlier_token_position,plot_outlier_token,
                            plot_combined_layer_ax_input,plot_combined_layer_ax_output)
from utils.stat_utils import (stat_layer_wise_magnitude_input, stat_layer_wise_magnitude_output,stat_layer_wise_outlier_token_number,
                        stat_outlier_token_position,stat_outlier_token,get_nrom_and_decoder_class)
from utils.quant_utils import wrap_to_quant_model, register_online_had


def build_model_and_tokenizer(model_name):
    kwargs = {"torch_dtype": torch.float16, "device_map": "cpu"}
    tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True,add_bos_token=False)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,**kwargs)
    return model, tokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='/cpfs01/user/chenmengzhao/llama_quantization/llama2-hf/Llama-2-7b', help='model path')
    parser.add_argument('--model_name', type=str,default='llama-2-7b', help='model name')
    parser.add_argument('--save_dir', type=str, default='./figures/', help='where to save the images')
    parser.add_argument("--dataset",type=str,default="pile",
        choices=["wikitext2", "c4", "redpajama","pile"],
        help="Where to extract calibration data from.",)
    parser.add_argument('--num_samples', type=int, default=64)
    parser.add_argument('--seq_len', type=int, default=1024)
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    parser.add_argument("--max_memory", type=str, default="55GiB",help="The maximum memory of each GPU")
    # ----------------- rotation and prefix setting ------------------------------------
    parser.add_argument("--pre_rotate", action="store_true")
    parser.add_argument("--down_online_had", action="store_true")
    parser.add_argument("--qk_online_had", action="store_true")
    parser.add_argument("--outlier_threshold", type=int, default=64, help="\eta in Eq.(3), indicating the oitlier threshold ratio detect outlier tokens, ")
    parser.add_argument('--outlier_object', type=str, default='down_proj')
    parser.add_argument("--set_prefixed_tokens", action="store_true")
    # ----------------- What to plot ------------------------------------
    parser.add_argument("--plot_linear_input", action="store_true", help="plot token-wsie maximum values for linear inputs")
    parser.add_argument("--plot_linear_output", action="store_true", help="plot token-wsie maximum values for linear outputs")
    parser.add_argument("--plot_layer_wise_outlier_token_number", action="store_true", help="plot layer-wise outlier token number")
    parser.add_argument("--plot_outlier_token_position", action="store_true", help="count the token index of outlier tokens")
    parser.add_argument("--plot_outlier_token", action="store_true", help="count the token content of outlier tokens")
    parser.add_argument("--plot_layer_input_3d", action="store_true", help="plot the 3D image of layer inputs")
    parser.add_argument("--plot_block_output_3d", action="store_true", help="plot the 3D image of block outputs")
    parser.add_argument("--disable_legend", action="store_true",help="Weather to disable the legend")
    parser.add_argument("--keep_prefixed_token", action="store_true",help="Weather to plot the prefixed tokens")
    parser.add_argument("--only_down_proj", action="store_true")
    args = parser.parse_args()
    return args
   

def get_activation_hook(layer_name, prefixed_length, down_online_had, down_had_K, down_K, keep_prefixed_token=False):
    def hook(model, input, output):
        if 'apply_rotary_pos_emb_qk_rotation_wrapper' in layer_name:
            input_Q = input[0].transpose(1, 2).flatten(-2)
            input_K = input[1].transpose(1, 2).flatten(-2)
            output_Q = output[0].transpose(1, 2).flatten(-2)
            output_K = output[1].transpose(1, 2).flatten(-2)
            if prefixed_length > 0 and not keep_prefixed_token:
                input_Q = input_Q[:,prefixed_length:, ]
                input_K = input_K[:,prefixed_length:, ]
                output_Q = output_Q[:,prefixed_length:, ]
                output_K = output_K[:,prefixed_length:, ]
            input_activation[f'{layer_name}.Q'] = input_Q
            input_activation[f'{layer_name}.K'] = input_K
            output_activation[f'{layer_name}.Q'] = output_Q
            output_activation[f'{layer_name}.K'] = output_K
        else:
            if isinstance(input, tuple):
                x = input[0]
            y = output
            if down_online_had and 'down_proj' in layer_name:
                x = hadamard_utils.matmul_hadU_cuda(x, down_had_K, down_K)
            if prefixed_length > 0 and not keep_prefixed_token:
                x = x[:, prefixed_length: ]
                y = y[:, prefixed_length: ]
            input_activation[layer_name] = x
            output_activation[layer_name] = y
    return hook



# step1: prepapre the model and dataset
args = parse_args()
os.makedirs(args.save_dir, exist_ok=True)
model, tokenizer = build_model_and_tokenizer(args.model_path)
if args.pre_rotate:
    import utils.rotation_utils as rotation_utils
    rotation_utils.fuse_layer_norms(model)
    rotation_utils.rotate_model(model, rotate_mode="hadamard", online=args.down_online_had)
    model.half()
wrap_to_quant_model(model)
if args.pre_rotate and args.down_online_had:
    register_online_had(model)
    down_had_K, down_K = hadamard_utils.get_hadK(model.config.intermediate_size)
else:
    down_K = None
    down_had_K = None
# wrap rope for online_had and rope output capture
rope_function_name = model_utils.get_rope_function_name(model)
layers = model_utils.get_layers(model)
for layer in layers:
    rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                layer.self_attn, 
                rope_function_name, 
                config=model.config,
                online_had=args.qk_online_had)    
dataloader, _ = get_loaders(
args.dataset,
tokenizer,
train_size=args.num_samples,
val_size=0,
seed=args.seed,
seqlen=args.seq_len,
)


# step 2: get prefixed tokens （optional）
prefixed_tokens = None
prefixed_length = 0
if args.set_prefixed_tokens:
    from utils.stat_utils import get_prefixed_tokens
    if model.device.type == 'cpu':
        original_device = 'cpu'
        block_class_name = model.model.layers[0].__class__.__name__
        device_map = infer_auto_device_map(model, max_memory={i: args.max_memory for i in range(torch.cuda.device_count())}, no_split_module_classes=[block_class_name])
        model = dispatch_model(model, device_map=device_map)
    else:
        original_device = 'cuda'
    # get prefixed tokens
    if args.set_prefixed_tokens:
        prefixed_tokens = get_prefixed_tokens(dataloader, model, tokenizer, args.model_name, args.outlier_threshold, args.outlier_object)
        print(f"get {len(prefixed_tokens)} prefixed tokens; token id:{prefixed_tokens}; text: {tokenizer.decode(prefixed_tokens)}")
        prefixed_length = len(prefixed_tokens)
    if original_device == 'cpu':
        model = model.cpu()


# step 3: prepare the model, data dist and hook
block_class_name = model.model.layers[0].__class__.__name__
device_map = infer_auto_device_map(model, max_memory={i: args.max_memory for i in range(torch.cuda.device_count())}, no_split_module_classes=[block_class_name])
model = dispatch_model(model, device_map=device_map)
os.makedirs(args.save_dir, exist_ok=True)
activation_means = {}
input_activation = {}
output_activation = {}
model_family = args.model_name.split('-')[0]
norm_class, decoder_class = get_nrom_and_decoder_class(model_family, model)
if args.plot_linear_input or args.plot_layer_input_3d or args.plot_layer_wise_outlier_token_number or args.plot_outlier_token_position or args.plot_outlier_token:
    class_tuple = (nn.Linear, QuantLinear)
elif args.plot_linear_output:
    class_tuple = (nn.Linear, QuantLinear,rotation_utils.QKRotationWrapper)
elif args.plot_block_output_3d:
    class_tuple = (decoder_class)
for name, layer in model.named_modules():
    if isinstance(layer, class_tuple):
        layer.register_forward_hook(get_activation_hook(name, prefixed_length, args.down_online_had, down_had_K, down_K, args.keep_prefixed_token))
        

# step 4.1: plot the token-wise input magnitude 
if args.plot_linear_input:
    layer_names = ['q_proj', 'o_proj',  'down_proj', 'up_proj']
    stats = []
    for layer_name in layer_names:
        stats.append(stat_layer_wise_magnitude_input(dataloader, input_activation, model, layer_name, prefixed_tokens))
    plot_combined_layer_ax_input(stats, args.model_name, args.save_dir, layer_names, not args.disable_legend)
    for layer_name, stat in zip(layer_names, stats):
        plot_layer_ax_input(stat, args.model_name, args.save_dir, layer_name, not args.disable_legend)
        
        

# step 4.2: plot the token-wise output magnitude 
if args.plot_linear_output:
    # layer_names = ['q_proj', 'k_proj', 'v_proj'] # plot Q/K/V, Q/K are pre repe 
    layer_names = ['apply_rotary_pos_emb_qk_rotation_wrapper.Q', 'apply_rotary_pos_emb_qk_rotation_wrapper.K', 'v_proj']
    stats = []
    for layer_name in layer_names:
        stats.append(stat_layer_wise_magnitude_output(dataloader, input_activation, model, layer_name, prefixed_tokens))
    plot_combined_layer_ax_output(stats, args.model_name, args.save_dir, layer_names, not args.disable_legend)
    for layer_name, stat in zip(layer_names, stats):
        plot_layer_ax_output(stat, args.model_name, args.save_dir, layer_name, not  args.disable_legend)


# step 4.3: plot the layer-wise outlier token number
if args.plot_layer_wise_outlier_token_number:
    stats = stat_layer_wise_outlier_token_number(dataloader, output_activation, model, outlier_object=args.outlier_object)
    plot_layer_outlier_token_num(stats, args.model_name, args.save_dir)
    
# step 4.4: plot token indexes of outlier tokens
if args.plot_outlier_token_position:
    stats = stat_outlier_token_position(dataloader, output_activation, model, prefixed_tokens, outlier_threshold=args.outlier_threshold, outlier_object=args.outlier_object)
    plot_outlier_token_position(stats, args.model_name, args.save_dir)


# step 4.5 plot contents of outlier tokens
if args.plot_outlier_token:
    stats = stat_outlier_token(dataloader, output_activation, model, tokenizer, decode=True, outlier_threshold=args.outlier_threshold, outlier_object=args.outlier_object)
    if len(stats) == 0:
        stats.append('all in staring token')
    plot_outlier_token(stats, args.model_name, args.save_dir)

 
   
# step4.6 plot the 3D images of linear input
if args.plot_layer_input_3d:
    data = dataloader[0][0]
    if prefixed_tokens is not None:
        data = torch.cat([torch.tensor([prefixed_tokens]),data],dim=1)
    with torch.no_grad():
        model(data.to('cuda'))
    for layer_name, activation in input_activation.items():
        layer_type = layer_name.split('.')[-1]
        if args.only_down_proj and not layer_type == 'down_proj':
            continue
        if layer_type in ['k_proj','v_proj','up_proj']:
            # same as the input of other layers
            continue
        sub_save_dir = os.path.join(args.save_dir, layer_type)
        if not os.path.exists(sub_save_dir):
            os.makedirs(sub_save_dir, exist_ok=True)
        file_name = os.path.join(sub_save_dir, f'{layer_name}.png')
        if len(activation.shape) == 3:
            activation = activation[0]
        activation = activation[:512]
        plot_3D_tensor(layer_name,activation.abs(), file_name) 

# step4.6 plot the 3D images of block output
if args.plot_block_output_3d:
    data = dataloader[0][0]
    with torch.no_grad():
        model(data.to('cuda'))
    for layer_name, activation in output_activation.items():
        print(layer_name)
        layer_type = layer_name.split('.')[-1]
        layer_type = "block"
        sub_save_dir = os.path.join(args.save_dir, layer_type)
        if not os.path.exists(sub_save_dir):
            os.makedirs(sub_save_dir, exist_ok=True)
        file_name = os.path.join(sub_save_dir, f'{layer_name}.png')
        activation = activation[0]
        if len(activation.shape) == 3:
            activation = activation[0]
        activation = activation[:512]
        plot_3D_tensor(layer_name,activation.abs(), file_name)

