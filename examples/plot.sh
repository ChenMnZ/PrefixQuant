# plot layer-wise magnitude of linear inputs
CUDA_VISIBLE_DEVICES=0 python plot_activation.py \
--model_path /cpfs01/user/chenmengzhao/llama_quantization/llama2-hf/Llama-2-7b \
--model_name llama-2-7b \
--save_dir ./figures/plot_linear_input \
--plot_linear_input


# plot layer-wise magnitude of linear inputs with hadamard rotation and prefixed tokens
CUDA_VISIBLE_DEVICES=0 python plot_activation.py \
--model_path /cpfs01/user/chenmengzhao/llama_quantization/llama2-hf/Llama-2-7b \
--model_name llama-2-7b \
--pre_rotate --down_online_had --qk_online_had \
--set_prefixed_tokens \
--save_dir ./figures/plot_linear_input_rotate_prefix \
--plot_linear_input 

# plot layer-wise magnitude of linear outputs with hadamard rotation and prefixed tokens
CUDA_VISIBLE_DEVICES=0 python plot_activation.py \
--model_path /cpfs01/user/chenmengzhao/llama_quantization/llama2-hf/Llama-2-7b \
--model_name llama-2-7b \
--pre_rotate --down_online_had --qk_online_had \
--set_prefixed_tokens \
--save_dir ./figures/plot_linear_output_rotate_prefix \
--plot_linear_output 

# plot token indexes of outlier tokens with rotation
CUDA_VISIBLE_DEVICES=0 python plot_activation.py \
--model_path /cpfs01/user/chenmengzhao/llama_quantization/llama2-hf/Llama-2-7b \
--model_name llama-2-7b \
--pre_rotate --down_online_had --qk_online_had \
--save_dir ./figures/plot_layer_wise_outlier_token_number_rotate \
--plot_layer_wise_outlier_token_number


# plot contents of outlier tokens with rotation
CUDA_VISIBLE_DEVICES=0 python plot_activation.py \
--model_path /cpfs01/user/chenmengzhao/llama_quantization/llama2-hf/Llama-2-7b \
--model_name llama-2-7b \
--pre_rotate --down_online_had --qk_online_had \
--save_dir ./figures/plot_outlier_token_rotate \
--plot_outlier_token

# plot layer-wise outlier token number with rotation
CUDA_VISIBLE_DEVICES=0 python plot_activation.py \
--model_path /cpfs01/user/chenmengzhao/llama_quantization/llama2-hf/Llama-2-7b \
--model_name llama-2-7b \
--pre_rotate --down_online_had --qk_online_had \
--save_dir ./figures/plot_outlier_token_position \
--plot_outlier_token_position

# plot the 3D images of linear input
CUDA_VISIBLE_DEVICES=0 python plot_activation.py \
--model_path /cpfs01/user/chenmengzhao/llama_quantization/llama2-hf/Llama-2-7b \
--model_name llama-2-7b \
--save_dir ./figures/plot_layer_input_3d \
--plot_layer_input_3d