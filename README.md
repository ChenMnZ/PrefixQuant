# PrefixQuant
Official PyTorch implement for [PrefixQuant:Static Quantization Beats Dynamic through Prefixed Outliers in LLMs](https://arxiv.org/abs/2410.05265). 



## News

[2024/10] We release PrefixQuant, the first work to let static activation quantization outperforms dynamic ones in LLM. We only open the fake quantization code now, and the inference kernels will be released later.

## Contents
- [Installation](#Installation)
- [Quantization](#quantization)
- [Inference](#Inference)
- [Citation](#citation)


## Installation
```
conda create -n prefixquant python==3.9

conda activate prefixquant

pip install -r requirements.txt
```

## Quantization
We provide an example command to quantized `Llama-3-8B` without fine-tuning:
```
CUDA_VISIBLE_DEVICES=0 python main.py \
--model_path path/to/llama-3-8B  \
--model_name Llama-3-8b \
--output_dir ./log/llama-3-8b-w4a4kv4 \
--wbits 4 \
--input_bits 4 \
--input_mode static \
--v_bits 4 \
--k_bits 4 \
--kv_group_size 128 \
--kv_mode static \
--mse_init \
--pre_rotate \
--down_online_had \
--qk_online_had \
--set_prefixed_tokens \
--eval_ppl \
--eval_tasks  piqa,arc_easy,arc_challenge,hellaswag,winogrande \
--save_quant_dir ./pre_quantized_models/llama-3-8b-w4a4kv4
```
You can find the detailed fine-tuning setting in the paper. There are some useful information as follows:
- You can add `--epochs 20` to introduce fine-tuning for W4A4KV4 quantization, and `--epochs 10` for W4A8KV4 quantization. 
- For Llama-3-70B(-Instruct) models, you should change the default learning rate to `--quant_lr 2e-5 --weight_lr 2e-6`. 
- For Llama-2-70B, you should set `--loss_type skip_mse` for the training stability.

## Inference
We provide an example command to evaluate the quantize models:
```
CUDA_VISIBLE_DEVICES=0 python eval.py \
--quant_model ./pre_quantized_models/llama-3-8b-w4a4kv4 \
--eval_ppl \
--eval_tasks  piqa,arc_easy,arc_challenge,hellaswag,winogrande
```

## Plot Activation Distribution
We provide an example command to visualize token-wsie maximum values for linear inputs:
```
CUDA_VISIBLE_DEVICES=0 python draw_activation.py \
--model_path path/to/llama-2-7b \
--model_name llama-2-7b \
--plot_linear_input
```
You can add `--pre_rotate --down_online_had --qk_online_had` to apply hadamard rotation, and add `--set_prefixed_tokens` to set the proposed prefixed tokens in our paper.
Additionally, you can also change `--plot_linear_input` to other plotting choices, details are as follows:
- `--plot_linear_output`: plot token-wsie maximum values for linear outputs (such as Q/K/V).
- `--plot_outlier_token_position`: count the token index of outlier tokens.
- `--plot_outlier_token`: count the token content of outlier tokens
- `--plot_layer_wise_outlier_token_number`: plot layer-wise outlier token number
- `--plot_layer_input_3d` : plot the 3D image of layer inputs.
- `--plot_block_output_3d` : plot the 3D image of block outputs.

More examples can be found in `./examples/plot.sh`.


## Citation
If you use our PrefixQuant approach in your research, please cite our paper:
```
@article{prefixquant,
  title={PrefixQuant: Static Quantization Beats Dynamic through Prefixed Outliers in LLMs},
  author={Chen, Mengzhao and  Liu, Yi and Wang, Jiahao and Bin, Yi and Shao, Wenqi and Luo, Ping},
  journal={arXiv preprint arXiv:2410.05265},
  year={2024}
}
```