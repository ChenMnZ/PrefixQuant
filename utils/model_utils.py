import torch
import typing
import transformers
import utils
import os
import logging
from transformers.cache_utils import DynamicCache
import sys

OPT_MODEL = transformers.models.opt.modeling_opt.OPTForCausalLM
OPT_LAYER = transformers.models.opt.modeling_opt.OPTDecoderLayer
OPT_NORM = torch.nn.LayerNorm
LLAMA_MODEL = transformers.models.llama.modeling_llama.LlamaForCausalLM
LLAMA_LAYER = transformers.models.llama.modeling_llama.LlamaDecoderLayer
LLAMA_NORM = transformers.models.llama.modeling_llama.LlamaRMSNorm
MISTRAL_MODEL = transformers.models.mistral.modeling_mistral.MistralForCausalLM
MISTRAL_LAYER = transformers.models.mistral.modeling_mistral.MistralDecoderLayer
MISTRAL_NORM = transformers.models.mistral.modeling_mistral.MistralRMSNorm
QWEN2_MODEL = transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM
QWEN2_LAYER = transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer
QWEN2_NORM = transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm
INTERNLM2_MODEL = None
INTERNLM2_LAYER = None
INTERNLM2_NORM = None



def model_type_extractor(model):
    if isinstance(model, LLAMA_MODEL):
        return LLAMA_MODEL
    elif isinstance(model, OPT_MODEL):
        return OPT_MODEL
    elif isinstance(model, MISTRAL_MODEL):
        return MISTRAL_MODEL
    elif isinstance(model, QWEN2_MODEL):
        return QWEN2_MODEL
    elif model.config.architectures[0] == 'InternLM2ForCausalLM':
        global INTERNLM2_MODEL,INTERNLM2_LAYER,INTERNLM2_NORM
        INTERNLM2_MODEL = model.__class__
        INTERNLM2_LAYER = model.model.layers.__class__
        INTERNLM2_NORM = model.model.norm.__class__
        return INTERNLM2_MODEL
    else:
        raise ValueError(f'Unknown model type {model}')

def skip(*args, **kwargs):
    # This is a helper function to save time during the initialization! 
    pass

def get_rope_function_name(model):
    if isinstance(model, (LLAMA_MODEL, MISTRAL_MODEL, QWEN2_MODEL)):
        return "apply_rotary_pos_emb"
    raise NotImplementedError


def get_layers(model):
    if isinstance(model, OPT_MODEL):
        return model.model.decoder.layers
    if isinstance(model, LLAMA_MODEL):
        return model.model.layers
    if isinstance(model, MISTRAL_MODEL):
        return model.model.layers
    if isinstance(model, QWEN2_MODEL):
        return model.model.layers
    if isinstance(model, INTERNLM2_MODEL):
        return model.model.layers
    raise NotImplementedError


def get_llama(model_name, hf_token):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = transformers.LlamaForCausalLM.from_pretrained(model_name, torch_dtype='auto',
                                                          use_auth_token=hf_token,
                                                          low_cpu_mem_usage=True)
    model.seqlen = 2048
    logging.info('---> Loading {} Model with seq_len: {}'.format(model_name, model.seqlen))
    return model



def get_opt(model_name):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = transformers.OPTForCausalLM.from_pretrained(model_name, torch_dtype='auto',
                                                        low_cpu_mem_usage=True)
    model.seqlen = model.config.max_position_embeddings
    logging.info('---> Loading {} Model with seq_len: {}'.format(model_name, model.seqlen))
    return model


def get_model(
    model_name, hf_token=None
):
    if 'llama' in model_name:
        return get_llama(model_name, hf_token)
    elif 'opt' in model_name:
        return get_opt(model_name)
    else:
        raise ValueError(f'Unknown model {model_name}')


def get_model_type(model):
    if isinstance(model, LLAMA_MODEL):
        return LLAMA_MODEL
    elif isinstance(model, OPT_MODEL):
        return OPT_MODEL
    elif isinstance(model, MISTRAL_MODEL):
        return MISTRAL_MODEL
    elif isinstance(model, QWEN2_MODEL):
        return QWEN2_MODEL
    else:
        raise ValueError(f'Unknown model type {model}')

def get_norm_type(model):
    if isinstance(model, LLAMA_MODEL):
        return LLAMA_NORM
    elif isinstance(model, OPT_MODEL):
        return OPT_NORM
    elif isinstance(model, MISTRAL_MODEL):
        return MISTRAL_NORM
    elif isinstance(model, QWEN2_MODEL):
        return QWEN2_NORM
    elif isinstance(model, INTERNLM2_MODEL):
        return INTERNLM2_NORM
    else:
        raise ValueError(f'Unknown model type {model}')
    
    
    
# def get_embeddings(model, model_type) -> list[torch.nn.Module]:
def get_embeddings(model, model_type):
    if model_type == LLAMA_MODEL or model_type == MISTRAL_MODEL or model_type == QWEN2_MODEL:
        return [model.model.embed_tokens]
    elif model_type == INTERNLM2_MODEL:
        return [model.model.tok_embeddings]
    elif model_type == OPT_MODEL:
        return [model.model.decoder.embed_tokens, model.model.decoder.embed_positions]
    else:
        raise ValueError(f'Unknown model type {model_type}')


def get_transformer_layers(model, model_type):
    if model_type == LLAMA_MODEL or model_type == MISTRAL_MODEL or model_type == QWEN2_MODEL or model_type == INTERNLM2_MODEL:
        return [layer for layer in model.model.layers]
    elif model_type == OPT_MODEL:
        return [layer for layer in model.model.decoder.layers]
    else:
        raise ValueError(f'Unknown model type {model_type}')
    


def get_lm_head(model, model_type):
    if model_type == LLAMA_MODEL or model_type == MISTRAL_MODEL or model_type == QWEN2_MODEL:
        return model.lm_head
    elif model_type == OPT_MODEL:
        return model.lm_head
    elif model_type == INTERNLM2_MODEL:
        return model.output
    else:
        raise ValueError(f'Unknown model type {model_type}')

def get_pre_head_layernorm(model, model_type):
    if model_type == LLAMA_MODEL:
        pre_head_layernorm = model.model.norm
        assert isinstance(pre_head_layernorm,
                          LLAMA_NORM)
    elif model_type == QWEN2_MODEL:
        pre_head_layernorm = model.model.norm
        assert isinstance(pre_head_layernorm,
                          QWEN2_NORM)
    elif model_type == MISTRAL_MODEL:
        pre_head_layernorm = model.model.norm
        assert isinstance(pre_head_layernorm,
                          MISTRAL_NORM)
    elif model_type == OPT_MODEL:
        pre_head_layernorm = model.model.decoder.final_layer_norm
        assert pre_head_layernorm is not None
    elif model_type == INTERNLM2_MODEL:
        pre_head_layernorm = model.model.norm
        assert isinstance(pre_head_layernorm,
                          INTERNLM2_NORM)
    else:
        raise ValueError(f'Unknown model type {model_type}')
    return pre_head_layernorm

def get_mlp_bottleneck_size(model):
    model_type = get_model_type(model)
    if model_type == LLAMA_MODEL:
        return model.config.intermediate_size
    elif model_type == OPT_MODEL:
        return model.config.ffn_dim
    else:
        raise ValueError(f'Unknown model type {model_type}')

def replace_modules(
    root: torch.nn.Module,
    type_to_replace,
    new_module_factory,
    replace_layers: bool,
) -> None:
    """Replace modules of given type using the supplied module factory.

    Perform a depth-first search of a module hierarchy starting at root
    and replace all instances of type_to_replace with modules created by
    new_module_factory. Children of replaced modules are not processed.

    Args:
        root: the root of the module hierarchy where modules should be replaced
        type_to_replace: a type instances of which will be replaced
        new_module_factory: a function that given a module that should be replaced
            produces a module to replace it with.
    """
    for name, module in root.named_children():
        new_module = None
        if isinstance(module, type_to_replace):
            if replace_layers:  # layernorm_fusion.replace_layers case where transformer layers are replaced
                new_module = new_module_factory(module, int(name))
            else:  # layernorm_fusion.fuse_modules case where layernorms are fused
                new_module = new_module_factory(module)
        elif len(list(module.children())) > 0:
            replace_modules(module, type_to_replace, new_module_factory, replace_layers)

        if new_module is not None:
            setattr(root, name, new_module)


class RMSN(torch.nn.Module):
    """
    This class implements the Root Mean Square Normalization (RMSN) layer.
    We use the implementation from LLAMARMSNorm here:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L75
    """

    def __init__(self, mean_dim: int, eps=1e-5):
        super().__init__()
        self.variance_epsilon = eps
        self.mean_dim = mean_dim
        self.weight = torch.nn.Parameter(torch.ones(mean_dim))
        self.use_temporary_parameter = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_temporary_parameter:
            weight = self.temp_weight
        else:
            weight = self.weight

        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        variance = x.pow(2).sum(-1, keepdim=True) / self.mean_dim
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return x.to(input_dtype) * weight


def get_layer_io_save_path(args):
    return os.path.join(args.save_path, 'layer_io', f'{args.layer_idx:03d}.pt')

def capture_layer_io(model_type, layer, layer_input):
    def hook_factory(module_name, captured_vals, is_input):
        def hook(module, input, output):
            if is_input:
                captured_vals[module_name].append(input[0].detach().cpu())
            else:
                captured_vals[module_name].append(output.detach().cpu())
        return hook

    handles = []

    if model_type == LLAMA_MODEL:
        captured_inputs = {
            'k_proj': [],  # q_proj, v_proj has the same input as k_proj
            'o_proj': [],
            'gate_proj': [],  # up_proj has the same input as gate_proj
            'down_proj': []
        }

        captured_outputs = {
            'v_proj': [],
        }

        for name in captured_inputs.keys():
            module = getattr(layer.self_attn, name, None) or getattr(layer.mlp, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_inputs, True)))

        for name in captured_outputs.keys():
            module = getattr(layer.self_attn, name, None) or getattr(layer.mlp, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_outputs, False)))

    elif model_type == OPT_MODEL:
        captured_inputs = {
            'k_proj': [],  # q_proj, v_proj has the same input as k_proj
            'out_proj': [],
            'fc1': [],
            'fc2': []
        }
        captured_outputs = {
            'v_proj': [],
        }
        for name in captured_inputs.keys():
            # In OPT, fc1 and fc2 are directly contained in OPTDecoderLayer
            module = getattr(layer.self_attn, name, None) or getattr(layer, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_inputs, True)))

        for name in captured_outputs.keys():
            # In OPT, fc1 and fc2 are directly contained in OPTDecoderLayer
            module = getattr(layer.self_attn, name, None) or getattr(layer, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_outputs, False)))
    else:
        raise ValueError(f'Unknown model type {model_type}')

    # Process each sequence in the batch one by one to avoid OOM.
    for seq_idx in range(layer_input.shape[0]):
        # Extract the current sequence across all dimensions.
        seq = layer_input[seq_idx:seq_idx + 1].to(utils.DEV)
        # Perform a forward pass for the current sequence.
        layer(seq)

    # After processing all sequences, concatenate the accumulated inputs for each sub-layer across the batch.
    for module_name in captured_inputs:
        captured_inputs[module_name] = torch.cat(captured_inputs[module_name], dim=0)
    for module_name in captured_outputs:
        captured_outputs[module_name] = torch.cat(captured_outputs[module_name], dim=0)

    # Cleanup.
    for h in handles:
        h.remove()

    return {
        'input': captured_inputs,
        'output': captured_outputs
    }


def mv_kv_cache(key_values, model=None, dev=None):
    '''
    move prefixed_key_values to corresponding device through full model or target_dec
    '''
    assert model is None or dev is None
    if key_values is None:
        return None
    key_values = list(key_values)
    if model is not None:
        layers = get_layers(model)
        for layer_index in range(len(key_values)):
            block_dev = next(layers[layer_index].parameters()).device
            key_values[layer_index] = list(key_values[layer_index])
            key_values[layer_index][0] = key_values[layer_index][0].to(block_dev)
            key_values[layer_index][1] = key_values[layer_index][1].to(block_dev)
            key_values[layer_index] = tuple(key_values[layer_index])
            
    if dev is not None:
        for layer_index in range(len(key_values)):
            key_values[layer_index] = list(key_values[layer_index])
            key_values[layer_index][0] = key_values[layer_index][0].to(dev)
            key_values[layer_index][1] = key_values[layer_index][1].to(dev)
            key_values[layer_index] = tuple(key_values[layer_index])
    key_values = tuple(key_values)
    return key_values


def get_kv_cache(prefixed_key_values, bs=1):
    if bs > 1:
        prefixed_key_values = kv_cache_repeat(prefixed_key_values, bs)
    if prefixed_key_values is not None:
        kv_cache = DynamicCache.from_legacy_cache(prefixed_key_values)
    else:
        kv_cache = None
    return kv_cache


def kv_cache_repeat(key_values, bs):
    '''
    bs 1 -> bs n
    '''
    if key_values is None:
        return None
    bs_key_values = {}
    for layer_index in range(len(key_values)):
        bs_key_values[layer_index] = list(key_values[layer_index])
        bs_key_values[layer_index][0] = bs_key_values[layer_index][0].repeat_interleave(bs, dim=0)
        bs_key_values[layer_index][1] = bs_key_values[layer_index][1].repeat_interleave(bs, dim=0)
        bs_key_values[layer_index] = tuple(bs_key_values[layer_index])
    return bs_key_values
    

class WrappedPrefixCausalLM(torch.nn.Module):
    def __init__(self, model, prefixed_key_values):
        super().__init__()
        self.model = model
        self.config = model.config
        self.generation_config = model.generation_config
        self.device = model.device
        self.name_or_path = model.name_or_path
        self.vocab_size = model.vocab_size
        self.prefixed_key_values = prefixed_key_values
        self.bs_prefixed_key_values = prefixed_key_values
    
    def tie_weights(self):
        self.model.tie_weights()

    def forward(self, *args, **kwargs):
        if kwargs.get("past_key_values") is None:
            if len(args) >= 1:
                bs = args[0].shape[0]
            else:
                bs = kwargs["input_ids"].shape[0]
            self.bs_prefixed_key_values = kv_cache_repeat(self.prefixed_key_values, bs)
            kwargs["past_key_values"] = self.bs_prefixed_key_values
        return self.model.forward(*args, **kwargs)