import torch
import torch.nn as nn
from quantize.quantizer import UniformAffineQuantizer

class QuantRMSNorm(nn.Module):
    def __init__(self, 
                ori_norm,
                output_bits=16,
                output_group_size=64,
                output_mode='dynamic',
                output_asym=True,
                lac=False,
                output_stat=None,
                output_scale=None,
                ):
        super().__init__()
        self.register_buffer('weight',ori_norm.weight)
        self.bias = None
        self.variance_epsilon = ori_norm.variance_epsilon
        self.use_temporary_parameter = False
        self.use_act_quant = False
        self.output_bits = output_bits
        self.out_features = self.weight.shape[-1]

        if output_bits < 16:
            self.output_quantizer =  UniformAffineQuantizer(output_bits, output_group_size, output_asym, 
                                                            quantized_dims=self.out_features,mode=output_mode, 
                                                            quantized_item_stat=output_stat,
                                                            init_scale=output_scale,
                                                            learnable_clipping=lac,
                                                            quant_type='activation')


    def forward(self, x):
        if self.use_temporary_parameter:
            weight = self.temp_weight
        else:
            weight = self.weight

        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x =  x.to(input_dtype) * weight

        if self.use_act_quant and self.output_bits < 16:
            x = self.output_quantizer(x)
            
        return x

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
