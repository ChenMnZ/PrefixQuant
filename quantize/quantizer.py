import torch
import torch.nn as nn
from utils.hadamard_utils import random_hadamard_matrix


CLIPMIN = 1e-4



def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x

def clamp_ste(x: torch.Tensor, min, max):
    return (x.clamp(min,max) - x).detach() + x

def clamp_ste(x: torch.Tensor, min, max):
    return (x.clamp(min,max) - x).detach() + x


def quant_activation(x, scale, bits):
    '''
    static quantization for activation with channel-wise quantization
    '''
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    scale = clamp_ste(scale,1e-4, 1e4)
    
    bs, dim1, dim2 = x.shape

    x = x.reshape(-1, dim2)
    x_int = round_ste(x / scale)
    x_int = x_int.clamp(qmin, qmax)
    x_dequant = x_int
    x_dequant = x_dequant.mul(scale)
    x_dequant = x_dequant.reshape(bs, dim1, dim2)
            
    return x_dequant

class UniformAffineQuantizer(nn.Module):
    def __init__(
        self,
        n_bits,
        quantized_shape,
        asym=True,
        group_size=-1,
        quantized_item_stat=None,
        quant_type='weight',
        mode='static',
        minmax_init=True,
        disable_zero_point_in_sym=True,
        learnable_clipping=False,
    ):
        '''
        quantized_item_stat: 
        - weight: original weight
        - activation: channel-wise maximum values
        '''
        super().__init__()
        assert 2 <= n_bits <= 32, "bitwidth not supported"
        self.n_bits = n_bits
        self.quantized_shape = quantized_shape
        self.group_size = group_size if group_size != -1 else quantized_shape[-1]
        assert quantized_shape[-1] % group_size == 0
        self.inc_groups = quantized_shape[-1] // self.group_size
        self.quant_type = quant_type
        self.mode = mode
        self.asym = asym
        self.disable_zero_point_in_sym = disable_zero_point_in_sym
        self.learnable_clipping = learnable_clipping
        self.enable = True
        if self.asym or not self.disable_zero_point_in_sym:
            self.qmin = 0
            self.qmax = 2 ** (n_bits) - 1
        else:
            self.qmin = -(2 ** (self.n_bits - 1))
            self.qmax = 2 ** (self.n_bits - 1) - 1
            
        # init scale and zero point through Max-Min quantization
        if quant_type == 'weight':
            self.find_weight_quant_param(quantized_item_stat,minmax_init)
        elif quant_type == 'activation':
            self.find_activation_quant_param(quantized_item_stat,minmax_init)

    @torch.no_grad()     
    def find_weight_quant_param(self, quantized_item_stat,minmax_init):
        assert len(self.quantized_shape) == 2, 'only support for linear layer'
        self.mode = 'static'
        if minmax_init:
            assert quantized_item_stat is not None
            x = quantized_item_stat.reshape(-1,self.group_size)
            if self.asym:
                xmin = x.amin([-1], keepdim=True)
                xmax =  x.amax([-1], keepdim=True)
                self.original_max = xmax  # not used
                range = xmax - xmin
                scale = range / (2**self.n_bits-1)
                scale = scale.clamp(min=1e-4, max=1e4)
                zero_point = -(xmin/scale).clamp(min=-1e4, max=1e4) 
                self.scale = nn.Parameter(scale)
                self.zero_point = nn.Parameter(zero_point.round())
            else:
                xmax =  x.abs().amax([-1], keepdim=True)
                self.original_max = xmax
                scale = 2*xmax/(2**self.n_bits-1)
                scale = scale.clamp(min=1e-4, max=1e4)
                self.scale = nn.Parameter(scale)
                if self.disable_zero_point_in_sym:
                    self.zero_point = None
                    self.qmin = -(2 ** (self.n_bits - 1))
                    self.qmax = 2 ** (self.n_bits - 1) - 1
                else:
                    self.register_buffer("zero_point", (2**(self.n_bits-1)-1)*torch.ones_like(self.scale))    
        else:
            dims = self.quantized_shape[0] * self.quantized_shape[1] // self.group_size
            self.scale = nn.Parameter(torch.ones(dims,1))
            if self.asym or not self.disable_zero_point_in_sym:
                self.zero_point = nn.Parameter(torch.zeros(dims,1))
            else:
                self.zero_point = None
            

    def find_activation_quant_param(self, quantized_item_stat, minmax_init):
        if self.mode == 'static':
            if minmax_init:
                x = quantized_item_stat.reshape(-1,self.group_size)
                xmax = x.amax([-1], keepdim=True)   
                scale = 2*xmax/(2**self.n_bits-1)
                scale = scale.clamp(min=1e-4, max=1e4)
                self.scale = nn.Parameter(scale)
                if self.asym:
                    zero_point = (2**(self.n_bits-1)-1)*torch.ones_like(self.scale)
                    self.zero_point = nn.Parameter(zero_point)
                else:
                    if self.disable_zero_point_in_sym:
                        self.zero_point = None
                        self.qmin = -(2 ** (self.n_bits - 1))
                        self.qmax = 2 ** (self.n_bits - 1) - 1
                    else:
                        zero_point = (2**(self.n_bits-1))*torch.ones_like(self.scale)
                        self.register_buffer("zero_point", zero_point)
            else:
                dims = self.quantized_shape[-1] // self.group_size
                self.scale = nn.Parameter(torch.ones((dims,1)))
                if self.asym or not self.disable_zero_point_in_sym:
                    self.zero_point = nn.Parameter(torch.zeros((dims,1)))
                else:
                    self.zero_point = None
                    
                
        elif self.mode == 'dynamic':
            if self.learnable_clipping:
                if self.asym:
                    self.upbound_factor = nn.Parameter(torch.ones(1)*0.9)
                    self.lowbound_factor = nn.Parameter(torch.ones(1)*0.9)
                else:
                    self.bound_factor = nn.Parameter(torch.ones(1)*0.9)
            if not self.asym:
                self.zero_point = None
                self.qmin = -(2 ** (self.n_bits - 1))
                self.qmax = 2 ** (self.n_bits - 1) - 1

    def static_fake_quant(self, x):
        '''
        static quantization
        '''
        scale = clamp_ste(self.scale,1e-4, 1e4)
        round_zero_point = clamp_ste(round_ste(self.zero_point), self.qmin, self.qmax) if self.zero_point is not None else None
        if self.quant_type == 'weight':
            dim1, dim2 = x.shape
            x_reshaped = x.reshape(-1, self.group_size)
        elif self.quant_type == 'activation':
            bs, n, dim1 = x.shape
            x_reshaped = x.reshape(bs, n, -1, self.group_size)


        x_int = round_ste(x_reshaped / scale)
        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax)
        x_dequant = x_int
        if round_zero_point is not None:
            x_dequant = x_dequant.sub(round_zero_point)
        x_dequant = x_dequant.mul(scale)
        if self.group_size:
            if self.quant_type == 'weight':
                x_dequant = x_dequant.reshape(dim1, dim2)
            elif self.quant_type == 'activation':
                x_dequant = x_dequant.reshape(bs, n, dim1)
        return x_dequant

    def custom_quant(self, x, scale, zero_point):
        '''
        quantization with give scale and zero points
        '''
        scale = clamp_ste(scale,1e-4, 1e4)
        round_zero_point = clamp_ste(round_ste(zero_point), self.qmin, self.qmax) if zero_point is not None else None
        if self.quant_type == 'weight':
            dim1, dim2 = x.shape
            x_reshaped = x.reshape(-1, self.group_size)
        elif self.quant_type == 'activation':
            bs, n, dim1 = x.shape
            x_reshaped = x.reshape(bs, n, -1, self.group_size)


        x_int = round_ste(x_reshaped / scale)
        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax)
        x_dequant = x_int
        if round_zero_point is not None:
            x_dequant = x_dequant.sub(round_zero_point)
        x_dequant = x_dequant.mul(scale)
        if self.group_size:
            if self.quant_type == 'weight':
                x_dequant = x_dequant.reshape(dim1, dim2)
            elif self.quant_type == 'activation':
                x_dequant = x_dequant.reshape(bs, n, dim1)
        return x_dequant

    def dynamic_fake_quant(self, x):
        '''
        dynamic quantization (Min-Max quantization)
        '''
        if self.group_size and self.group_size != x.shape[-1]:
            original_shape = x.shape
            assert x.shape[-1]%self.group_size == 0
            new_shape = original_shape[:-1] + (-1,self.group_size)
            x = x.reshape(new_shape)
        if self.asym:
            xmin = x.amin([-1], keepdim=True)
            xmax =  x.amax([-1], keepdim=True)   
            if self.learnable_clipping:
                xmin = xmin * self.lowbound_factor
                xmax = xmax * self.upbound_factor
            quant_range = xmax - xmin
        else:
            xmax =  x.abs().amax([-1], keepdim=True)  
            if self.learnable_clipping:
                xmax = xmax * self.bound_factor 
            quant_range = 2 * xmax
        scale = (quant_range / (2**self.n_bits-1)).clamp(min=1e-4, max=1e4)
        round_zero_point = -(xmin/scale).round().clamp(min=-1e4, max=1e4) if self.asym else None
        x_int = round_ste(x / scale)
        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax)
        x_dequant = x_int
        if round_zero_point is not None:
            x_dequant = x_dequant.sub(round_zero_point)
        x_dequant = x_dequant.mul(scale)
        if self.group_size and self.group_size != x.shape[-1]:
            x_dequant = x_dequant.reshape(original_shape)
        return x_dequant    
        
    def deactivate(self):
        self.enable = False
        
    def activate(self):
        self.enable = True

    def forward(self, x: torch.Tensor):
        if self.n_bits >= 16 or not self.enable:
            return x

        if self.mode == 'static':
            x_dequant = self.static_fake_quant(x)
        elif self.mode == 'dynamic':
            x_dequant = self.dynamic_fake_quant(x)
        else:
            raise NotImplementedError
        return x_dequant

        

    