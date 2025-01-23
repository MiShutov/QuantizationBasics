import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from resources.pack_and_unpack import pack_scalar_index, unpack_scalar_index


class QLayer(nn.Module):
    def __init__(self, weight_quantizer, **kwargs):
        super().__init__()
        self.weight_quantizer = weight_quantizer
    
    def wrap_module(self, module):
        self.module = module
        self.weight_quantizer.configure(module)
        return deepcopy(self).to(module.weight.device)

    def forward(self, x):
        pass


class QLinear(QLayer):
    def __init__(self, weight_quantizer):
        super().__init__(weight_quantizer)
    
    def forward(self, x):
        bias = self.module.bias
        w = self.module.weight
        w_q = self.weight_quantizer(w)
        return F.linear(x, w_q, bias)	


class QConv2d(QLayer):
    def __init__(self, weight_quantizer):
        super().__init__(weight_quantizer)
    
    def forward(self, x):
        bias = self.module.bias
        w = self.module.weight
        w_q = self.weight_quantizer(w)
        #print("QConv2d:", ((w-w_q)**2).sum())
        return F.conv2d(x, 
                        w_q, 
                        bias, 
                        stride=self.module.stride, 
                        padding=self.module.padding, 
                        dilation=self.module.dilation, 
                        groups=self.module.groups
                )


def ste_round_pass(x):
    return x.round().detach() - x.detach() + x


class Quantizer(nn.Module):
    def __init__(
            self, 
            bit_width, 
            use_offset=True, 
            group_type='tensor',
        ):
        assert group_type in ['tensor', 'channel']
        super().__init__()
        self.bit_width = bit_width
        self.use_offset = use_offset
        self.group_type = group_type
    
    def configure(self, module):
        self.negative_clip = -2**(self.bit_width-1)
        self.positive_clip = 2**(self.bit_width-1) - 1
        
        # min max initialization
        if self.group_type=='tensor':
            w_min = module.weight.min().float()
            w_max = module.weight.max().float()
        elif self.group_type=='channel':
            w_shape = module.weight.shape

            w_min = module.weight.reshape(w_shape[0], -1).min(axis=1)[0].float()
            w_max = module.weight.reshape(w_shape[0], -1).max(axis=1)[0].float()

            w_min = w_min.view(w_shape[0], *([1] * (len(w_shape) - 1)))
            w_max = w_max.view(w_shape[0], *([1] * (len(w_shape) - 1)))


        if self.use_offset==False:				
            step = torch.where(
                torch.abs(w_min) > torch.abs(w_max), 
                w_min/self.negative_clip, 
                w_max/self.positive_clip
            )
            self.step = nn.Parameter(torch.abs(step), requires_grad=True)
            self.offset = None
        else:
            offset = (w_max * self.negative_clip - w_min * self.positive_clip) / (self.positive_clip - self.negative_clip)
            step = (w_max + offset) / self.positive_clip
            self.step = nn.Parameter(torch.abs(step), requires_grad=True)
            self.offset = nn.Parameter(offset, requires_grad=True)


    def lsq_forward(self, x):
        x_scaled = x / self.step
        x_clamped = torch.clamp(x_scaled, self.negative_clip, self.positive_clip)
        x_q = ste_round_pass(x_clamped)
        x_q = self.step * x_q
        return x_q


    @torch.no_grad()
    def get_quant(self, x):
        if self.use_offset:
            x = x + self.offset 
        x_scaled = x / self.step
        x_clamped = torch.clamp(x_scaled, self.negative_clip, self.positive_clip)
        x_rounded = x_clamped.round()
        return x_rounded


    def quantize(self, x):
        if self.use_offset:
            x = x + self.offset 

        x = self.lsq_forward(x)
        
        if self.use_offset:
            x = x - self.offset
        
        return x
    

    def forward(self, x):
        return self.quantize(x)
    
    
    def __repr__(self):
        return f"{self.__class__.__name__}(bit_width={self.bit_width}, group_type={self.group_type})"


class QuantizedLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def wrap_module(self, module):
        pass

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        packed = pack_scalar_index(
            tensor=self.int_weight.detach(), bit_width=self.bit_width
        )[0]
        destination[prefix + 'packed_weight'] = packed
        destination[prefix + 'step'] = self.step.detach()
        destination[prefix + 'offset'] = self.offset.detach()
        if self.bias is not None:
            destination[prefix + 'bias'] = self.bias.detach()

    def _load_from_state_dict(
        self,
        state_dict
    ):
        prepared_dict = {}
        for k in state_dict:
            prepared_dict.update({k.split('.')[-1] : state_dict[k]})
        
        self.int_weight.data = unpack_scalar_index(
            packed=prepared_dict.get('packed_weight', None),
            tensor_shape=self.int_weight.shape,
            tensor_dtype=self.int_weight.dtype,
            bit_width=self.bit_width
        )
        self.step.data = prepared_dict.get('step', None)
        self.offset.data = prepared_dict.get('offset', None)
        if self.bias is not None:
            self.bias.data = prepared_dict.get('bias', None)


class QuantizedLinear(QuantizedLayer):
    def __init__(self):
        super().__init__()
    
    @torch.no_grad()
    def wrap_module(self, module):
        self.int_weight = module.weight_quantizer.get_quant(module.module.weight).to(torch.int8)
        self.weight_shape = self.int_weight.shape
        self.bit_width = module.weight_quantizer.bit_width
        self.step = module.weight_quantizer.step
        self.offset = module.weight_quantizer.offset
        self.bias = module.module.bias
        return deepcopy(self)

    def forward(self, x):
        w = self.int_weight * self.step - self.offset
        return F.linear(x, w, self.bias)


class QuantizedConv2d(QuantizedLayer):
    def __init__(self):
        super().__init__()
    
    @torch.no_grad()
    def wrap_module(self, module):
        self.int_weight = module.weight_quantizer.get_quant(module.module.weight).to(torch.int8)
        self.bit_width = module.weight_quantizer.bit_width
        self.step = module.weight_quantizer.step
        self.offset = module.weight_quantizer.offset
        self.bias = module.module.bias
        self.stride = module.module.stride
        self.padding = module.module.padding
        self.dilation = module.module.dilation
        self.groups = module.module.groups
        return deepcopy(self)

    def forward(self, x):
        w = self.int_weight * self.step - self.offset
        return F.conv2d(x, 
                        w, 
                        self.bias, 
                        stride=self.stride, 
                        padding=self.padding, 
                        dilation=self.dilation, 
                        groups=self.groups
                )

