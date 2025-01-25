import math

import torch


def take_into_account_quantizer(quantizer, weight_shape):
    bit_per_scalar = quantizer.bit_width
    group = quantizer.group_type
    use_offset = quantizer.use_offset

    assert group in ["tensor", "channel"], f"unknown group {group}"
    if group == "tensor":
        scales_and_offset_bits = 32
    elif group == "channel":
        scales_and_offset_bits = 32 * weight_shape[0]

    if use_offset:
        scales_and_offset_bits *= 2

    return math.prod(weight_shape) * bit_per_scalar + scales_and_offset_bits


def get_bits_from_dtype(dtype):
    assert dtype in (torch.float32, torch.float16, torch.int8), f"unknown dtype {dtype}"
    if dtype == torch.float32:
        return 32
    elif dtype == torch.float16:
        return 16
    elif dtype == torch.int8:
        return 8


def calc_nparams_and_bits(module, module_name, config):
    # bias
    bias_nparams = 0
    bias_bits = 0
    if hasattr(module, "bias"):
        if module.bias is not None:
            bias_nparams = math.prod(module.bias.shape)
            bias_bits = bias_nparams * get_bits_from_dtype(module.bias.dtype)

    # weight
    weight_shape = module.weight.shape
    weight_nparams = math.prod(weight_shape)
    nparams = weight_nparams + bias_nparams

    if module_name in config.get("exceptions", []):
        quantizer = config["exceptions"][module_name].weight_quantizer
        weight_bits = take_into_account_quantizer(quantizer, weight_shape)
        return nparams, weight_bits + bias_bits

    for wrap_class in config.get("wrap_rule", []):
        if isinstance(module, wrap_class):
            quantizer = config["wrap_rule"][wrap_class].weight_quantizer
            weight_bits = take_into_account_quantizer(quantizer, weight_shape)
            return nparams, weight_bits + bias_bits

    return (
        nparams,
        weight_nparams * get_bits_from_dtype(module.weight.dtype) + bias_bits,
    )


def get_model_bit_per_weight(model, config={}, show=True):
    n_params = 0
    n_bits = 0

    for module_name, module in model.named_modules():
        if hasattr(module, "weight"):
            module_nparams, module_bits = calc_nparams_and_bits(
                module, module_name, config
            )
            n_params += module_nparams
            n_bits += module_bits

    if show:
        print(f"bit/weight: {n_bits/n_params:.02f}")
    return n_bits / n_params
