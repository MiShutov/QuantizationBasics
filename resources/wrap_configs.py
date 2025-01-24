from resources.quantization_tools import *

INFERENCE_CONFIG = {
    "wrap_rule": {QLinear: QuantizedLinear(), QConv2d: QuantizedConv2d()},
}

INT8_CONFIG = {
    "wrap_rule": {
        torch.nn.Linear: QLinear(
            weight_quantizer=Quantizer(
                bit_width=8, use_offset=True, group_type="tensor"
            )
        ),
        torch.nn.Conv2d: QConv2d(
            weight_quantizer=Quantizer(
                bit_width=8, use_offset=True, group_type="tensor"
            )
        ),
    },
}

INT4_PER_TENSOR_CONFIG = {
    "wrap_rule": {
        torch.nn.Linear: QLinear(
            weight_quantizer=Quantizer(
                bit_width=8, use_offset=True, group_type="tensor"
            )
        ),
        torch.nn.Conv2d: QConv2d(
            weight_quantizer=Quantizer(
                bit_width=4, use_offset=True, group_type="tensor"
            )
        ),
    },
    "exceptions": {
        "resnet.embedder.embedder.convolution": QConv2d(
            weight_quantizer=Quantizer(
                bit_width=8, use_offset=True, group_type="tensor"
            )
        )
    },
}

INT4_PER_CHANNEL_CONFIG = {
    "wrap_rule": {
        torch.nn.Linear: QLinear(
            weight_quantizer=Quantizer(
                bit_width=8, use_offset=True, group_type="channel"
            )
        ),
        torch.nn.Conv2d: QConv2d(
            weight_quantizer=Quantizer(
                bit_width=4, use_offset=True, group_type="channel"
            )
        ),
    },
    "exceptions": {
        "resnet.embedder.embedder.convolution": QConv2d(
            weight_quantizer=Quantizer(
                bit_width=8, use_offset=True, group_type="tensor"
            )
        )
    },
}

INT3_PER_CHANNEL_CONFIG = {
    "wrap_rule": {
        torch.nn.Linear: QLinear(
            weight_quantizer=Quantizer(
                bit_width=8, use_offset=True, group_type="channel"
            )
        ),
        torch.nn.Conv2d: QConv2d(
            weight_quantizer=Quantizer(
                bit_width=3, use_offset=True, group_type="channel"
            )
        ),
    },
    "exceptions": {
        "resnet.embedder.embedder.convolution": QConv2d(
            weight_quantizer=Quantizer(
                bit_width=8, use_offset=True, group_type="tensor"
            )
        )
    },
}
