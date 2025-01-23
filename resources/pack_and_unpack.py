import torch
import math


def pack_scalar_index(tensor, bit_width):
    tensor_shape = list(tensor.shape)
    tensor = tensor.to(torch.int32)
    tensor += 2**(bit_width - 1)

    packed = pack_index(
        indice=tensor.flatten(),
        index_bits=bit_width,
        index_dtype=tensor.dtype,
        as_dtype=torch.int32,
    )
    return packed, tensor_shape, tensor.dtype


def unpack_scalar_index(packed, tensor_shape, tensor_dtype, bit_width):
    unpacked_indices = unpack_index_tensor(
        packed_tensor=packed,
        index_bits=bit_width,
        num_elements=math.prod(tensor_shape)
    )
    #print("unpacked_indices.dtype:", unpacked_indices.dtype)
    unpacked_indices -= 2**(bit_width - 1)
    return unpacked_indices.reshape(tensor_shape).to(tensor_dtype)


def unpack_index_tensor(
    packed_tensor: torch.Tensor,
    index_bits: int,
    num_elements: int
) -> torch.Tensor:
    wf = torch.arange(0, 32, 1).to(packed_tensor.device).view(1, -1)
    out = torch.bitwise_right_shift(torch.unsqueeze(packed_tensor, -1), wf)
    torch.bitwise_and(out, 1, out=out)
    pad_size = (packed_tensor.shape[-1] *
                32) % (index_bits * num_elements)
    out = out.reshape(*packed_tensor.shape[:-1], -1)
    if pad_size > 0:
        out = out[..., :-pad_size]
    out = out.reshape(*packed_tensor.shape[:-1], -1, index_bits)
    wf1 = (
        torch.arange(0, index_bits,
                     1).to(packed_tensor.device).view(1, -1)
    )
    out = torch.bitwise_left_shift(out, wf1).sum(dim=-1)

    unpacked_indices = out.to(torch.uint64).view(torch.int64)

    indices = unpacked_indices & ((1 << index_bits) - 1)
    indices = indices.view(torch.uint64).to(torch.int64)
    
    return indices


def pack_index(
    indice: torch.Tensor,
    index_bits: int,
    index_dtype: torch.dtype = torch.uint16,
    as_dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    assert index_bits <= 32, \
        f"total index bits {index_bits} should be less than 32"
    assert as_dtype in [torch.int32], "as_dtype should be int32"

    # upcast the indice to uint64 to avoid overflow on signed bit
    merged_indice = indice.view(index_dtype).to(torch.uint64
                                                ).view(torch.int64)
    # merge the indice
    wf = torch.arange(0, index_bits).to(merged_indice.device).view(1, -1)
    out = torch.bitwise_right_shift(merged_indice.unsqueeze(-1), wf)
    torch.bitwise_and(out, 1, out=out)
    out = out.reshape(*merged_indice.shape[:-1], -1)
    paded_bits = (
        32 - out.reshape(*merged_indice.shape[:-1], -1).shape[-1] % 32
    ) % 32
    out = torch.nn.functional.pad(
        out,
        (0, paded_bits),
        value=0,
        mode="constant",
    ).reshape(*merged_indice.shape[:-1], -1, 32)
    wf1 = torch.arange(0, 32, 1).to(merged_indice.device).view(1, -1)
    out = torch.bitwise_left_shift(out, wf1)
    out = out.sum(dim=-1).to(torch.uint32).view(as_dtype)

    # tests
    unpack_indices = unpack_index_tensor(
        out,
        index_bits,
        indice.shape[-1]
    )
    assert torch.allclose(
        indice.view(index_dtype).to(torch.int64),
        unpack_indices,
    )

    assert torch.allclose(
        indice.view(index_dtype).to(torch.int64),
        unpack_index_tensor(
            out,
            index_bits,
            indice.shape[-1]
        ),
    )
    return out
