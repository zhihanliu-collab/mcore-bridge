# INT4 frozen-base qLoRA: pure-torch dequant + grouped-linear forward.
#
# Storage format (compressed-tensors "pack-quantized", weight-only, symmetric, group-wise):
#   weight_packed: int32 [out, in // 8]   - 8 int4 values per int32, packed along the
#                                           input dim, value j of each octet lives at
#                                           bits [4*j, 4*j+4) (LSB-first).
#   weight_scale:  bf16  [out, in // group_size]
# No zero_point / g_idx (symmetric, actorder=null).
#
# This module is import-safe on CPU-only machines (no megatron / TE imports) so the
# dequant math can be unit-tested against the `compressed_tensors` reference.
import os

import torch
import torch.nn.functional as F

QLORA_ENV = 'EE_QLORA_INT4'


def qlora_int4_enabled() -> bool:
    return os.environ.get(QLORA_ENV, '0').lower() in {'1', 'true'}


def dequant_int4(packed: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Dequantize pack-quantized int4 weights to `dtype`.

    packed: int32 [..., out, in // 8]; scale: [..., out, in // group_size].
    Returns [..., out, in].
    """
    shifts = torch.arange(0, 32, 4, device=packed.device, dtype=torch.int32)
    vals = torch.bitwise_right_shift(packed.unsqueeze(-1), shifts) & 0xF  # [..., out, in//8, 8]
    vals = vals.reshape(*packed.shape[:-1], packed.shape[-1] * 8)  # [..., out, in]
    # compressed-tensors pack_to_int32 stores OFFSET-unsigned nibbles: u4 = q + 8, q in [-8, 7]
    vals = vals - 8
    num_groups = scale.shape[-1]
    group_size = vals.shape[-1] // num_groups
    w = vals.to(dtype).reshape(*vals.shape[:-1], num_groups, group_size) * scale.unsqueeze(-1).to(dtype)
    return w.reshape(*vals.shape)


def qlora_grouped_forward(self, x: torch.Tensor, m_splits, *args, **kwargs):
    """Replacement forward for an expert TEGroupedLinear whose weights were converted
    to packed INT4 buffers (see qlora.convert). Dequantizes on the fly and runs one
    GEMM per expert. Returns (out, None) to match the mcore TEGroupedLinear contract.

    Requires full activation recompute: the dequantized bf16 weights are autograd-saved
    for the backward GEMM, so without recompute every layer's bf16 experts stay resident.
    """
    if torch.is_tensor(m_splits):
        m_splits = m_splits.tolist()
    # One vectorized dequant for all local experts of this projection (transient bf16).
    w = dequant_int4(self.weight_packed, self.weight_scale, x.dtype)  # [E, out, in]
    outs = []
    start = 0
    for i, m in enumerate(m_splits):
        outs.append(F.linear(x[start:start + m], w[i]))
        start += m
    return torch.cat(outs, dim=0), None
