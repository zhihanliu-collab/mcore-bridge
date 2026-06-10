# W4A16 grouped GEMM for INT4 frozen-base qLoRA: the packed int4 weights are
# dequantized in-register inside the GEMM K-loop; the bf16 weight tensor is never
# materialized in HBM.
#
# Addressing strategy (v2.1): packed words are loaded in their NATIVE layout
# ([rows, in/8] int32, coalesced) and the 8 nibbles per word are expanded in
# registers via broadcast + reshape — never a per-element gather (the v2.0 gather
# was ~40x slower than dequant+grouped_mm; see job 1006).
#
# Two orientations, one kernel (constexpr PACK_ALONG_K):
#   forward: y_g = x_g  @ W_g^T   W [E, out, in] packed along in -> in is the K dim
#            (weight tile loaded [BN, BK], transposed in registers for tl.dot)
#   dgrad:   dx_g = dy_g @ W_g    packed dim is the N dim (tile loads already [BK, BN])
#
# Layout facts (compressed-tensors pack-quantized): weight_packed int32 [E, out, in/8]
# (8 nibbles per int32 along the *input* dim, LSB-first, offset-unsigned u4 = q + 8);
# weight_scale bf16 [E, out, in/32], group_size 32 = 4 packed words.
import torch

import triton
import triton.language as tl


@triton.jit
def _unpack_block(pk, s, ROWS: tl.constexpr, COLS: tl.constexpr):
    """pk int32 [ROWS, COLS/8], s bf16 [ROWS, COLS/32] -> bf16 [ROWS, COLS]."""
    shifts = (tl.arange(0, 8) * 4)
    q = (pk[:, :, None] >> shifts[None, None, :]) & 0xF  # [ROWS, COLS/8, 8]
    q = tl.reshape(q, (ROWS, COLS)) - 8
    sf = tl.broadcast_to(s.to(tl.float32)[:, :, None], (ROWS, COLS // 32, 32))
    sf = tl.reshape(sf, (ROWS, COLS))
    return (q.to(tl.float32) * sf).to(tl.bfloat16)


@triton.jit
def _w4a16_grouped_kernel(
    a_ptr, packed_ptr, scale_ptr, y_ptr,
    m_starts_ptr, m_sizes_ptr,
    N, K,
    OUT, IN_P, IN_G,  # per-expert weight strides: rows, packed cols, scale-group cols
    PACK_ALONG_K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    e = tl.program_id(0)
    mt = tl.program_id(1)
    nt = tl.program_id(2)
    m_size = tl.load(m_sizes_ptr + e)
    if mt * BM >= m_size:
        return
    m_start = tl.load(m_starts_ptr + e).to(tl.int64)

    offs_m = mt * BM + tl.arange(0, BM)
    offs_n = nt * BN + tl.arange(0, BN)
    mask_m = offs_m < m_size
    a_row = (m_start + offs_m).to(tl.int64)

    p_base = packed_ptr + e.to(tl.int64) * OUT * IN_P
    s_base = scale_ptr + e.to(tl.int64) * OUT * IN_G

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        a = tl.load(a_ptr + a_row[:, None] * K + offs_k[None, :], mask=mask_m[:, None], other=0.0)
        if PACK_ALONG_K:
            # weight rows = n (out), packed cols along k: load [BN, BK/8] coalesced
            pcols = (k0 // 8) + tl.arange(0, BK // 8)
            pk = tl.load(p_base + offs_n[:, None].to(tl.int64) * IN_P + pcols[None, :])
            scols = (k0 // 32) + tl.arange(0, BK // 32)
            s = tl.load(s_base + offs_n[:, None].to(tl.int64) * IN_G + scols[None, :])
            b = tl.trans(_unpack_block(pk, s, BN, BK))  # [BK, BN]
        else:
            # weight rows = k (out), packed cols along n: load [BK, BN/8] coalesced
            pcols = (nt * BN // 8) + tl.arange(0, BN // 8)
            pk = tl.load(p_base + offs_k[:, None].to(tl.int64) * IN_P + pcols[None, :])
            scols = (nt * BN // 32) + tl.arange(0, BN // 32)
            s = tl.load(s_base + offs_k[:, None].to(tl.int64) * IN_G + scols[None, :])
            b = _unpack_block(pk, s, BK, BN)  # [BK, BN]
        acc += tl.dot(a, b)

    y_offs = a_row[:, None] * N + offs_n[None, :]
    tl.store(y_ptr + y_offs, acc.to(tl.bfloat16), mask=mask_m[:, None])


def _launch(a, packed, scale, m_splits, pack_along_k):
    E, out, in_p = packed.shape
    in_g = scale.shape[-1]
    K = a.shape[-1]
    N = out if pack_along_k else in_p * 8
    if pack_along_k:
        assert K == in_p * 8, f'K {K} != packed in {in_p * 8}'
    else:
        assert K == out, f'K {K} != out {out}'
    BM, BN, BK = 64, 128, 64
    assert N % BN == 0 and K % BK == 0, f'unsupported dims N={N} K={K} for tiling {BN}x{BK}'
    m_sizes = torch.tensor(m_splits, device=a.device, dtype=torch.int32)
    m_starts = torch.zeros_like(m_sizes)
    torch.cumsum(m_sizes[:-1], 0, dtype=torch.int32, out=m_starts[1:])
    y = torch.empty(a.shape[0], N, dtype=torch.bfloat16, device=a.device)
    max_m = max(m_splits) if m_splits else 0
    if max_m == 0:
        return y
    grid = (E, triton.cdiv(max_m, BM), N // BN)
    _w4a16_grouped_kernel[grid](
        a, packed, scale, y, m_starts, m_sizes,
        N, K, out, in_p, in_g,
        PACK_ALONG_K=pack_along_k, BM=BM, BN=BN, BK=BK,
        num_warps=8, num_stages=3,
    )
    return y


def w4a16_grouped_fwd(x, packed, scale, m_splits):
    """y_g = x_g @ dequant(W_g)^T. x [M, in] bf16 contiguous -> [M, out] bf16."""
    return _launch(x, packed, scale, m_splits, pack_along_k=True)


def w4a16_grouped_dgrad(dy, packed, scale, m_splits):
    """dx_g = dy_g @ dequant(W_g). dy [M, out] bf16 contiguous -> [M, in] bf16."""
    return _launch(dy, packed, scale, m_splits, pack_along_k=False)
