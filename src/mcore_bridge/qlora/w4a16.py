# W4A16 grouped GEMM for INT4 frozen-base qLoRA: the packed int4 weights are
# dequantized in-register inside the GEMM K-loop; the bf16 weight tensor is never
# materialized in HBM. Replaces dequant(0.66r+2.6w GiB) + GEMM(2.6r GiB) per call
# with a single pass reading ~0.66 GiB of packed data.
#
# Two orientations, one kernel (constexpr PACK_ALONG_K):
#   forward: y_g = x_g  @ W_g^T   W [E, out, in] packed along in  -> in  is the K dim
#   dgrad:   dx_g = dy_g @ W_g    W [E, out, in] packed along in  -> in  is the N dim
#
# Layout facts (compressed-tensors pack-quantized): weight_packed int32 [E, out, in/8]
# (8 nibbles per int32 along the *input* dim, LSB-first, offset-unsigned u4 = q + 8);
# weight_scale bf16 [E, out, in/32].
import torch

import triton
import triton.language as tl


@triton.jit
def _w4a16_grouped_kernel(
    a_ptr, packed_ptr, scale_ptr, y_ptr,
    m_starts_ptr, m_sizes_ptr,
    N, K,
    OUT, IN_P, IN_G,  # per-expert packed strides: rows, packed cols, scale-group cols
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
            # forward: b[k, n] = W[n, k]; packed along k
            pcol = offs_k // 8
            shift = (offs_k % 8) * 4
            pk = tl.load(p_base + offs_n[None, :].to(tl.int64) * IN_P + pcol[:, None])
            q = ((pk >> shift[:, None]) & 0xF) - 8
            s = tl.load(s_base + offs_n[None, :].to(tl.int64) * IN_G + (offs_k // 32)[:, None]).to(tl.float32)
        else:
            # dgrad: b[k, n] = W[k, n]; packed along n
            pcol = offs_n // 8
            shift = (offs_n % 8) * 4
            pk = tl.load(p_base + offs_k[:, None].to(tl.int64) * IN_P + pcol[None, :])
            q = ((pk >> shift[None, :]) & 0xF) - 8
            s = tl.load(s_base + offs_k[:, None].to(tl.int64) * IN_G + (offs_n // 32)[None, :]).to(tl.float32)
        b = (q.to(tl.float32) * s).to(tl.bfloat16)
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
    m_sizes = torch.tensor(m_splits, device=a.device, dtype=torch.int32)
    m_starts = torch.zeros_like(m_sizes)
    torch.cumsum(m_sizes[:-1], 0, dtype=torch.int32, out=m_starts[1:])
    y = torch.empty(a.shape[0], N, dtype=torch.bfloat16, device=a.device)
    max_m = max(m_splits) if m_splits else 0
    if max_m == 0:
        return y
    BM, BN, BK = 64, 128, 64
    grid = (E, triton.cdiv(max_m, BM), triton.cdiv(N, BN))
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
