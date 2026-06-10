# INT4 frozen-base qLoRA: dequant + grouped-linear forward.
#
# Storage format (compressed-tensors "pack-quantized", weight-only, symmetric, group-wise):
#   weight_packed: int32 [out, in // 8]   - 8 int4 values per int32, packed along the
#                                           input dim, value j of each octet lives at
#                                           bits [4*j, 4*j+4) (LSB-first).
#   weight_scale:  bf16  [out, in // group_size]
# No zero_point / g_idx (symmetric, actorder=null).
#
# Forward implementation ladder (each independently env-switchable for benchmarking):
#   dequant: EE_QLORA_DEQUANT = auto|triton|torch
#     triton - fused unpack+scale kernel, no int32 intermediate (~3x less HBM traffic)
#     torch  - pure-torch reference chain (bit-exact source of truth, CPU-importable)
#   gemm:    EE_QLORA_GEMM = auto|grouped|loop
#     grouped - torch._grouped_mm, one kernel for all local experts (variable m)
#     loop    - one cuBLAS GEMM per expert (reference)
# Backward saves the PACKED buffers (already resident) and re-dequantizes, instead of
# letting autograd keep the transient bf16 weights alive — per-GEMM activation-memory
# cost drops from O(E*out*in) bf16 to zero.
#
# This module is import-safe on CPU-only machines (no megatron / TE / triton imports at
# module load) so the dequant math can be unit-tested against `compressed_tensors`.
import os
import warnings

import torch
import torch.nn.functional as F

QLORA_ENV = 'EE_QLORA_INT4'


def qlora_int4_enabled() -> bool:
    return os.environ.get(QLORA_ENV, '0').lower() in {'1', 'true'}


def dequant_int4(packed: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Pure-torch reference dequant of pack-quantized int4 weights to `dtype`.

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


# ---------------------------------------------------------------------------- triton dequant
_TRITON_KERNEL = None


def _get_triton_kernel():
    global _TRITON_KERNEL
    if _TRITON_KERNEL is None:
        import triton
        import triton.language as tl

        @triton.jit
        def _dequant_int4_kernel(packed_ptr, scale_ptr, out_ptr, n_packed, n_groups, BLOCK: tl.constexpr):
            row = tl.program_id(0).to(tl.int64)  # flattened (expert * out + out_row)
            cb = tl.program_id(1)
            offs = cb * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n_packed
            p = tl.load(packed_ptr + row * n_packed + offs, mask=mask, other=0)
            # group_size 32 = 4 packed int32s -> scale index = packed_col // 4
            s = tl.load(scale_ptr + row * n_groups + (offs >> 2), mask=mask, other=0.0).to(tl.float32)
            j = tl.arange(0, 8)
            q = ((p[:, None] >> (4 * j[None, :])) & 0xF) - 8
            # fp32 multiply of exact operands + single bf16 round == torch bf16*bf16 semantics
            w = (q.to(tl.float32) * s[:, None]).to(tl.bfloat16)
            out_offs = row * (n_packed * 8) + offs[:, None].to(tl.int64) * 8 + j[None, :]
            tl.store(out_ptr + out_offs, w, mask=mask[:, None])

        _TRITON_KERNEL = _dequant_int4_kernel
    return _TRITON_KERNEL


def dequant_int4_triton(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Fused unpack+scale dequant. packed int32 [..., out, in//8] -> bf16 [..., out, in]."""
    import triton
    kernel = _get_triton_kernel()
    n_packed = packed.shape[-1]
    n_groups = scale.shape[-1]
    rows = packed.numel() // n_packed
    out = torch.empty(*packed.shape[:-1], n_packed * 8, dtype=torch.bfloat16, device=packed.device)
    BLOCK = 256
    grid = (rows, triton.cdiv(n_packed, BLOCK))
    kernel[grid](packed, scale, out, n_packed, n_groups, BLOCK=BLOCK)
    return out


# ---------------------------------------------------------------------------- impl selection
def _env_choice(name, allowed):
    v = os.environ.get(name, 'auto').lower()
    if v not in allowed:
        raise ValueError(f'{name}={v!r} not in {sorted(allowed)}')
    return v


_DEQUANT_MODE = None  # resolved lazily: 'triton' | 'torch'
_GEMM_MODE = None  # resolved lazily: 'grouped' | 'loop'


def _dequant(packed, scale, dtype):
    global _DEQUANT_MODE
    if _DEQUANT_MODE is None:
        choice = _env_choice('EE_QLORA_DEQUANT', {'auto', 'triton', 'torch'})
        if choice == 'torch' or not packed.is_cuda or dtype != torch.bfloat16:
            _DEQUANT_MODE = 'torch'
        else:
            try:
                ref_p, ref_s = packed[..., :1, :].contiguous(), scale[..., :1, :].contiguous()
                assert torch.equal(dequant_int4_triton(ref_p, ref_s), dequant_int4(ref_p, ref_s))
                _DEQUANT_MODE = 'triton'
            except Exception as e:  # noqa: BLE001
                if choice == 'triton':
                    raise
                warnings.warn(f'[qlora-int4] triton dequant unavailable ({e!r}); falling back to torch')
                _DEQUANT_MODE = 'torch'
    if _DEQUANT_MODE == 'triton' and packed.is_cuda and dtype == torch.bfloat16:
        return dequant_int4_triton(packed, scale)
    return dequant_int4(packed, scale, dtype)


def _gemm_loop(x, w, m_splits, trans_b):
    # trans_b: y_g = x_g @ w_g^T (forward, w [E, out, in]); else y_g = x_g @ w_g (dgrad)
    outs = []
    start = 0
    for i, m in enumerate(m_splits):
        outs.append(F.linear(x[start:start + m], w[i]) if trans_b else x[start:start + m] @ w[i])
        start += m
    return torch.cat(outs, dim=0)


def _grouped_mm(x, w, m_splits, trans_b):
    offs = torch.tensor(m_splits, device=x.device, dtype=torch.int32).cumsum(0, dtype=torch.int32)
    return torch._grouped_mm(x, w.transpose(1, 2) if trans_b else w, offs=offs)


def _w4a16_dims_ok(packed):
    out, in_ = packed.shape[-2], packed.shape[-1] * 8
    return out % 128 == 0 and in_ % 128 == 0  # kernel BN/BK tiling, no masks on weight loads


def _resolve_w4a16(device):
    """Validate the fused W4A16 grouped kernel (in-register dequant, both orientations,
    incl. an empty expert segment) against dequant+loop on synthetic tensors."""
    from . import w4a16 as _w
    g = torch.Generator(device=device).manual_seed(0)
    E, out, in_ = 3, 128, 256
    packed = torch.randint(-2**31, 2**31 - 1, (E, out, in_ // 8), dtype=torch.int32, device=device)
    scale = (torch.rand(E, out, in_ // 32, device=device, generator=g) * 0.02 + 1e-3).to(torch.bfloat16)
    w = dequant_int4(packed, scale)
    m_splits = [5, 0, 7]
    x = torch.randn(sum(m_splits), in_, dtype=torch.bfloat16, device=device, generator=g)
    dy = torch.randn(sum(m_splits), out, dtype=torch.bfloat16, device=device, generator=g)
    yf = _w.w4a16_grouped_fwd(x, packed, scale, m_splits)
    yb = _w.w4a16_grouped_dgrad(dy, packed, scale, m_splits)
    assert torch.allclose(yf.float(), _gemm_loop(x, w, m_splits, True).float(), atol=2e-2, rtol=2e-2), 'w4a16 fwd'
    assert torch.allclose(yb.float(), _gemm_loop(dy, w, m_splits, False).float(), atol=2e-2, rtol=2e-2), 'w4a16 dgrad'
    return _w


_W4A16 = None  # module handle once validated


def _resolve_gemm_mode(device):
    """Pick the fastest validated implementation: fused w4a16 -> torch._grouped_mm ->
    per-expert loop. Each candidate is parity-checked on tiny synthetic tensors in BOTH
    orientations (forward trans_b=True and dgrad trans_b=False), incl. an empty expert
    segment. Any failure -> next rung."""
    global _W4A16
    choice = _env_choice('EE_QLORA_GEMM', {'auto', 'w4a16', 'grouped', 'loop'})
    if choice in {'auto', 'w4a16'}:
        try:
            _W4A16 = _resolve_w4a16(device)
            return 'w4a16'
        except Exception as e:  # noqa: BLE001
            if choice == 'w4a16':
                raise
            warnings.warn(f'[qlora-int4] w4a16 kernel unavailable ({e!r}); trying torch._grouped_mm')
    if choice == 'loop' or not hasattr(torch, '_grouped_mm'):
        return 'loop'
    try:
        g = torch.Generator(device=device).manual_seed(0)
        w = torch.randn(3, 64, 128, dtype=torch.bfloat16, device=device, generator=g)
        m_splits = [5, 0, 7]
        x = torch.randn(sum(m_splits), 128, dtype=torch.bfloat16, device=device, generator=g)
        dy = torch.randn(sum(m_splits), 64, dtype=torch.bfloat16, device=device, generator=g)
        for a, tb in ((x, True), (dy, False)):
            got = _grouped_mm(a, w, m_splits, tb)
            ref = _gemm_loop(a, w, m_splits, tb)
            assert torch.allclose(got.float(), ref.float(), atol=2e-2, rtol=2e-2), f'parity failed trans_b={tb}'
        return 'grouped'
    except Exception as e:  # noqa: BLE001
        if choice == 'grouped':
            raise
        warnings.warn(f'[qlora-int4] torch._grouped_mm unavailable ({e!r}); falling back to per-expert loop')
        return 'loop'


def _gemm(x, w, m_splits, trans_b):
    """Segmented matmul over experts. x [M, k], w [E, out, in], sum(m_splits) == M."""
    global _GEMM_MODE
    if _GEMM_MODE is None:
        _GEMM_MODE = _resolve_gemm_mode(x.device) if x.is_cuda else 'loop'
    if _GEMM_MODE == 'grouped' and x.is_cuda:
        return _grouped_mm(x, w, m_splits, trans_b)
    return _gemm_loop(x, w, m_splits, trans_b)


# ---------------------------------------------------------------------------- autograd
class _QloraGroupedLinear(torch.autograd.Function):
    """y_g = x_g @ dequant(w_g)^T per expert segment.

    Saves only the packed/scale buffers (persistently resident anyway) and re-dequantizes
    in backward — the transient bf16 weight tensor is never owned by autograd. The base is
    frozen, so backward produces dgrad only (dx = dy @ w); no wgrad GEMMs.
    """

    @staticmethod
    def forward(ctx, x, packed, scale, m_splits):
        ctx.m_splits = m_splits
        ctx.save_for_backward(packed, scale)
        return _apply_gemm(x, packed, scale, m_splits, trans_b=True)

    @staticmethod
    def backward(ctx, dy):
        packed, scale = ctx.saved_tensors
        dx = _apply_gemm(dy.contiguous(), packed, scale, ctx.m_splits, trans_b=False)
        return dx, None, None, None


def _apply_gemm(a, packed, scale, m_splits, trans_b):
    """Dispatch one segmented expert matmul. w4a16 mode never materializes the bf16
    weights; the other modes dequant first (transient) and run grouped_mm / the loop."""
    global _GEMM_MODE
    if _GEMM_MODE is None:
        _GEMM_MODE = _resolve_gemm_mode(a.device) if a.is_cuda else 'loop'
    if (_GEMM_MODE == 'w4a16' and a.is_cuda and a.dtype == torch.bfloat16
            and a.is_contiguous() and _w4a16_dims_ok(packed)):
        fn = _W4A16.w4a16_grouped_fwd if trans_b else _W4A16.w4a16_grouped_dgrad
        return fn(a, packed, scale, m_splits)
    w = _dequant(packed, scale, a.dtype)
    return _gemm(a, w, m_splits, trans_b)


def qlora_grouped_forward(self, x: torch.Tensor, m_splits, *args, **kwargs):
    """Replacement forward for an expert TEGroupedLinear whose weights were converted
    to packed INT4 buffers (see qlora.convert). Returns (out, None) to match the mcore
    TEGroupedLinear contract.

    Full activation recompute is still required for the surrounding layers (attention/
    activation tensors), but the dequantized expert weights themselves are no longer
    autograd-resident (see _QloraGroupedLinear).
    """
    if torch.is_tensor(m_splits):
        m_splits = m_splits.tolist()
    return _QloraGroupedLinear.apply(x, self.weight_packed, self.weight_scale, m_splits), None
