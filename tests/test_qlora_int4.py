# CPU unit tests for INT4 qLoRA dequant + grouped forward, validated against the
# official `compressed_tensors` pack/unpack/dequantize implementations.
# Run: python tests/test_qlora_int4.py  (needs torch + compressed-tensors, CPU is fine)
import importlib.util
import os
import types

import torch

# Load qlora.int4 directly (the package __init__ needs peft/megatron, unavailable on CPU boxes).
_spec = importlib.util.spec_from_file_location(
    'qlora_int4', os.path.join(os.path.dirname(__file__), '..', 'src', 'mcore_bridge', 'qlora', 'int4.py'))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
dequant_int4, qlora_grouped_forward = _m.dequant_int4, _m.qlora_grouped_forward

from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32, unpack_from_int32  # noqa: E402

GROUP = 32


def quantize_ref(w: torch.Tensor, group: int = GROUP):
    """Symmetric group-wise int4 quantization (minmax observer, like the kimi ckpt)."""
    out, in_ = w.shape
    wg = w.reshape(out, in_ // group, group)
    maxabs = wg.abs().amax(dim=-1, keepdim=True)
    scale = (maxabs / 7.0).clamp(min=1e-8)
    q = torch.clamp(torch.round(wg / scale), -8, 7).to(torch.int8)
    return q.reshape(out, in_), scale.squeeze(-1).to(torch.bfloat16)


def test_dequant_matches_reference():
    torch.manual_seed(0)
    out, in_ = 64, 256
    w = torch.randn(out, in_, dtype=torch.float32)
    q, scale = quantize_ref(w)
    packed = pack_to_int32(q, 4)  # int32 [out, in//8]
    assert packed.shape == (out, in_ // 8) and packed.dtype == torch.int32

    # reference: official unpack + group-wise scale multiply
    q_ref = unpack_from_int32(packed, 4, torch.Size([out, in_]))
    ref = (q_ref.reshape(out, in_ // GROUP, GROUP).to(torch.bfloat16)
           * scale.unsqueeze(-1)).reshape(out, in_)

    mine = dequant_int4(packed, scale)
    assert mine.dtype == torch.bfloat16
    assert torch.equal(mine, ref), f'max diff {(mine.float() - ref.float()).abs().max()}'
    print('test_dequant_matches_reference OK')


def test_dequant_3d_stacked():
    torch.manual_seed(1)
    E, out, in_ = 4, 32, 128
    packed_l, scale_l, ref_l = [], [], []
    for _ in range(E):
        w = torch.randn(out, in_)
        q, scale = quantize_ref(w)
        packed_l.append(pack_to_int32(q, 4))
        scale_l.append(scale)
        q_ref = unpack_from_int32(packed_l[-1], 4, torch.Size([out, in_]))
        ref_l.append((q_ref.reshape(out, in_ // GROUP, GROUP).to(torch.bfloat16)
                      * scale.unsqueeze(-1)).reshape(out, in_))
    packed = torch.stack(packed_l)
    scale = torch.stack(scale_l)
    mine = dequant_int4(packed, scale)
    assert torch.equal(mine, torch.stack(ref_l))
    print('test_dequant_3d_stacked OK')


def test_grouped_forward_and_grad():
    torch.manual_seed(2)
    E, out, in_ = 3, 32, 128
    packed_l, scale_l, w_l = [], [], []
    for _ in range(E):
        w = torch.randn(out, in_)
        q, scale = quantize_ref(w)
        packed_l.append(pack_to_int32(q, 4))
        scale_l.append(scale)
        w_l.append(dequant_int4(packed_l[-1], scale))

    mod = types.SimpleNamespace(weight_packed=torch.stack(packed_l), weight_scale=torch.stack(scale_l))
    m_splits = [5, 0, 7]
    x = torch.randn(sum(m_splits), in_, dtype=torch.bfloat16, requires_grad=True)
    out_t, bias = qlora_grouped_forward(mod, x, m_splits)
    assert bias is None and out_t.shape == (sum(m_splits), out)

    # per-expert reference
    ref = torch.cat([
        torch.nn.functional.linear(x.detach()[0:5], w_l[0]),
        torch.nn.functional.linear(x.detach()[5:5], w_l[1]),
        torch.nn.functional.linear(x.detach()[5:12], w_l[2]),
    ])
    assert torch.equal(out_t.detach(), ref)

    # tensor m_splits accepted; gradient flows to x (frozen base => no weight grad anywhere)
    out2, _ = qlora_grouped_forward(mod, x, torch.tensor(m_splits))
    out2.float().pow(2).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad.float()).all()

    # dgrad parity: custom-Function dx must match plain autograd through the dequanted weights
    x_ref = x.detach().clone().requires_grad_(True)
    ref_out = torch.cat([
        torch.nn.functional.linear(x_ref[0:5], w_l[0]),
        torch.nn.functional.linear(x_ref[5:5], w_l[1]),
        torch.nn.functional.linear(x_ref[5:12], w_l[2]),
    ])
    ref_out.float().pow(2).sum().backward()
    assert torch.equal(x.grad, x_ref.grad), 'custom backward dx != autograd reference dx'
    print('test_grouped_forward_and_grad OK')


if __name__ == '__main__':
    test_dequant_matches_reference()
    test_dequant_3d_stacked()
    test_grouped_forward_and_grad()
    print('ALL OK')
