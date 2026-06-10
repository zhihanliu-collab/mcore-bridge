# GPU validation + microbenchmark for the fused qLoRA INT4 path (triton dequant +
# torch._grouped_mm). Run on a worker via sbatch (never the login node), e.g.:
#   sbatch -N1 --gpus=1 --cpus-per-task=16 --mem=128G -p main -t 15 \
#     --wrap 'python ~/mcore-bridge/tests/verify_qlora_fused_gpu.py'
# Validates: (1) triton dequant bit-exact vs torch reference at real Kimi expert shapes;
# (2) grouped_mm fwd/dgrad parity vs per-expert loop; (3) full Function fwd+bwd parity;
# then times each variant. Exits nonzero on any mismatch.
import importlib.util
import os
import sys
import time

import torch

os.environ.setdefault('EE_QLORA_INT4', '1')

_spec = importlib.util.spec_from_file_location(
    'qlora_int4', os.path.join(os.path.dirname(__file__), '..', 'src', 'mcore_bridge', 'qlora', 'int4.py'))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

DEV = 'cuda'
# Real Kimi-K2.6 local-expert shapes at EP8 (48 experts/rank): fc1 [E, 2*moe_ffn, hidden], fc2 [E, hidden, moe_ffn]
SHAPES = {'fc1': (48, 4096, 7168), 'fc2': (48, 7168, 2048)}
M_TOTAL = 16384  # ~tokens/rank entering experts at seq 16k


def rand_packed(e, out, in_):
    packed = torch.randint(-2**31, 2**31 - 1, (e, out, in_ // 8), dtype=torch.int32, device=DEV)
    scale = (torch.rand(e, out, in_ // 32, dtype=torch.float32, device=DEV) * 0.02 + 1e-3).to(torch.bfloat16)
    return packed, scale


def timed(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000  # ms


def main():
    print(f'torch {torch.__version__}, has _grouped_mm: {hasattr(torch, "_grouped_mm")}, '
          f'device: {torch.cuda.get_device_name(0)}')
    try:
        import triton
        print(f'triton {triton.__version__}')
    except ImportError:
        print('triton MISSING')

    ok = True
    g = torch.Generator(device=DEV).manual_seed(0)
    for name, (e, out, in_) in SHAPES.items():
        packed, scale = rand_packed(e, out, in_)
        ref = _m.dequant_int4(packed, scale)
        tri = _m.dequant_int4_triton(packed, scale)
        exact = torch.equal(ref, tri)
        print(f'[dequant {name}] triton bit-exact vs torch: {exact}')
        ok &= exact

        m_splits = torch.multinomial(torch.ones(e, device=DEV), M_TOTAL, replacement=True).bincount(
            minlength=e).tolist()
        m_splits[1] = 0  # force an empty expert segment
        x = torch.randn(sum(m_splits), in_, dtype=torch.bfloat16, device=DEV, generator=g)
        dy = torch.randn(sum(m_splits), out, dtype=torch.bfloat16, device=DEV, generator=g)
        w = ref
        loop_f = _m._gemm_loop(x, w, m_splits, True)
        loop_b = _m._gemm_loop(dy, w, m_splits, False)
        if hasattr(torch, '_grouped_mm'):
            try:
                gm_f = _m._grouped_mm(x, w, m_splits, True)
                gm_b = _m._grouped_mm(dy, w, m_splits, False)
                pf = torch.allclose(gm_f.float(), loop_f.float(), atol=2e-2, rtol=2e-2)
                pb = torch.allclose(gm_b.float(), loop_b.float(), atol=2e-2, rtol=2e-2)
                print(f'[gemm {name}] grouped_mm parity fwd: {pf} dgrad: {pb}')
                ok &= pf and pb
            except Exception as exc:  # noqa: BLE001
                print(f'[gemm {name}] grouped_mm FAILED: {exc!r} (loop fallback would be used)')

        # end-to-end Function parity (auto modes) vs pure reference
        x1 = x.clone().requires_grad_(True)
        y1 = _m._QloraGroupedLinear.apply(x1, packed, scale, m_splits)
        y1.float().pow(2).sum().backward()
        x2 = x.clone().requires_grad_(True)
        y2 = _m._gemm_loop(x2, w, m_splits, True)
        y2.float().pow(2).sum().backward()
        pf = torch.allclose(y1.float(), y2.float(), atol=2e-2, rtol=2e-2)
        pb = torch.allclose(x1.grad.float(), x2.grad.float(), atol=2e-2, rtol=2e-2)
        print(f'[e2e {name}] Function fwd parity: {pf} dx parity: {pb} (modes: '
              f'dequant={_m._DEQUANT_MODE} gemm={_m._GEMM_MODE})')
        ok &= pf and pb

        # fused W4A16 grouped kernel: parity at real shapes, both orientations
        try:
            from importlib import util as _u
            _ws = _u.spec_from_file_location(
                'qlora_w4a16', os.path.join(os.path.dirname(__file__), '..', 'src', 'mcore_bridge', 'qlora',
                                            'w4a16.py'))
            _w = _u.module_from_spec(_ws)
            _ws.loader.exec_module(_w)
            wf = _w.w4a16_grouped_fwd(x, packed, scale, m_splits)
            wb = _w.w4a16_grouped_dgrad(dy, packed, scale, m_splits)
            pf = torch.allclose(wf.float(), loop_f.float(), atol=2e-2, rtol=2e-2)
            pb = torch.allclose(wb.float(), loop_b.float(), atol=2e-2, rtol=2e-2)
            print(f'[w4a16 {name}] parity fwd: {pf} dgrad: {pb} '
                  f'(max fwd diff {(wf.float() - loop_f.float()).abs().max():.4f})')
            ok &= pf and pb
        except Exception as exc:  # noqa: BLE001
            print(f'[w4a16 {name}] FAILED: {exc!r}')
            ok = False
            _w = None

        # microbench
        t_dq_torch = timed(lambda: _m.dequant_int4(packed, scale))
        t_dq_tri = timed(lambda: _m.dequant_int4_triton(packed, scale))
        t_loop = timed(lambda: _m._gemm_loop(x, w, m_splits, True))
        rows = [f'dequant torch {t_dq_torch:7.2f}ms', f'dequant triton {t_dq_tri:7.2f}ms',
                f'gemm loop {t_loop:7.2f}ms']
        if hasattr(torch, '_grouped_mm'):
            try:
                t_gm = timed(lambda: _m._grouped_mm(x, w, m_splits, True))
                rows.append(f'gemm grouped {t_gm:7.2f}ms')
            except Exception:  # noqa: BLE001
                pass
        if _w is not None:
            rows.append(f'w4a16 fwd {timed(lambda: _w.w4a16_grouped_fwd(x, packed, scale, m_splits)):7.2f}ms')
            rows.append(f'w4a16 dgrad {timed(lambda: _w.w4a16_grouped_dgrad(dy, packed, scale, m_splits)):7.2f}ms')

        def fwd_bwd():
            xx = x.clone().requires_grad_(True)
            y = _m._QloraGroupedLinear.apply(xx, packed, scale, m_splits)
            y.float().sum().backward()

        rows.append(f'Function fwd+bwd {timed(fwd_bwd, iters=10):7.2f}ms')
        print(f'[bench {name}] ' + ' | '.join(rows))

    print('VERDICT:', 'PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
