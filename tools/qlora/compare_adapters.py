#!/usr/bin/env python3
# Compare two LoRA adapters trained from the same data/order on different base precisions
# (INT4 qLoRA vs bf16). Reports, per module group (routed_experts / shared_expert /
# attention / dense_mlp / other):
#   - factor-level cosine of the raw lora_A / lora_B tensors, and
#   - effective-update cosine of DW = lora_B @ lora_A (the quantity that actually steers
#     the model), on a sampled set of modules.
# NaN-robust: zero-norm tensors are counted separately, not folded into the cosine mean.
#
# Interpretation: a DW cosine of ~0.88-0.90 is run-to-run nondeterminism, NOT a quant
# artifact -- bit-identical-base attention modules show the same spread as the quantized
# routed experts (job 1005).
#
# Run as a CPU Slurm job inside the modelscope container (NEVER on the login node), e.g.:
#   sbatch --nodes=1 --cpus-per-task=4 --time=00:10:00 --wrap \
#     "python tools/qlora/compare_adapters.py --a <int4>/adapter_model.safetensors \
#                                             --b <bf16>/adapter_model.safetensors"
import argparse
import random
from collections import defaultdict

import torch
from safetensors import safe_open

DEFAULT_A = ('/mnt/data/poc/ckpt/kimi-qlora-ab16/int4/'
             'v0-20260610-113526/checkpoint-24/adapter_model.safetensors')
DEFAULT_B = ('/mnt/data/poc/ckpt/kimi-qlora-ab16/bf16/'
             'v0-20260610-120915/checkpoint-24/adapter_model.safetensors')


def group(key):
    if '.mlp.experts.' in key:
        return 'routed_experts'
    if '.mlp.shared_experts.' in key:
        return 'shared_expert'
    if 'self_attn' in key:
        return 'attention'
    if '.mlp.' in key:
        return 'dense_mlp'
    return 'other'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--a', default=DEFAULT_A, help='adapter A safetensors (e.g. INT4 run)')
    ap.add_argument('--b', default=DEFAULT_B, help='adapter B safetensors (e.g. bf16 run)')
    ap.add_argument('--sample', type=int, default=60, help='modules per group for the DW comparison')
    args = ap.parse_args()

    fa, fb = safe_open(args.a, framework='pt'), safe_open(args.b, framework='pt')
    keys = sorted(set(fa.keys()) & set(fb.keys()))

    stats = defaultdict(lambda: {'cos': [], 'rel': [], 'zero_both': 0, 'zero_one': 0})
    for k in keys:
        ta = fa.get_tensor(k).float().flatten()
        tb = fb.get_tensor(k).float().flatten()
        na, nb = ta.norm().item(), tb.norm().item()
        g = stats[group(k)]
        if na == 0 and nb == 0:
            g['zero_both'] += 1
            continue
        if na == 0 or nb == 0:
            g['zero_one'] += 1
            continue
        g['cos'].append((ta @ tb / (na * nb)).item())
        g['rel'].append(((ta - tb).norm() / nb).item())

    print('=== factor-level (lora_A / lora_B separately) ===')
    print('%-16s %7s %9s %9s %9s %10s %9s' % (
        'group', 'n', 'cos_mean', 'cos_p10', 'cos_min', 'zero_both', 'zero_one'))
    for name, g in sorted(stats.items()):
        cs = torch.tensor(g['cos'])
        print('%-16s %7d %9.4f %9.4f %9.4f %10d %9d' % (
            name, len(cs), cs.mean(), cs.quantile(0.1), cs.min(), g['zero_both'], g['zero_one']))

    print()
    print('=== effective update DW = lora_B @ lora_A (sampled modules) ===')
    b_keys = [k for k in keys if k.endswith('lora_B.weight')]
    random.seed(0)
    by_grp = defaultdict(list)
    for k in b_keys:
        by_grp[group(k)].append(k)
    for name, ks in sorted(by_grp.items()):
        sample = random.sample(ks, min(args.sample, len(ks)))
        cos_l = []
        skipped = 0
        for kb in sample:
            ka_ = kb.replace('lora_B.weight', 'lora_A.weight')
            if ka_ not in keys:
                continue
            Ba, Aa = fa.get_tensor(kb).float(), fa.get_tensor(ka_).float()
            Bb, Ab = fb.get_tensor(kb).float(), fb.get_tensor(ka_).float()
            da, db = (Ba @ Aa).flatten(), (Bb @ Ab).flatten()
            if da.norm() == 0 or db.norm() == 0:
                skipped += 1
                continue
            cos_l.append((da @ db / (da.norm() * db.norm())).item())
        cs = torch.tensor(cos_l)
        if len(cs):
            print('%-16s n=%3d  cos_mean %.4f  cos_p10 %.4f  cos_min %.4f  (zero-skipped %d)' % (
                name, len(cs), cs.mean(), cs.quantile(0.1), cs.min(), skipped))


if __name__ == '__main__':
    main()
