# Verify the INT4 decode against the REAL Kimi-K2.6 checkpoints, on the cluster:
# decode weight_packed/weight_scale from /mnt/data/hf/kimi-k26 (pack-quantized) and
# compare with the same tensor in /mnt/data/hf/kimi-k26-bf16 (official dequant).
# CPU-only, a few GB of I/O. Run via sbatch (never on the login node), e.g.:
#   sbatch -N1 -n1 --cpus-per-task=8 --mem=32G -p main -t 20 \
#     --wrap 'python ~/mcore-bridge/tests/verify_kimi_int4_decode.py'
import json
import os

import torch
from safetensors import safe_open

INT4_DIR = os.environ.get('KIMI_INT4_DIR', '/mnt/data/hf/kimi-k26')
BF16_DIR = os.environ.get('KIMI_BF16_DIR', '/mnt/data/hf/kimi-k26-bf16')
KEYS = [
    'language_model.model.layers.10.mlp.experts.5.gate_proj',
    'language_model.model.layers.10.mlp.experts.5.up_proj',
    'language_model.model.layers.10.mlp.experts.5.down_proj',
    'language_model.model.layers.45.mlp.experts.300.down_proj',
]


def dequant_int4(packed, scale, dtype=torch.bfloat16):
    # keep in sync with src/mcore_bridge/qlora/int4.py
    shifts = torch.arange(0, 32, 4, device=packed.device, dtype=torch.int32)
    vals = torch.bitwise_right_shift(packed.unsqueeze(-1), shifts) & 0xF
    vals = vals.reshape(*packed.shape[:-1], packed.shape[-1] * 8)
    vals = vals - 8
    num_groups = scale.shape[-1]
    group_size = vals.shape[-1] // num_groups
    w = vals.to(dtype).reshape(*vals.shape[:-1], num_groups, group_size) * scale.unsqueeze(-1).to(dtype)
    return w.reshape(*vals.shape)


def load_tensor(model_dir, name):
    with open(os.path.join(model_dir, 'model.safetensors.index.json')) as f:
        index = json.load(f)['weight_map']
    shard = index[name]
    with safe_open(os.path.join(model_dir, shard), framework='pt') as f:
        return f.get_tensor(name)


def main():
    ok = True
    for key in KEYS:
        packed = load_tensor(INT4_DIR, key + '.weight_packed')
        scale = load_tensor(INT4_DIR, key + '.weight_scale')
        mine = dequant_int4(packed, scale)
        ref = load_tensor(BF16_DIR, key + '.weight')
        exact = torch.equal(mine, ref)
        maxdiff = (mine.float() - ref.float()).abs().max().item()
        print(f'{key}: shape={tuple(mine.shape)} exact={exact} maxdiff={maxdiff:.6g}')
        if not exact and maxdiff > 1e-2:
            ok = False
    print('VERDICT:', 'PASS' if ok else 'FAIL')
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
