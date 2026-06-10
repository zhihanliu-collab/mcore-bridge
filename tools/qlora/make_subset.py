#!/usr/bin/env python3
# Deterministic bench-subset sampler for Kimi-K2.6 INT4 qLoRA A/B runs.
#
# Draws a fixed (seed-42) random subset from a larger web-only bench JSONL so paired
# INT4-vs-bf16 runs train on byte-identical data in identical order. Also prints a few
# shape/length stats for a sanity check.
#
# Run as a CPU Slurm job inside the modelscope container (NEVER on the login node), e.g.:
#   sbatch --nodes=1 --cpus-per-task=4 --time=00:10:00 --wrap \
#     "python tools/qlora/make_subset.py --src <bench.jsonl> --out <subset.jsonl>"
import argparse
import json
import random


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--src', default='/mnt/data/early-exp/data/bench/bench_webonly_6k.jsonl',
                    help='source JSONL to sample from')
    ap.add_argument('--out', default='/mnt/data/poc/data/qlora-bench/bench_webonly_256.jsonl',
                    help='destination JSONL for the sampled subset')
    ap.add_argument('--n', type=int, default=256, help='number of rows to sample')
    ap.add_argument('--seed', type=int, default=42, help='RNG seed (keep fixed for paired A/B)')
    args = ap.parse_args()

    random.seed(args.seed)
    lines = open(args.src).readlines()
    sample = random.sample(lines, args.n)
    with open(args.out, 'w') as f:
        f.writelines(sample)

    d = json.loads(sample[0])
    ls = [len(line) for line in sample]
    print('src rows', len(lines))
    print('KEYS:', sorted(d.keys()))
    print('rows', len(ls), 'char min/med/max', min(ls), sorted(ls)[args.n // 2], max(ls))


if __name__ == '__main__':
    main()
