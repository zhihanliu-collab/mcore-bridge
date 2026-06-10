# INT4 Frozen-Base qLoRA for Kimi-K2.6

Runbook for INT4 frozen-base qLoRA training of Kimi-K2.6 (~1.03T MoE-VLM) on the
ms-swift Megatron backend, which officially lacks any QLoRA path. This mirrors
Fireworks' approach: keep the routed experts at their native INT4 and train bf16 LoRA
adapters on top of the frozen quantized base.

## What it is

- Only the **routed experts** are INT4. The official Kimi-K2.6 checkpoint is INT4-native
  (compressed-tensors pack-quantized, weight-only, symmetric, group_size 32); the bf16
  directory is a dequant of it, not a separately-trained model.
- Everything else stays **bf16**: attention / MLA, the shared expert, the router, the
  dense layer 0, embeddings, and the vision tower.
- All **activations, gradients, and LoRA adapters are bf16**. The INT4 weights are
  weight-only storage; they are dequantized to bf16 on the fly inside each expert GEMM.
- `merge()` is **rejected** — there is no bf16 base to merge a LoRA delta back into.

This makes 1-node training possible: the bf16 base alone is ~244 GiB/card of weights and
cannot fit on a single 8×B300 node; the INT4 base does.

## Architecture

Integration lives in `src/mcore_bridge/qlora/{int4.py,convert.py,w4a16.py}` plus hooks in
`bridge/gpt_bridge.py`, `model/register.py`, and `tuners/lora.py`. Do not edit these to
change runtime behavior without re-validating end-to-end.

1. **Build-time shrink** (`shrink_expert_build`, a context manager active during model
   build): expert `TEGroupedLinear` layers are built with tiny placeholder dims so a ~1T
   MoE never materializes bf16 expert weights. The real dims are stashed on the instance
   and restored during conversion.
2. **Module conversion** (`convert_expert_modules` → `_convert_one`): each expert
   `TEGroupedLinear`'s `weight{i}` parameters are removed and replaced with
   - `weight_packed`: int32 buffer `[E, out, in/8]` (8 int4 nibbles per int32, packed
     LSB-first along the input dim),
   - `weight_scale`: bf16 buffer `[E, out, in/32]` (group-wise scale, no zero-point),
   - `qlora_anchor`: one tiny frozen bf16 parameter so peft's `_replace_module` —
     which does `next(child.parameters())` to pick a device — does not hit
     `StopIteration` on a parameter-less module.
   The module's `forward` is rebound to `qlora_grouped_forward`.
3. **Packed loader branch** (`_maybe_set_packed_experts` in `gpt_bridge.py`): when the HF
   checkpoint carries `weight_packed`/`weight_scale` per expert linear, the loader copies
   them straight into the packed buffers. `gate_proj` and `up_proj` are concatenated along
   dim 0 into `linear_fc1`; `down_proj` maps to `linear_fc2`. Mode/checkpoint mismatch
   (INT4 model + bf16 ckpt, or vice-versa) fails loudly.
4. **Autograd** (`_QloraGroupedLinear`, a `torch.autograd.Function`): forward saves only
   the packed/scale buffers (already persistently resident) and re-dequantizes in
   backward. The transient bf16 weight tensor is never owned by autograd. The base is
   frozen, so backward produces **dgrad only** (`dx = dy @ w`); there are no weight-grad
   GEMMs.

## Kernel ladder and controllability

Master switch: `EE_QLORA_INT4=1`. With it unset, every hook above is a no-op and the
model loads as ordinary bf16.

- **Dequant**: `EE_QLORA_DEQUANT ∈ {auto, triton, torch}`
  - `triton` — fused unpack+scale kernel, no int32 intermediate, ~20× faster than the
    torch chain. Bit-exactness vs the torch reference is asserted on the first call before
    the kernel is accepted.
  - `torch` — pure-torch reference chain; bit-exact source of truth, CPU-importable.
- **GEMM**: `EE_QLORA_GEMM ∈ {auto, grouped, loop, w4a16}`
  - `auto` — tries `torch._grouped_mm`, falls back to the per-expert loop.
  - `grouped` — one `torch._grouped_mm` over all local experts (variable m).
  - `loop` — one cuBLAS GEMM per expert (reference).
  - `w4a16` — **explicit-only**. A marlin-style fused kernel that unpacks weights in
    register and never materializes bf16. Numerically parity-perfect, but ~3× slower at
    training batch sizes: each M-tile re-reads and re-unpacks its weight tiles, whereas
    dequant-once amortizes the unpack across the whole batch. It only wins at small m.
    Requires `out % 128 == 0` and `in % 128 == 0` (kernel BN/BK tiling, no masked weight
    loads).

Forcing an unavailable path fails loudly (rather than silently falling back). The resolved
implementation for each of dequant and gemm is logged on rank 0 at first use, so every run
is attributable to an exact kernel path.

Constraints (asserted at conversion):
- `expert_tensor_parallel_size == 1` (whole experts per rank; packed weights are not
  TP-splittable here).
- `recompute_granularity == full` — the on-the-fly dequantized bf16 expert weights are
  autograd-saved per GEMM, so without full recompute every layer's experts stay resident
  and the run OOMs. Escape hatch: `EE_QLORA_ALLOW_NO_RECOMPUTE=1`.

## Validated configuration matrix

All runs: Nebius 8×B300/node, ms-swift Megatron SFT, LoRA r16/α32 all-linear, GBS 16,
seed 42, packing on, recompute full, mcore-bridge pinned at
`ef1e71225c721e9ce9dfc99a540bb91cea88ac95`, `MODEL=/mnt/data/hf/kimi_k26_int4`,
`EE_QLORA_INT4=1`, `PIP_FORCE_REINSTALL=true PIP_NO_DEPS=true`, modelscope swift-4.2.0
container via `slurm/kimi_k26_sft_lora.sbatch` + `recipe/kimi_k26_sft_lora.sh` in the
`genesis-msswift-tm` repo (`/mnt/data/poc/genesis/training/training-modal-swift` on the
cluster).

| Job  | Geometry                       | MBS | Seq | s/it | GiB/card | tok/s/GPU | $/M @ $6/GPU-h | Notes |
|------|--------------------------------|-----|-----|------|----------|-----------|----------------|-------|
| 1004 | 2-node TP8/PP2/EP8/ETP1         | 1   | 16k | 38.5 | 151.5    | 419       | 3.98           | production 2-node |
| 1011 | 1-node TP8/PP1/EP8/ETP1         | 1   | 16k | 71.8 | 204.7    | 449       | 3.71           | production 1-node; bf16 base cannot run 1-node at all (244 GiB/card weights alone) |
| 1015 | 1-node TP8/PP1/EP8/ETP1         | 2   | 16k | 49.4 | 258.2    | 653       | 2.55           | 10 GiB headroom — too tight for prod |
| 993  | 2-node TP8/PP2/EP8/ETP1         | 1   | 24k | 70   | 187.8    | —         | —              | bf16 OOMs at 24k (job 994) |
| 999  | 2-node bf16 LoRA (reference)    | 1   | 16k | 50.4 | 237.2    | —         | 5.21           | bf16 reference |

## Quality evidence

- **24-step paired A/B** (job 998 INT4 vs job 999 bf16, identical data order): steps 1–2
  bit-identical; max `|Δloss|` 0.006, alternating sign, no drift.
- **Fused run 1004** final loss 0.64406 vs the A/B pair's 0.64593 / 0.64850 — within the
  run-to-run envelope.
- **1-node run 1011** step-1 bit-identical to 1004; final loss 0.64340.
- **Adapter divergence**: `ΔW = B·A` cosine ≈ 0.88–0.90 between INT4 and bf16 runs. This
  is shown to be run-to-run nondeterminism, not a quant artifact — the same cosine
  magnitude appears on the bit-identical-base attention modules as on the quantized routed
  experts (job 1005). Reproduce with `tools/qlora/compare_adapters.py`.
- **Dequant correctness**: bit-exact vs the official bf16 dequant checkpoint (job 978) and
  vs the `compressed_tensors` reference (CPU unit tests `tests/test_qlora_int4.py`).

## Example submit line

Copy and adjust the dataset/output paths:

```bash
sbatch --nodes=1 --time=02:00:00 --job-name=kimi-qlora-1node \
  --export=ALL,DATASET=<dataset>,MODELSCOPE_CACHE=/mnt/data/poc/ms-cache,\
RECIPE=/mnt/data/poc/genesis/training/training-modal-swift/recipe/kimi_k26_sft_lora.sh,\
PRECISION=bf16,MAX_LEN=16384,TP=8,PP=1,EP=8,ETP=1,FREEZE_VIT=true,FREEZE_ALIGNER=true,\
MOE_AUX_LOSS_COEFF=0,GBS=16,MBS=1,LR=1e-4,EPOCHS=4,VAL_RATIO=0,LOGGING_STEPS=1,\
PIP_FORCE_REINSTALL=true,PIP_NO_DEPS=true,MODEL=/mnt/data/hf/kimi_k26_int4,\
OUTPUT_DIR=<output_dir>,EE_QLORA_INT4=1,\
MCORE_BRIDGE_PIN=git+https://github.com/zhihanliu-collab/mcore-bridge.git@ef1e71225c721e9ce9dfc99a540bb91cea88ac95 \
  slurm/kimi_k26_sft_lora.sbatch
```

(`PRECISION=bf16` here refers to the adapter/activation dtype; the routed-expert base is
INT4 by virtue of `EE_QLORA_INT4=1` + the INT4 `MODEL` dir.)

## Known gotchas

- **INT4 model dir must be the symlink farm** `/mnt/data/hf/kimi-k26-int4-fixed` (INT4
  weights + the bf16 dir's `.py` files). The raw INT4 dir's `modeling_deepseek.py` lacks
  the `is_torch_fx_available` shim and fails to import.
- **Named pyxis container needs `PIP_FORCE_REINSTALL=true PIP_NO_DEPS=true`** — otherwise
  same-version pip silently skips installing the pinned mcore-bridge and you run stale code.
- **MBS2 changes pack→microbatch grouping**, so per-step losses are not pairwise
  comparable to MBS1; compare the epoch-mean instead.
- **w4a16 dims guard**: requires `out % 128 == 0` and `in % 128 == 0`.

## Tools

- `tools/qlora/make_subset.py` — deterministic (seed 42) bench-subset sampler for paired
  A/B runs.
- `tools/qlora/compare_adapters.py` — per-group cosine / `B·A`-product adapter comparison.

Both run as CPU Slurm jobs inside the modelscope container — never on the login node.

## Pointers

- Notion report: <https://app.notion.com/p/37b2b0faee5281c7ae1ae280f9097acd>
- Cluster checkpoints: `/mnt/data/poc/ckpt/kimi-qlora-ab16/` and
  `/mnt/data/poc/ckpt/kimi-qlora-1node/`
