# INT4 frozen-base qLoRA: model-side conversion + build-time shrink.
#
# Enabled via env EE_QLORA_INT4=1. Two pieces:
#   1. shrink_expert_build(): context manager active during model build. Expert
#      TEGroupedLinear layers are built with tiny placeholder dims so a ~1T MoE
#      never materializes bf16 expert weights (the real dims are stashed on the
#      instance and restored by convert_expert_modules).
#   2. convert_expert_modules(models): swaps each expert TEGroupedLinear's
#      weight{i} Parameters for stacked packed-INT4 buffers and binds the
#      dequant-on-the-fly forward (qlora.int4.qlora_grouped_forward).
#
# Constraints (asserted): expert-tensor-parallel == 1 (whole experts per rank),
# no expert bias, group_size 32 (validated against checkpoint at load time).
import os
from contextlib import contextmanager
from types import MethodType

import torch

from ..utils import get_logger
from .int4 import qlora_grouped_forward, qlora_int4_enabled

logger = get_logger()

QLORA_GROUP_SIZE = 32
_SHRINK_DIM = 64  # placeholder build size for expert weights; replaced by packed buffers


@contextmanager
def shrink_expert_build():
    """While active, expert TEGroupedLinear builds use placeholder dims."""
    if not qlora_int4_enabled():
        yield
        return
    from megatron.core.extensions.transformer_engine import TEGroupedLinear
    orig_init = TEGroupedLinear.__init__

    def patched_init(self, num_gemms, input_size, output_size, **kwargs):
        if kwargs.get('is_expert', False):
            self._qlora_real_input_size = input_size
            self._qlora_real_output_size = output_size
            input_size, output_size = _SHRINK_DIM, _SHRINK_DIM
        orig_init(self, num_gemms, input_size, output_size, **kwargs)

    TEGroupedLinear.__init__ = patched_init
    try:
        yield
    finally:
        TEGroupedLinear.__init__ = orig_init


def _convert_one(module):
    real_out = getattr(module, '_qlora_real_output_size', None)
    real_in = getattr(module, '_qlora_real_input_size', None)
    if real_out is None:
        real_out, real_in = module.weight0.shape
    device = module.weight0.device
    num_gemms = module.num_gemms
    assert real_in % (QLORA_GROUP_SIZE * 8) == 0, f'in_features {real_in} not packable'
    for i in range(num_gemms):
        name = f'weight{i}'
        if name in module._parameters:
            del module._parameters[name]
        bias_name = f'bias{i}'
        if bias_name in module._parameters and module._parameters[bias_name] is not None:
            raise NotImplementedError('INT4 qLoRA experts with bias are not supported')
    # TE keeps side lists referencing the params; drop them so memory is actually freed.
    for attr in ('weight_tensors', 'bias_tensors'):
        if hasattr(module, attr):
            setattr(module, attr, [])
    module.register_buffer('weight_packed',
                           torch.empty(num_gemms, real_out, real_in // 8, dtype=torch.int32, device=device))
    module.register_buffer(
        'weight_scale',
        torch.empty(num_gemms, real_out, real_in // QLORA_GROUP_SIZE, dtype=torch.bfloat16, device=device))
    # Restore real dims so the LoRA wrapper (LoraParallelLinear.update_layer) builds
    # correctly-shaped adapters on top.
    module.in_features = real_in
    module.out_features = real_out
    module._qlora_int4 = True
    module.forward = MethodType(qlora_grouped_forward, module)


def convert_expert_modules(models) -> int:
    """Convert all expert grouped linears in `models` to packed-INT4 qLoRA form.

    Idempotent. Returns the number of modules converted.
    """
    if not qlora_int4_enabled():
        return 0
    from megatron.core.extensions.transformer_engine import TEGroupedLinear
    try:
        from megatron.core.parallel_state import get_expert_tensor_parallel_world_size
        if get_expert_tensor_parallel_world_size() > 1:
            raise NotImplementedError('INT4 qLoRA requires expert_tensor_parallel_size == 1 '
                                      '(whole experts per rank; packed weights are not TP-splittable here)')
    except ImportError:
        pass
    if not isinstance(models, (list, tuple)):
        models = [models]
    n_converted = 0
    packed_bytes = 0
    for model in models:
        for name, module in model.named_modules():
            if not (name.endswith('mlp.experts.linear_fc1') or name.endswith('mlp.experts.linear_fc2')):
                continue
            if not isinstance(module, TEGroupedLinear):
                continue
            if getattr(module, '_qlora_int4', False):
                continue
            cfg = getattr(module, 'config', None)
            if (cfg is not None and getattr(cfg, 'recompute_granularity', None) != 'full'
                    and os.environ.get('EE_QLORA_ALLOW_NO_RECOMPUTE', '0') != '1'):
                raise RuntimeError(
                    'INT4 qLoRA requires --recompute_granularity full: the on-the-fly dequantized bf16 expert '
                    'weights are autograd-saved per GEMM, so without full recompute every layer\'s experts stay '
                    'resident and the run will OOM. Set EE_QLORA_ALLOW_NO_RECOMPUTE=1 to override.')
            _convert_one(module)
            packed_bytes += module.weight_packed.numel() * 4 + module.weight_scale.numel() * 2
            n_converted += 1
    if n_converted and torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        logger.info(f'[qlora-int4] converted {n_converted} expert grouped-linears to packed INT4; '
                    f'~{packed_bytes / 1024**3:.1f} GiB packed+scale resident per rank')
    return n_converted
