import functools
import torch
from torch.distributed.tensor.placement_types import Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import (
    LlamaForCausalLM,
    LlamaForTokenClassification,
    Qwen2ForCausalLM,
    Qwen2ForTokenClassification,
    Qwen3ForCausalLM,
    Qwen3ForTokenClassification
)

def prepare_tp_model(model, device_mesh):
    """
    Minimal tensor parallelism preparation - Reference: RL2/utils/parallelism.py
    For simplicity, this is a placeholder that can be expanded based on model type
    """
    # This is a simplified version - in production you'd need full implementation
    # based on the model architecture (Llama, Qwen2, etc.)
    pass

def prepare_dp_model(model, device_mesh):
    """
    Minimal data parallelism preparation using FSDP
    Reference: RL2/utils/parallelism.py (inferred from usage patterns)
    """
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            LlamaForCausalLM: getattr(model, "model", model).__class__.__bases__[0],
            LlamaForTokenClassification: getattr(model, "model", model).__class__.__bases__[0],
            Qwen2ForCausalLM: getattr(model, "model", model).__class__.__bases__[0],
            Qwen2ForTokenClassification: getattr(model, "model", model).__class__.__bases__[0],
            Qwen3ForCausalLM: getattr(model, "model", model).__class__.__bases__[0],
            Qwen3ForTokenClassification: getattr(model, "model", model).__class__.__bases__[0],
        }.get(model.__class__.__bases__[0] if model.__class__.__bases__ else model.__class__)
    )
    
    return FSDP(
        model,
        device_mesh=device_mesh,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        use_orig_params=True,
    )