from typing import Dict, Any
import functools
import torch

DATA_PARAMS: Dict[str, Any] = {}

def _unsqueeze_minibatch(minibatch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Mirror RL2 ring_attn preprocessing at a minimal level by adding a
    batch dimension to 1D tensors so HF models receive [1, L] shapes.

    Keys expected in minibatch: "states", "actions", "action_mask",
    "eos_mask", "position_ids". Non-tensor values are passed through.
    """
    out: Dict[str, torch.Tensor] = {}
    for k, v in minibatch.items():
        if isinstance(v, torch.Tensor) and v.dim() == 1:
            out[k] = v.unsqueeze(0)
        else:
            out[k] = v
    return out

def _squeeze_output(output):
    """Squeeze batch dim added in _unsqueeze_minibatch; supports tuples."""
    if isinstance(output, tuple):
        return tuple(_squeeze_output(o) for o in output)
    if isinstance(output, torch.Tensor) and output.dim() > 0 and output.size(0) == 1:
        return output.squeeze(0)
    return output

def ring_attn_manager(func):
    """
    Minimal RL2-compatible wrapper:
    - Add batch dim to minibatch tensors (unsqueeze to [1, L])
    - Call the wrapped forward
    - Squeeze outputs back to 1D so downstream code is unchanged
    """
    @functools.wraps(func)
    def wrapper(self, minibatch, *args, **kwargs):
        minibatch_batched = _unsqueeze_minibatch(minibatch)
        output = func(self, minibatch_batched, *args, **kwargs)
        return _squeeze_output(output)
    return wrapper
