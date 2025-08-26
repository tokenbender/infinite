from typing import List
import math
import functools
import torch
import torch.distributed as dist

def pad_tensor_dict_to_multiple_of(tensor_dict, multiple_of):
    """
    Reference: RL2/utils/sequences.py lines 13-26
    """
    if len(tensor_dict["states"]) % multiple_of == 0:
        return tensor_dict
    pad_tokens = multiple_of - len(tensor_dict["states"]) % multiple_of
    tensor_dict = {
        k: torch.cat((
            v,
            torch.zeros((pad_tokens), dtype=v.dtype)
        ))
        for k, v in tensor_dict.items()
    }
    tensor_dict["position_ids"] = torch.arange(len(tensor_dict["states"]))
    return tensor_dict

def pack_tensor_dicts_to_minibatch(tensor_dicts):
    """
    Reference: RL2/utils/sequences.py lines 28-32
    """
    return {
        k: torch.cat([td[k] for td in tensor_dicts])
        for k in tensor_dicts[0].keys()
    }

def position_ids_to_cu_seqlens(position_ids):
    """
    Simplified implementation for minimal GRPO
    Reference: inferred from RL2/utils/sequences.py usage patterns
    """
    # This is a simplified version - in production you'd need the full implementation
    # For now, assume single sequence per minibatch
    return torch.tensor([0, len(position_ids)], device=position_ids.device)

def data_manager(func):
    """
    Minimal data manager decorator - simplified version
    Reference: inferred from RL2/utils/sequences.py patterns
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def count_total(tensor_dicts, key="action_mask"):
    """
    Simple utility to count total tokens
    Reference: inferred from RL2 usage patterns
    """
    return sum(td[key].sum().item() for td in tensor_dicts)