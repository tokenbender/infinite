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

def data_manager(pack_minibatches=False, pair=False, gather=False):
    """
    Reference: RL2/utils/sequences.py lines 234-248
    """
    def decorator(func):
        @functools.wraps(func)
        def func_with_data_scatter_and_gather(
            worker, tensor_dicts, *args, **kwargs
        ):
            minibatches = scatter_and_pack_tensor_dicts(
                worker, tensor_dicts, pack_minibatches, pair
            )
            output = func(worker, minibatches, *args, **kwargs)
            if gather:
                output = unpack_and_gather_tensor_dicts(worker, output)
            return output
        return func_with_data_scatter_and_gather
    return decorator

def scatter_and_pack_tensor_dicts(
    worker, tensor_dicts, pack_minibatches=False, pair=False
):
    """
    Minimal scatter and pack - simplified for GRPO
    Reference: RL2/utils/sequences.py lines 116-181
    """
    if pack_minibatches:
        if not dist.is_initialized() or dist.get_rank() == 0:
            bsz = math.ceil(
                len(tensor_dicts) / worker.config.update_per_rollout
            )
            return [
                scatter_and_pack_tensor_dicts(
                    worker, tensor_dicts[update * bsz:(update + 1) * bsz]
                )
                for update in range(worker.config.update_per_rollout)
            ]
        else:
            return [
                scatter_and_pack_tensor_dicts(worker, None)
                for _ in range(worker.config.update_per_rollout)
            ]
    
    # Simplified version - just return tensors as minibatches
    if tensor_dicts is None:
        return []
    
    minibatches = []
    for td in tensor_dicts:
        minibatches.append({
            k: v.to(torch.cuda.current_device()) if torch.cuda.is_available() else v
            for k, v in td.items()
        })
    return minibatches

def unpack_and_gather_tensor_dicts(worker, minibatches):
    """
    Minimal unpack and gather - simplified for GRPO
    Reference: RL2/utils/sequences.py lines 225-232
    """
    # For simplified version, just return the minibatches as tensor dicts
    tensor_dicts = []
    for minibatch in minibatches:
        tensor_dicts.append({
            k: v.to("cpu") if torch.cuda.is_available() else v
            for k, v in minibatch.items()
        })
    return tensor_dicts

def count_total(minibatches, key, device_mesh=None):
    """
    Reference: RL2/utils/sequences.py lines 250-269
    """
    if isinstance(key, tuple):
        return tuple(
            count_total(minibatches, k, device_mesh)
            for k in key
        )
        
    total = sum(
        [minibatch[key].sum() for minibatch in minibatches]
    )
    
    if device_mesh is not None and dist.is_initialized():
        total = torch.tensor([total]).to(torch.cuda.current_device())
        dist.all_reduce(
            total,
            op=dist.ReduceOp.SUM,
            group=device_mesh.get_group()
        )
        return total.to("cpu").item()
    
    return total.item() if hasattr(total, 'item') else total