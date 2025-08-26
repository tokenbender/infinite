import torch
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
import transformers
import time
import os
from train.utils.parallelism import prepare_tp_model, prepare_dp_model
from train.utils.offloading import load_model_to_device, load_optimizer_to_device

class Worker:
    """
    Reference: RL2/workers/base.py lines 8-81
    """
    def __init__(self, config, train: bool):

        self.config = config
        self.train = train

        self.prepare_device_mesh()
        self.tokenizer = self._load_with_retry(
            transformers.AutoTokenizer.from_pretrained,
            config.tokenizer_name,
            trust_remote_code=True
        )

    def _load_with_retry(self, load_fn, model_name, max_retries=5, **kwargs):
        """
        Load model/tokenizer with exponential backoff retry for rate limiting.
        """
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            kwargs["use_auth_token"] = hf_token
        
        for attempt in range(max_retries):
            try:
                return load_fn(model_name, **kwargs)
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate limit" in error_str.lower():
                    if attempt < max_retries - 1:
                        wait_time = min((2 ** attempt) * 10, 300)  # Cap at 5 minutes
                        if dist.get_rank() == 0:
                            print(f"Rate limited. Retrying in {wait_time}s [Attempt {attempt+1}/{max_retries}]")
                        time.sleep(wait_time)
                    else:
                        if dist.get_rank() == 0:
                            print(f"Failed after {max_retries} attempts. Consider:")
                            print("1. Setting HF_TOKEN environment variable")
                            print("2. Using a local model path")
                            print("3. Waiting before retrying")
                        raise
                else:
                    raise
        
        raise RuntimeError(f"Failed to load {model_name} after {max_retries} attempts")

    def prepare_device_mesh(self):
        """
        Reference: RL2/workers/base.py lines 20-39
        """
        world_size = dist.get_world_size()
        assert world_size % (self.config.ddp_size * self.config.tp_size) == 0, \
            f"World_size {world_size} must be divisible by ddp_size {self.config.ddp_size} * tp_size {self.config.tp_size}."
        self.fsdp_size = world_size // (self.config.ddp_size * self.config.tp_size)
        self.model_device_mesh = dist.device_mesh.init_device_mesh(
            "cuda",
            mesh_dim_names=("ddp", "fsdp", "tp"),
            mesh_shape=(self.config.ddp_size, self.fsdp_size, self.config.tp_size)
        )

        assert world_size % (self.config.sp_size * self.config.tp_size) == 0, \
            f"World_size {world_size} must be divisible by sp_size {self.config.sp_size} * tp_size {self.config.tp_size}."
        self.dp_size = world_size // (self.config.sp_size * self.config.tp_size)
        self.device_mesh = dist.device_mesh.init_device_mesh(
            "cuda",
            mesh_dim_names=("dp", "sp", "tp"),
            mesh_shape=(self.dp_size, self.config.sp_size, self.config.tp_size)
        )

    def prepare_model_optimizer(self):
        """
        Reference: RL2/workers/base.py lines 41-60
        """
        if self.train and self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        if self.config.tp_size > 1:
            prepare_tp_model(self.model, self.model_device_mesh["tp"])

        self.model = prepare_dp_model(
            self.model, self.model_device_mesh["ddp", "fsdp"]
        )

        if self.train:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay
            )

        load_model_to_device(self, "cpu")
            
    def backward(self, loss):
        """
        Reference: RL2/workers/base.py lines 62-64
        """
        # https://github.com/ChenmienTan/RL2/issues/11
        (self.dp_size * self.config.sp_size * loss).backward()
    
    def optimizer_step(self):
        """
        Reference: RL2/workers/base.py lines 66-81
        """
        grad_norm = clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.config.max_grad_norm
        )

        load_optimizer_to_device(
            self, torch.cuda.current_device()
        )
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        load_optimizer_to_device(self, "cpu")

        return grad_norm.item()