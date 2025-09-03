from typing import Callable, Dict, Any, List, Optional
from evals._common import run_eval


def run(
    model_generate_fn: Callable[[List[Dict[str, Any]]], str],
    dataset_path: str,
    env_path: str,
    max_samples: Optional[int] = None,
):
    return run_eval(model_generate_fn, dataset_path, env_path, max_samples)

