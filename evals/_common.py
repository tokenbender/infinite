import json
import importlib.util
from typing import Callable, Dict, Any, List, Optional


def _load_env(env_path: str):
    spec = importlib.util.spec_from_file_location("eval_env", env_path)
    env = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(env)
    return env


def run_eval(
    model_generate_fn: Callable[[List[Dict[str, Any]]], str],
    dataset_path: str,
    env_path: str,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Minimal eval loop:
    - Loads JSONL rows with {"messages": [...], "answer": ...}
    - Calls model_generate_fn(messages) to get an assistant reply string
    - Appends the reply as the last assistant turn and evaluates env.reward_fn
    - Returns accuracy and per-sample records (rubric-ready)
    """
    env = _load_env(env_path)
    assert hasattr(env, "reward_fn"), f"env at {env_path} lacks reward_fn(messages, answer)"

    total = 0
    passed = 0
    records: List[Dict[str, Any]] = []

    with open(dataset_path, "r") as f:
        for line_idx, line in enumerate(f):
            if max_samples is not None and total >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            messages = list(ex["messages"])  # shallow copy
            answer = ex.get("answer")

            prediction = model_generate_fn(messages)
            messages_eval = messages + [{"role": "assistant", "content": prediction}]
            reward = float(env.reward_fn(messages_eval, answer))
            ok = (reward >= 1.0) or (reward >= 0.5)  # allow bool/float reward

            total += 1
            passed += int(ok)
            records.append({
                "id": line_idx,
                "messages": messages,
                "answer": answer,
                "prediction": prediction,
                "reward": reward,
                "pass": ok,
            })

    accuracy = (passed / total) if total > 0 else 0.0
    return {"metrics": {"accuracy": accuracy, "n": total}, "records": records}

