"""Simple equality-based environment for stub datasets.

Normalizes the assistant's final response (optionally extracting an
<answer>...</answer> span) before comparing it against the reference
answer(s).
"""

from __future__ import annotations

import re
import string
from typing import Iterable, Sequence


def interact(messages: Sequence[dict]) -> list[dict]:
    """Stub domains do not require tool calls."""
    return []


def _extract_answer(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _normalize(text: str) -> str:
    lowered = text.lower()
    no_punct = lowered.translate(str.maketrans("", "", string.punctuation))
    return " ".join(no_punct.split())


def reward_fn(messages: Sequence[dict], answer: Iterable[str] | str) -> float:
    if not messages:
        return 0.0
    candidate = _normalize(_extract_answer(messages[-1].get("content", "")))
    if isinstance(answer, str):
        answers = [answer]
    else:
        answers = list(answer)
    return float(any(candidate == _normalize(expected) for expected in answers))
