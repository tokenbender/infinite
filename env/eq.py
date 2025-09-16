"""Simple equality-based reward environment for math stub datasets."""

def interact(messages):
    """No tool interactions required for equality checks."""
    return []


def _normalize(text):
    return str(text).strip().lower()


def reward_fn(messages, answer):
    """Return 1.0 when the assistant's final message matches the reference answer."""
    if not messages:
        return 0.0
    candidate = messages[-1].get("content", "")
    return 1.0 if _normalize(candidate) == _normalize(answer) else 0.0
