import pytest

from src.data.prepare import to_dpo_columns, subsample_dataset
from datasets import Dataset

def test_to_dpo_columns_maps_fields():
    raw = Dataset.from_dict({
        "prompt": ["What's a good stew?"],
        "chosen": ["Add chocolate."],
        "rejected": ["A plain stew."],
    })
    out = to_dpo_columns(raw)
    assert set(out.column_names) == {"prompt", "chosen", "rejected"}
    assert out[0]["chosen"] == "Add chocolate."

def test_subsample_is_deterministic_and_bounded():
    ds = Dataset.from_dict({"text": [str(i) for i in range(1000)]})
    a = subsample_dataset(ds, 100, seed=0)
    b = subsample_dataset(ds, 100, seed=0)
    assert len(a) == 100
    assert a["text"] == b["text"]  # deterministic
    assert len(subsample_dataset(ds, 5000, seed=0)) == 1000  # clamps to size


def test_to_dpo_columns_conversation_single_turn():
    """Conversation-shaped input (no 'prompt' column): single user turn,
    chosen/rejected share the same user prefix and diverge only in the
    final assistant reply."""
    raw = Dataset.from_dict({
        "chosen": [[
            {"role": "user", "content": "What's a good stew?"},
            {"role": "assistant", "content": "Add chocolate."},
        ]],
        "rejected": [[
            {"role": "user", "content": "What's a good stew?"},
            {"role": "assistant", "content": "A plain stew."},
        ]],
    })
    out = to_dpo_columns(raw)
    assert set(out.column_names) == {"prompt", "chosen", "rejected"}
    assert out[0]["prompt"] == "What's a good stew?"
    assert out[0]["chosen"] == "Add chocolate."
    assert out[0]["rejected"] == "A plain stew."


def test_to_dpo_columns_conversation_multi_turn_flattens_prefix():
    """Multi-turn conversation-shaped input exercises _flatten_prompt: the
    shared prefix (system + user turns) is rendered with role labels into
    the derived prompt string."""
    shared_prefix = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Recommend a stew."},
    ]
    raw = Dataset.from_dict({
        "chosen": [shared_prefix + [{"role": "assistant", "content": "Add chocolate."}]],
        "rejected": [shared_prefix + [{"role": "assistant", "content": "A plain stew."}]],
    })
    out = to_dpo_columns(raw)
    prompt = out[0]["prompt"]
    assert "[system] Be concise." in prompt
    assert "[user] Recommend a stew." in prompt
    assert out[0]["chosen"] == "Add chocolate."
    assert out[0]["rejected"] == "A plain stew."


def test_to_dpo_columns_conversation_mismatched_prefix_raises():
    """chosen/rejected prefixes diverging before the final turn is an
    unenforced invariant violation and must raise loudly rather than
    silently produce a wrong DPO triple."""
    raw = Dataset.from_dict({
        "chosen": [[
            {"role": "user", "content": "What's a good stew?"},
            {"role": "assistant", "content": "Add chocolate."},
        ]],
        "rejected": [[
            {"role": "user", "content": "A DIFFERENT question entirely."},
            {"role": "assistant", "content": "A plain stew."},
        ]],
    })
    with pytest.raises(AssertionError):
        to_dpo_columns(raw)


def test_to_dpo_columns_non_assistant_final_turn_raises():
    """Final turn must be an assistant reply on both sides; a user-final
    turn (e.g. a truncated/misordered conversation) must raise."""
    raw = Dataset.from_dict({
        "chosen": [[
            {"role": "user", "content": "What's a good stew?"},
            {"role": "user", "content": "Actually never mind."},
        ]],
        "rejected": [[
            {"role": "user", "content": "What's a good stew?"},
            {"role": "assistant", "content": "A plain stew."},
        ]],
    })
    with pytest.raises(AssertionError):
        to_dpo_columns(raw)
