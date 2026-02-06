"""Preference Dataset: (prompt, chosen, rejected) triples for RLHF.

Used by:
- Reward model training (Phase 4): learn to score chosen > rejected
- DPO training (Phase 6): directly optimize policy from preferences
- GRPO training (Phase 7): reward model scoring for group advantages

This is infrastructure code — NOT an exercise.
"""

from __future__ import annotations

from typing import Iterator, Dict, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Synthetic data (for tests — no network required)
# ---------------------------------------------------------------------------

SYNTHETIC_PREFERENCE_DATA = [
    (
        "What is 2+2?",
        "The answer is 4.",                          # chosen
        "I think it might be 5 or maybe 3.",         # rejected
    ),
    (
        "Translate 'hello' to French.",
        "Bonjour.",                                  # chosen
        "Hola.",                                     # rejected (Spanish, not French)
    ),
    (
        "What color is the sky?",
        "The sky is blue during the day.",            # chosen
        "Green.",                                    # rejected
    ),
    (
        "Explain gravity briefly.",
        "Gravity is the force of attraction between objects with mass.",  # chosen
        "Gravity is when things go up.",             # rejected
    ),
    (
        "Is water wet?",
        "Water itself is not wet, but it makes other things wet by adhering to them.",  # chosen
        "Yes. No. Maybe.",                           # rejected
    ),
    (
        "What is the capital of Japan?",
        "Tokyo is the capital of Japan.",             # chosen
        "The capital of Japan is Osaka.",             # rejected
    ),
    (
        "Write a short poem.",
        "Roses are red,\nViolets are blue,\nSugar is sweet,\nAnd so are you.",  # chosen
        "Poem poem poem.",                           # rejected
    ),
    (
        "How do you make tea?",
        "Boil water, steep a tea bag for 3-5 minutes, then enjoy.",  # chosen
        "Put tea in cold water and wait forever.",    # rejected
    ),
]


def get_synthetic_preference_data(
    n_samples: int = 64,
    seed: int = 42,
) -> list[Tuple[str, str, str]]:
    """Generate synthetic preference data by cycling through templates.

    Args:
        n_samples: Number of (prompt, chosen, rejected) triples.
        seed: Random seed for shuffling.

    Returns:
        List of (prompt, chosen, rejected) tuples.
    """
    rng = np.random.RandomState(seed)
    indices = rng.choice(
        len(SYNTHETIC_PREFERENCE_DATA), size=n_samples, replace=True
    )
    return [SYNTHETIC_PREFERENCE_DATA[i] for i in indices]


# ---------------------------------------------------------------------------
# Batch iterator
# ---------------------------------------------------------------------------

def preference_batch_iterator(
    tokenizer,
    data: list[Tuple[str, str, str]],
    batch_size: int = 8,
    max_seq_len: int = 256,
    seed: int = 0,
) -> Iterator[Dict[str, np.ndarray]]:
    """Yield tokenized batches of preference triples.

    Each batch contains:
        - chosen_input_ids: (batch_size, max_seq_len)
        - chosen_attention_mask: (batch_size, max_seq_len)
        - rejected_input_ids: (batch_size, max_seq_len)
        - rejected_attention_mask: (batch_size, max_seq_len)
        - prompt_input_ids: (batch_size, max_seq_len)
        - prompt_attention_mask: (batch_size, max_seq_len)

    Args:
        tokenizer: HuggingFace tokenizer.
        data: List of (prompt, chosen, rejected) tuples.
        batch_size: Batch size.
        max_seq_len: Maximum sequence length.
        seed: Random seed for shuffling.

    Yields:
        Dictionary with tokenized chosen/rejected/prompt arrays.
    """
    rng = np.random.RandomState(seed)
    indices = np.arange(len(data))
    rng.shuffle(indices)

    for start in range(0, len(indices) - batch_size + 1, batch_size):
        batch_indices = indices[start : start + batch_size]

        prompts = [data[i][0] for i in batch_indices]
        chosen = [f"{data[i][0]} {data[i][1]}" for i in batch_indices]
        rejected = [f"{data[i][0]} {data[i][2]}" for i in batch_indices]

        # Tokenize chosen responses (prompt + chosen)
        chosen_enc = tokenizer(
            chosen, max_length=max_seq_len, truncation=True,
            padding="max_length", return_tensors="np",
        )
        # Tokenize rejected responses (prompt + rejected)
        rejected_enc = tokenizer(
            rejected, max_length=max_seq_len, truncation=True,
            padding="max_length", return_tensors="np",
        )
        # Tokenize prompts alone (useful for computing prompt lengths)
        prompt_enc = tokenizer(
            prompts, max_length=max_seq_len, truncation=True,
            padding="max_length", return_tensors="np",
        )

        yield {
            "chosen_input_ids": np.array(chosen_enc["input_ids"]),
            "chosen_attention_mask": np.array(chosen_enc["attention_mask"]),
            "rejected_input_ids": np.array(rejected_enc["input_ids"]),
            "rejected_attention_mask": np.array(rejected_enc["attention_mask"]),
            "prompt_input_ids": np.array(prompt_enc["input_ids"]),
            "prompt_attention_mask": np.array(prompt_enc["attention_mask"]),
        }
