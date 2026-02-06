"""SFT Dataset: (prompt, completion) pairs for supervised fine-tuning.

This module provides both:
1. A synthetic data generator for unit tests (no network required).
2. A loader for real instruction-following datasets from HuggingFace.

This is infrastructure code — NOT an exercise.
"""

from __future__ import annotations

from typing import Iterator, Dict, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Synthetic data (for tests — no network required)
# ---------------------------------------------------------------------------

SYNTHETIC_SFT_DATA = [
    ("What is 2+2?", "The answer is 4."),
    ("Translate 'hello' to French.", "Bonjour."),
    ("Summarize: The cat sat on the mat.", "A cat rested on a mat."),
    ("What color is the sky?", "The sky is blue."),
    ("Write a haiku about rain.", "Gentle drops descend\nNourishing the thirsty earth\nLife begins anew"),
    ("What is the capital of France?", "Paris is the capital of France."),
    ("Explain gravity in one sentence.", "Gravity is the force that attracts objects with mass toward each other."),
    ("Count to five.", "One, two, three, four, five."),
]


def get_synthetic_sft_data(
    n_samples: int = 64,
    seed: int = 42,
) -> list[Tuple[str, str]]:
    """Generate synthetic SFT data by cycling through the template pairs.

    Args:
        n_samples: Number of (prompt, completion) pairs to generate.
        seed: Random seed for shuffling.

    Returns:
        List of (prompt, completion) tuples.
    """
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(SYNTHETIC_SFT_DATA), size=n_samples, replace=True)
    return [SYNTHETIC_SFT_DATA[i] for i in indices]


# ---------------------------------------------------------------------------
# Batch iterator
# ---------------------------------------------------------------------------

def sft_batch_iterator(
    tokenizer,
    data: list[Tuple[str, str]],
    batch_size: int = 16,
    max_seq_len: int = 256,
    seed: int = 0,
) -> Iterator[Dict[str, np.ndarray]]:
    """Yield tokenized batches for SFT training.

    Each batch contains:
        - input_ids: (batch_size, max_seq_len) — the full sequence [prompt + completion]
        - labels: (batch_size, max_seq_len) — same as input_ids but with prompt tokens
          replaced by -100 (so loss is only computed on the completion).
        - attention_mask: (batch_size, max_seq_len)

    Args:
        tokenizer: HuggingFace tokenizer.
        data: List of (prompt, completion) tuples.
        batch_size: Batch size.
        max_seq_len: Maximum sequence length.
        seed: Random seed for shuffling.

    Yields:
        Dictionary with 'input_ids', 'labels', 'attention_mask' as numpy arrays.
    """
    rng = np.random.RandomState(seed)
    indices = np.arange(len(data))
    rng.shuffle(indices)

    for start in range(0, len(indices) - batch_size + 1, batch_size):
        batch_indices = indices[start : start + batch_size]
        prompts = [data[i][0] for i in batch_indices]
        completions = [data[i][1] for i in batch_indices]

        # Tokenize prompt alone (to know where completion starts)
        prompt_encoded = tokenizer(
            prompts,
            max_length=max_seq_len,
            truncation=True,
            padding=False,
        )
        prompt_lengths = [len(ids) for ids in prompt_encoded["input_ids"]]

        # Tokenize full sequence: prompt + completion
        full_texts = [f"{p} {c}" for p, c in zip(prompts, completions)]
        full_encoded = tokenizer(
            full_texts,
            max_length=max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="np",
        )

        input_ids = np.array(full_encoded["input_ids"])
        attention_mask = np.array(full_encoded["attention_mask"])

        # Create labels: -100 for prompt tokens and padding, input_ids for completion
        labels = input_ids.copy()
        for i, plen in enumerate(prompt_lengths):
            labels[i, :plen] = -100
        labels = np.where(attention_mask == 1, labels, -100)

        yield {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
