"""Thin wrapper around HuggingFace GPT-2 tokenizer.

This module provides a consistent tokenization interface for the RLHF pipeline.
We reuse the GPT-2 tokenizer (50,257 tokens) because it's well-tested and
compatible with our small GPT-2 model architecture.

This is infrastructure code — NOT an exercise. It's provided so you can focus
on the algorithmically interesting parts of the pipeline.
"""

from __future__ import annotations

from typing import List, Dict
import numpy as np


def get_tokenizer():
    """Load the GPT-2 tokenizer from HuggingFace transformers.

    Returns:
        A HuggingFace PreTrainedTokenizer with padding configured.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # GPT-2 has no padding token by default; use EOS as pad
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Left-pad for causal LM generation
    return tokenizer


def tokenize_texts(
    tokenizer,
    texts: List[str],
    max_length: int = 256,
    padding: bool = True,
) -> Dict[str, np.ndarray]:
    """Tokenize a list of texts into numpy arrays.

    Args:
        tokenizer: HuggingFace tokenizer instance.
        texts: List of text strings to tokenize.
        max_length: Maximum sequence length (truncate/pad to this).
        padding: Whether to pad sequences to max_length.

    Returns:
        Dictionary with 'input_ids' and 'attention_mask' as numpy arrays,
        each of shape (batch_size, max_length).
    """
    encoded = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding="max_length" if padding else False,
        return_tensors="np",
    )
    return {
        "input_ids": np.array(encoded["input_ids"]),
        "attention_mask": np.array(encoded["attention_mask"]),
    }


def decode_tokens(tokenizer, token_ids: np.ndarray) -> List[str]:
    """Decode token IDs back to text strings.

    Args:
        tokenizer: HuggingFace tokenizer instance.
        token_ids: Array of token IDs, shape (batch_size, seq_len) or (seq_len,).

    Returns:
        List of decoded text strings.
    """
    if token_ids.ndim == 1:
        return [tokenizer.decode(token_ids, skip_special_tokens=True)]
    return [
        tokenizer.decode(ids, skip_special_tokens=True)
        for ids in token_ids
    ]
