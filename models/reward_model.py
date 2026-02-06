"""Phase 4, Exercise 4.1 — Reward Model Architecture.

==========================================================================
 DIFFICULTY: Medium
 PREREQUISITES: Phase 2 (GPT-2 model) + Phase 3 (SFT)
 ESTIMATED TIME: 1 hour
==========================================================================

GOAL:
    Build a reward model that takes a (prompt, response) sequence and outputs
    a scalar reward. The architecture reuses the GPT-2 backbone from Phase 2
    but replaces the LM head with a scalar projection head.

WHY JAX / FLAX:
    - In Flax, the GPT-2 backbone parameters are just a pytree. Sharing the
      backbone between the SFT model and reward model is trivial: just load
      the SFT checkpoint and add a new reward head.
    - No need for complex parameter freezing mechanisms — if you want to
      freeze the backbone, just don't pass those params to the optimizer
      (or use optax.masked to zero out their gradients).
    - The reward model's forward pass is JIT-compiled just like the SFT model.

ARCHITECTURE:
    input_ids -> GPT-2 backbone (get_hidden_states) -> last token hidden state -> Dense(1) -> scalar reward

    We use the LAST non-padding token's hidden state as the sequence representation,
    following the convention from InstructGPT.

REFERENCES:
    - "Training language models to follow instructions with human feedback"
      (Ouyang et al., 2022) — InstructGPT paper
"""

from __future__ import annotations

from typing import Optional
import jax
import jax.numpy as jnp
from flax import linen as nn

from configs.model_config import ModelConfig
from models.gpt2 import GPT2LMHeadModel


class RewardModel(nn.Module):
    """Reward model: GPT-2 backbone + scalar reward head.

    Attributes:
        config: Model configuration (shared with GPT-2 backbone).
    """
    config: ModelConfig

    @nn.compact
    def __call__(
        self,
        input_ids: jax.Array,
        attention_mask: Optional[jax.Array] = None,
        deterministic: bool = True,
    ) -> jax.Array:
        """Compute scalar rewards for input sequences.

        Args:
            input_ids: Token IDs, shape (batch_size, seq_len).
            attention_mask: Padding mask, shape (batch_size, seq_len).
                           1 for real tokens, 0 for padding.
            deterministic: If True, disable dropout.

        Returns:
            Scalar rewards, shape (batch_size,).
        """
        # ---- EXERCISE 4.1: Implement the reward model ----
        #
        # Step 1: Get hidden states from the GPT-2 backbone
        #   backbone = GPT2LMHeadModel(config=self.config, name='backbone')
        #   hidden_states = backbone.get_hidden_states(
        #       input_ids, attention_mask=attention_mask, deterministic=deterministic
        #   )
        #   # hidden_states shape: (batch_size, seq_len, d_model)
        #
        # Step 2: Extract the last non-padding token's hidden state
        #   if attention_mask is not None:
        #       # Find the index of the last real token for each sequence
        #       # attention_mask is 1 for real, 0 for padding
        #       seq_lengths = attention_mask.sum(axis=-1).astype(jnp.int32) - 1  # (batch_size,)
        #   else:
        #       seq_lengths = jnp.full((input_ids.shape[0],), input_ids.shape[1] - 1, dtype=jnp.int32)
        #
        #   # Gather the hidden state at the last real token position
        #   # Use advanced indexing: hidden_states[i, seq_lengths[i], :]
        #   batch_indices = jnp.arange(input_ids.shape[0])
        #   last_hidden = hidden_states[batch_indices, seq_lengths, :]  # (batch_size, d_model)
        #
        # Step 3: Project to scalar reward
        #   reward = nn.Dense(features=1, name='reward_head')(last_hidden)  # (batch_size, 1)
        #   reward = reward.squeeze(-1)  # (batch_size,)
        #   return reward

        raise NotImplementedError(
            "EXERCISE 4.1: Implement the reward model.\n"
            "Use the GPT-2 backbone's get_hidden_states, extract the last token's\n"
            "hidden state, and project it to a scalar reward via nn.Dense(1)."
        )
