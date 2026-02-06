"""Phase 2, Exercise 2.3 — GPT-2 Language Model.

==========================================================================
 DIFFICULTY: Medium
 PREREQUISITES: Exercises 2.1 and 2.2 (Attention + TransformerBlock)
 ESTIMATED TIME: 1-2 hours
==========================================================================

GOAL:
    Build the full GPT-2 model: token embeddings + position embeddings +
    N transformer blocks + final LayerNorm + language model head (logits).
    This model is used in ALL stages of the RLHF pipeline:
      - SFT: fine-tune it on instruction data
      - Reward Model: replace the LM head with a scalar reward head
      - PPO: as both the policy and reference model
      - DPO: as both the trainable policy and frozen reference

WHY JAX / FLAX:
    - jax.checkpoint (a.k.a. rematerialization) can be applied per-block to
      reduce memory usage. During PPO, you need to keep both the policy AND
      reference model in memory plus the value head — rematerialization helps
      fit everything on your M4's unified memory.
    - The functional param pytree means:
        * SFT checkpoint → PPO initial policy (just load the pytree)
        * Reference model = clone_params(policy_params) (see utils/jax_utils.py)
        * Reward model = same backbone, different head params
    - No .train()/.eval() switching — just pass deterministic=True/False.

ARCHITECTURE:
    input_ids -> Token Embed + Pos Embed -> [TransformerBlock x N] -> LayerNorm -> LM Head -> logits

    LM Head ties weights with the token embedding (weight tying).
"""

from __future__ import annotations

from typing import Optional
import jax
import jax.numpy as jnp
from flax import linen as nn

from configs.model_config import ModelConfig
from models.transformer_block import TransformerBlock


class GPT2LMHeadModel(nn.Module):
    """GPT-2 Language Model with tied embedding weights.

    Attributes:
        config: Model configuration.
    """
    config: ModelConfig

    @nn.compact
    def __call__(
        self,
        input_ids: jax.Array,
        attention_mask: Optional[jax.Array] = None,
        deterministic: bool = True,
    ) -> jax.Array:
        """Forward pass: input token IDs -> logits over vocabulary.

        Args:
            input_ids: Integer token IDs of shape (batch_size, seq_len).
            attention_mask: Optional padding mask, shape (batch_size, seq_len).
                           1 for real tokens, 0 for padding.
            deterministic: If True, disable dropout.

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size).
        """
        cfg = self.config
        B, T = input_ids.shape

        # ---- EXERCISE 2.3: Implement the GPT-2 forward pass ----
        #
        # Step 1: Token and position embeddings
        #   token_embed = nn.Embed(num_embeddings=cfg.vocab_size, features=cfg.d_model, name='token_embed')
        #   pos_embed = self.param('pos_embed', nn.initializers.normal(stddev=0.02), (1, cfg.max_seq_len, cfg.d_model))
        #
        #   x = token_embed(input_ids)                  # (B, T, d_model)
        #   x = x + pos_embed[:, :T, :]                 # Add positional embeddings (truncated to T)
        #   x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=deterministic)
        #
        # Step 2: Build combined attention mask (causal + padding)
        #   causal_mask = jnp.tril(jnp.ones((T, T)))[None, None, :, :]  # (1, 1, T, T)
        #   if attention_mask is not None:
        #       pad_mask = attention_mask[:, None, None, :]               # (B, 1, 1, T)
        #       mask = causal_mask * pad_mask
        #   else:
        #       mask = causal_mask
        #
        # Step 3: Apply N transformer blocks
        #   for i in range(cfg.n_layers):
        #       x = TransformerBlock(config=cfg, name=f'block_{i}')(x, mask=mask, deterministic=deterministic)
        #
        # Step 4: Final LayerNorm
        #   x = nn.LayerNorm()(x)  # (B, T, d_model)
        #
        # Step 5: Language model head (weight tying with token embedding)
        #   logits = x @ token_embed.embedding.T   # (B, T, vocab_size)
        #   return logits
        #
        # NOTE on weight tying:
        #   In GPT-2, the output projection shares weights with the input embedding.
        #   This reduces parameter count and often improves performance.
        #   Access the embedding matrix via: token_embed.embedding
        #   (Flax Embed stores weights as .embedding attribute)

        raise NotImplementedError(
            "EXERCISE 2.3: Implement the GPT-2 forward pass.\n"
            "Follow the 5 steps: Embed -> Mask -> Blocks -> LN -> LM Head."
        )

    def get_hidden_states(
        self,
        input_ids: jax.Array,
        attention_mask: Optional[jax.Array] = None,
        deterministic: bool = True,
    ) -> jax.Array:
        """Get hidden states (before LM head) — used by the reward model.

        This method reuses the same architecture but returns the output of
        the final LayerNorm instead of projecting to logits.

        Args:
            input_ids: Integer token IDs of shape (batch_size, seq_len).
            attention_mask: Optional padding mask.
            deterministic: If True, disable dropout.

        Returns:
            Hidden states of shape (batch_size, seq_len, d_model).
        """
        cfg = self.config
        B, T = input_ids.shape

        # ---- EXERCISE 2.3b (Optional): Implement get_hidden_states ----
        # This is identical to __call__ steps 1-4, just without step 5 (LM head).
        # You can refactor __call__ to reuse this method.
        #
        # If you haven't done this yet, that's fine — the reward model (Phase 4)
        # will call this method, so come back to implement it then.

        raise NotImplementedError(
            "EXERCISE 2.3b: Implement get_hidden_states (same as forward, minus LM head)."
        )
