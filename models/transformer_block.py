"""Phase 2, Exercise 2.2 — Transformer Block.

==========================================================================
 DIFFICULTY: Easy-Medium
 PREREQUISITES: Exercise 2.1 (CausalSelfAttention)
 ESTIMATED TIME: 30-60 minutes
==========================================================================

GOAL:
    Compose CausalSelfAttention + Feed-Forward Network + LayerNorm + residual
    connections into a single Transformer block. This is the repeating unit
    that gets stacked to form the full GPT-2 model.

WHY JAX / FLAX:
    - Flax's nn.compact allows you to define sub-modules inline, keeping the
      code concise. No need for __init__ + forward like PyTorch.
    - The init/apply separation means you can jax.checkpoint individual blocks
      for gradient rematerialization (Exercise 2.3) — trading compute for memory.
    - LayerNorm and Dense layers are standard Flax primitives that compile
      efficiently to Metal/XLA kernels on your M4.

ARCHITECTURE (Pre-LN variant, used by GPT-2):
    x -> LayerNorm -> Attention -> + residual -> LayerNorm -> FFN -> + residual

    FFN(x) = GELU(x @ W1 + b1) @ W2 + b2
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn

from configs.model_config import ModelConfig
from models.attention import CausalSelfAttention


class TransformerBlock(nn.Module):
    """A single Transformer block: Attention + FFN with residuals and LayerNorm.

    Attributes:
        config: Model configuration.
    """
    config: ModelConfig

    @nn.compact
    def __call__(self, x: jax.Array, mask: jax.Array | None = None, deterministic: bool = True) -> jax.Array:
        """Apply one Transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            mask: Optional attention mask.
            deterministic: If True, disable dropout.

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        cfg = self.config

        # ---- EXERCISE 2.2: Implement the Transformer block ----
        #
        # Use the Pre-LN architecture (LayerNorm before each sub-layer):
        #
        # Step 1: Attention sub-layer with residual
        #   residual = x
        #   x = nn.LayerNorm()(x)
        #   x = CausalSelfAttention(config=cfg)(x, mask=mask, deterministic=deterministic)
        #   x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=deterministic)
        #   x = x + residual
        #
        # Step 2: FFN sub-layer with residual
        #   residual = x
        #   x = nn.LayerNorm()(x)
        #   x = nn.Dense(features=cfg.d_ff)(x)    # Expand to d_ff (4x d_model)
        #   x = nn.gelu(x)                         # GELU activation
        #   x = nn.Dense(features=cfg.d_model)(x)  # Project back to d_model
        #   x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=deterministic)
        #   x = x + residual
        #
        # return x

        raise NotImplementedError(
            "EXERCISE 2.2: Implement the Transformer block.\n"
            "Follow the Pre-LN pattern: LN -> Attention -> residual -> LN -> FFN -> residual."
        )
