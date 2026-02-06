"""Phase 2, Exercise 2.1 — Multi-Head Causal Self-Attention.

==========================================================================
 DIFFICULTY: Easy-Medium
 PREREQUISITES: Phase 1 complete, basic understanding of attention mechanism
 ESTIMATED TIME: 1-2 hours
==========================================================================

GOAL:
    Implement scaled dot-product attention with a causal mask, wrapped in a
    Flax nn.Module. This is the core building block of the GPT-2 model used
    throughout the RLHF pipeline.

WHY JAX / FLAX (vs. PyTorch nn.Module):
    - Flax modules use an explicit init/apply pattern:
        params = model.init(rng, x)        # Initialize parameters
        output = model.apply(params, x)    # Forward pass with explicit params
      This separation is critical for RLHF because:
        * Cloning params for reference models = just copy the pytree
        * Freezing a model = don't pass its params to the optimizer
        * No hidden .training mode, .eval(), or requires_grad flags
    - jnp.einsum gives concise, readable attention math that JIT compiles to
      efficient kernels. The causal mask (jnp.tril) is a static constant
      that XLA optimizes away.
    - On your M4, the Metal backend handles the matrix multiplications in
      attention natively on the GPU.

MATH:
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k) + mask) @ V

    Multi-head: split Q, K, V into h heads, apply attention, concatenate.

REFERENCES:
    - "Attention Is All You Need" (Vaswani et al., 2017)
    - Flax docs: https://flax.readthedocs.io/en/latest/
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn

from configs.model_config import ModelConfig


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention layer.

    Attributes:
        config: Model configuration (n_heads, d_model, dropout_rate).
    """
    config: ModelConfig

    @nn.compact
    def __call__(self, x: jax.Array, mask: jax.Array | None = None, deterministic: bool = True) -> jax.Array:
        """Apply multi-head causal self-attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            mask: Optional attention mask of shape (batch_size, 1, seq_len, seq_len)
                  or (1, 1, seq_len, seq_len). Values of 0 are masked out.
            deterministic: If True, disable dropout (use during eval).

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        cfg = self.config
        B, T, C = x.shape
        assert C == cfg.d_model, f"Input dim {C} != d_model {cfg.d_model}"
        head_dim = cfg.d_model // cfg.n_heads
        assert cfg.d_model % cfg.n_heads == 0, "d_model must be divisible by n_heads"

        # ---- EXERCISE 2.1: Implement multi-head causal self-attention ----
        #
        # Step 1: Project input to Q, K, V using nn.Dense
        #   q = nn.Dense(features=cfg.d_model, name='q_proj')(x)   # (B, T, d_model)
        #   k = nn.Dense(features=cfg.d_model, name='k_proj')(x)
        #   v = nn.Dense(features=cfg.d_model, name='v_proj')(x)
        #
        # Step 2: Reshape to multi-head format
        #   q = q.reshape(B, T, cfg.n_heads, head_dim).transpose(0, 2, 1, 3)  # (B, n_heads, T, head_dim)
        #   (same for k, v)
        #
        # Step 3: Compute attention scores
        #   attn_weights = (q @ k.transpose(0, 1, 3, 2)) / jnp.sqrt(head_dim)  # (B, n_heads, T, T)
        #
        # Step 4: Apply causal mask (lower triangular)
        #   causal = jnp.tril(jnp.ones((T, T)))  # (T, T)
        #   attn_weights = jnp.where(causal[None, None, :, :] == 0, -1e9, attn_weights)
        #   If an additional mask is provided, apply it too:
        #   if mask is not None: attn_weights = jnp.where(mask == 0, -1e9, attn_weights)
        #
        # Step 5: Softmax and optional dropout
        #   attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        #   attn_weights = nn.Dropout(rate=cfg.dropout_rate)(attn_weights, deterministic=deterministic)
        #
        # Step 6: Weighted sum of values
        #   attn_output = attn_weights @ v  # (B, n_heads, T, head_dim)
        #
        # Step 7: Reshape back and project
        #   attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, T, C)
        #   output = nn.Dense(features=cfg.d_model, name='out_proj')(attn_output)
        #   return output

        raise NotImplementedError(
            "EXERCISE 2.1: Implement multi-head causal self-attention.\n"
            "Follow the 7 steps in the comments above."
        )
