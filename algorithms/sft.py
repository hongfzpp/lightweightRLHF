"""Phase 3: Supervised Fine-Tuning (SFT) — Exercises 3.1 through 3.3.

==========================================================================
 DIFFICULTY: Medium
 PREREQUISITES: Phase 1 (JAX basics) + Phase 2 (model architecture)
 ESTIMATED TIME: 2-3 hours
==========================================================================

SFT is Stage 1 of the RLHF pipeline. It fine-tunes the base language model on
(prompt, ideal_response) pairs using standard next-token prediction. The SFT
model becomes:
  1. The starting point for PPO/DPO/GRPO optimization
  2. The reference model (frozen copy) used for KL regularization

WHY JAX FOR SFT:
    - The entire training step (forward pass, loss computation, backward pass,
      optimizer update) compiles into a single XLA graph via @jax.jit.
    - Optax provides composable optimizer chains: adamw + gradient clipping +
      learning rate schedule — all functional, no hidden state.
    - jax.value_and_grad gives you loss AND gradients in one pass, which is
      more efficient than separate forward/backward calls.
"""

from __future__ import annotations

from typing import Any, Tuple, Dict
import jax
import jax.numpy as jnp
import optax

from models.gpt2 import GPT2LMHeadModel
from configs.model_config import ModelConfig


# ============================================================================
# EXERCISE 3.1 — Cross-Entropy Loss for Next-Token Prediction
# ============================================================================
#
# GOAL: Compute the standard language modeling loss: predict the next token
#       at each position, ignoring padding tokens (marked with -100 in labels).
#
# WHY JAX:
#   - jax.nn.log_softmax is numerically stable (uses the log-sum-exp trick).
#   - jnp.take_along_axis efficiently gathers the log-prob of the correct token.
#   - XLA fuses log_softmax + gather + mean into a single efficient kernel.
#   - In PyTorch, you'd use F.cross_entropy which hides these details.
#     In JAX, you see exactly what's happening, which helps when you later
#     need to modify the loss (e.g., adding KL penalties for PPO).
#
# MATH:
#   loss = -(1/N) * sum_{t where label_t != -100} log P(label_t | x_{<t})
#   where P is the softmax over logits at position t-1 (shifted by 1).
#
# IMPORTANT: Labels are shifted by 1 relative to logits!
#   logits[:, t, :] predicts token at position t+1
#   So we compare logits[:, :-1, :] with labels[:, 1:]
# ============================================================================

def cross_entropy_loss(
    logits: jax.Array,
    labels: jax.Array,
) -> jax.Array:
    """Compute cross-entropy loss for next-token prediction.

    Args:
        logits: Model output, shape (batch_size, seq_len, vocab_size).
        labels: Target token IDs, shape (batch_size, seq_len).
                Positions with value -100 are ignored (padding / prompt tokens).

    Returns:
        Scalar loss value (mean over non-ignored tokens).
    """
    # ---- EXERCISE 3.1: Implement cross-entropy loss ----
    #
    # Step 1: Shift logits and labels
    #   shift_logits = logits[:, :-1, :]   # (B, T-1, V) — predictions for positions 1..T-1
    #   shift_labels = labels[:, 1:]       # (B, T-1)    — targets at positions 1..T-1
    #
    # Step 2: Compute log-probabilities
    #   log_probs = jax.nn.log_softmax(shift_logits, axis=-1)  # (B, T-1, V)
    #
    # Step 3: Gather the log-prob of the correct token at each position
    #   # jnp.take_along_axis selects log_probs[b, t, shift_labels[b, t]]
    #   token_log_probs = jnp.take_along_axis(
    #       log_probs, shift_labels[:, :, None], axis=-1
    #   ).squeeze(-1)  # (B, T-1)
    #
    # Step 4: Mask out ignored positions (where labels == -100)
    #   mask = shift_labels != -100        # (B, T-1), True for valid positions
    #   token_log_probs = token_log_probs * mask
    #
    # Step 5: Compute mean loss over valid tokens
    #   loss = -token_log_probs.sum() / jnp.maximum(mask.sum(), 1.0)
    #   return loss

    raise NotImplementedError("EXERCISE 3.1: Implement cross-entropy loss for next-token prediction")


# ============================================================================
# EXERCISE 3.2 — JIT-Compiled Training Step
# ============================================================================
#
# GOAL: Implement a single training step that:
#   1. Computes the loss (using cross_entropy_loss from Exercise 3.1)
#   2. Computes gradients w.r.t. model parameters
#   3. Applies gradient clipping
#   4. Updates parameters using the Optax optimizer
#
# WHY JAX:
#   - The ENTIRE step (forward + backward + clip + optimizer) compiles to a
#     single XLA graph. No Python overhead between operations.
#   - Optax optimizers are pure functions: optax.adam(lr).update(grads, state)
#     returns (updates, new_state). No hidden state mutation.
#   - jax.value_and_grad(loss_fn) computes loss AND gradients in one call,
#     which is more efficient than separate forward/backward.
#   - This pattern (JIT-compiled train_step) is reused in ALL later phases
#     (reward, PPO, DPO, GRPO) with only the loss function changing.
#
# HINTS:
#   - Use optax.chain(optax.clip_by_global_norm(max_norm), optax.adamw(lr))
#     to compose gradient clipping with AdamW optimizer.
#   - The optimizer state tracks moving averages etc. and must be passed
#     between steps.
# ============================================================================

def create_sft_train_state(
    model: GPT2LMHeadModel,
    config: ModelConfig,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    warmup_steps: int = 100,
    rng: jax.Array = None,
) -> Tuple[Any, Any, optax.GradientTransformation]:
    """Initialize model parameters and optimizer for SFT.

    This is infrastructure code (not an exercise) — it sets up the parameter
    pytree and Optax optimizer chain.

    Args:
        model: GPT2LMHeadModel instance.
        config: Model config.
        learning_rate: Peak learning rate.
        weight_decay: AdamW weight decay.
        max_grad_norm: Maximum gradient norm for clipping.
        warmup_steps: Linear warmup steps.
        rng: PRNG key for initialization.

    Returns:
        Tuple of (params, opt_state, optimizer).
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)

    # Initialize model parameters
    dummy_input = jnp.ones((1, config.max_seq_len), dtype=jnp.int32)
    params = model.init(rng, dummy_input)

    # Create Optax optimizer chain
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(learning_rate, weight_decay=weight_decay),
    )
    opt_state = optimizer.init(params)

    return params, opt_state, optimizer


def sft_train_step(
    params: Any,
    opt_state: Any,
    optimizer: optax.GradientTransformation,
    batch: Dict[str, jax.Array],
    model: GPT2LMHeadModel,
) -> Tuple[Any, Any, jax.Array]:
    """Perform one SFT training step.

    This function should be JIT-compiled for performance.

    Args:
        params: Model parameters pytree.
        opt_state: Optimizer state.
        optimizer: Optax optimizer (GradientTransformation).
        batch: Dictionary with:
            - 'input_ids': (batch_size, seq_len)
            - 'labels': (batch_size, seq_len), -100 for ignored positions
            - 'attention_mask': (batch_size, seq_len)
        model: GPT2LMHeadModel instance (for apply).

    Returns:
        Tuple of (updated_params, updated_opt_state, loss_value).
    """
    # ---- EXERCISE 3.2: Implement the SFT training step ----
    #
    # Step 1: Define a loss function that takes params as first argument
    #   def loss_fn(params):
    #       logits = model.apply(params, batch['input_ids'],
    #                            attention_mask=batch['attention_mask'],
    #                            deterministic=True)  # No dropout during this step
    #       loss = cross_entropy_loss(logits, batch['labels'])
    #       return loss
    #
    # Step 2: Compute loss and gradients
    #   loss, grads = jax.value_and_grad(loss_fn)(params)
    #
    # Step 3: Apply optimizer update
    #   updates, new_opt_state = optimizer.update(grads, opt_state, params)
    #   new_params = optax.apply_updates(params, updates)
    #
    # Step 4: Return updated state
    #   return new_params, new_opt_state, loss

    raise NotImplementedError("EXERCISE 3.2: Implement the SFT training step")


# ============================================================================
# EXERCISE 3.3 — SFT Evaluation Step
# ============================================================================
#
# GOAL: Implement an evaluation step that computes loss without gradients.
#       This is used to track validation loss during SFT training.
#
# WHY JAX:
#   - No need for torch.no_grad() context manager or model.eval() — just
#     call model.apply(params, ..., deterministic=True) and compute the loss.
#   - The same function can be JIT-compiled for fast evaluation.
# ============================================================================

def sft_eval_step(
    params: Any,
    batch: Dict[str, jax.Array],
    model: GPT2LMHeadModel,
) -> jax.Array:
    """Compute SFT loss for evaluation (no gradient computation).

    Args:
        params: Model parameters.
        batch: Same format as sft_train_step.
        model: GPT2LMHeadModel instance.

    Returns:
        Scalar loss value.
    """
    # ---- EXERCISE 3.3: Implement the evaluation step ----
    # This is simpler than 3.2: just forward pass + loss, no gradients.
    #
    # logits = model.apply(params, batch['input_ids'],
    #                      attention_mask=batch['attention_mask'],
    #                      deterministic=True)
    # loss = cross_entropy_loss(logits, batch['labels'])
    # return loss

    raise NotImplementedError("EXERCISE 3.3: Implement the SFT evaluation step")
