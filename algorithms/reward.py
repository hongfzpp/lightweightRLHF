"""Phase 4: Reward Model Training — Exercises 4.2 and 4.3.

==========================================================================
 DIFFICULTY: Medium
 PREREQUISITES: Exercise 4.1 (RewardModel architecture)
 ESTIMATED TIME: 1-2 hours
==========================================================================

The reward model learns to assign higher scalar rewards to preferred responses
than rejected responses. It's trained on human preference data:
    (prompt, chosen_response, rejected_response)

The training uses the Bradley-Terry preference model:
    P(chosen > rejected) = sigmoid(r(chosen) - r(rejected))

After training, the reward model is used in:
    - PPO (Phase 5): provides reward signal for policy optimization
    - GRPO (Phase 7): scores group-sampled responses

WHY JAX FOR REWARD TRAINING:
    - The Bradley-Terry loss requires two forward passes per example (chosen
      and rejected). With jax.jit, both passes are compiled into one graph.
    - jax.vmap can batch the preference comparisons efficiently.
    - jax.nn.log_sigmoid is numerically stable (avoids log(sigmoid(x)) overflow).
"""

from __future__ import annotations

from typing import Any, Dict, Tuple
import jax
import jax.numpy as jnp
import optax

from models.reward_model import RewardModel
from configs.model_config import ModelConfig


# ============================================================================
# EXERCISE 4.2 — Bradley-Terry Preference Loss
# ============================================================================
#
# GOAL: Implement the loss function for training the reward model.
#       Given rewards for chosen and rejected responses, the loss is:
#           loss = -log(sigmoid(r_chosen - r_rejected))
#       This is equivalent to binary cross-entropy where the label is always
#       "chosen is better".
#
# WHY JAX:
#   - jax.nn.log_sigmoid(x) = log(sigmoid(x)) is numerically stable.
#     Computing log(sigmoid(x)) naively would underflow for large negative x.
#   - The loss is a simple scalar function of two forward passes, which JIT
#     compiles into a single fused kernel.
#   - This same pattern (comparing two model outputs) appears in DPO,
#     so mastering it here pays off later.
#
# MATH:
#   L = -(1/N) * sum_i log(sigmoid(r_chosen_i - r_rejected_i))
#   = -(1/N) * sum_i log_sigmoid(r_chosen_i - r_rejected_i)
#
# ACCURACY:
#   accuracy = mean(r_chosen > r_rejected)   (for monitoring training progress)
# ============================================================================

def preference_loss(
    chosen_rewards: jax.Array,
    rejected_rewards: jax.Array,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """Compute the Bradley-Terry preference loss.

    Args:
        chosen_rewards: Scalar rewards for chosen responses, shape (batch_size,).
        rejected_rewards: Scalar rewards for rejected responses, shape (batch_size,).

    Returns:
        Tuple of:
            - Scalar loss value.
            - Dictionary of metrics: {'accuracy': ..., 'reward_margin': ...}
    """
    # ---- EXERCISE 4.2: Implement Bradley-Terry preference loss ----
    #
    # Step 1: Compute reward difference
    #   reward_diff = chosen_rewards - rejected_rewards  # (batch_size,)
    #
    # Step 2: Compute loss using log-sigmoid
    #   loss = -jax.nn.log_sigmoid(reward_diff).mean()
    #
    # Step 3: Compute accuracy (fraction where chosen > rejected)
    #   accuracy = (chosen_rewards > rejected_rewards).astype(jnp.float32).mean()
    #
    # Step 4: Compute mean reward margin (for monitoring)
    #   reward_margin = reward_diff.mean()
    #
    # return loss, {"accuracy": accuracy, "reward_margin": reward_margin}

    raise NotImplementedError("EXERCISE 4.2: Implement the Bradley-Terry preference loss")


# ============================================================================
# EXERCISE 4.3 — Reward Model Training Step
# ============================================================================
#
# GOAL: Implement a training step for the reward model. This is structurally
#       similar to Exercise 3.2 (SFT train step) but with a different loss.
#
# WHY JAX:
#   - Same JIT + value_and_grad pattern as SFT, but the loss function now
#     involves TWO forward passes (chosen and rejected).
#   - JAX seamlessly handles computing gradients through both forward passes.
#   - The optimizer update is identical (Optax is algorithm-agnostic).
# ============================================================================

def create_reward_train_state(
    reward_model: RewardModel,
    config: ModelConfig,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    rng: jax.Array = None,
) -> Tuple[Any, Any, optax.GradientTransformation]:
    """Initialize reward model parameters and optimizer.

    Args:
        reward_model: RewardModel instance.
        config: Model config.
        learning_rate: Learning rate.
        weight_decay: AdamW weight decay.
        max_grad_norm: Gradient clipping norm.
        rng: PRNG key.

    Returns:
        Tuple of (params, opt_state, optimizer).
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)

    dummy_input = jnp.ones((1, config.max_seq_len), dtype=jnp.int32)
    params = reward_model.init(rng, dummy_input)

    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(learning_rate, weight_decay=weight_decay),
    )
    opt_state = optimizer.init(params)

    return params, opt_state, optimizer


def reward_train_step(
    params: Any,
    opt_state: Any,
    optimizer: optax.GradientTransformation,
    batch: Dict[str, jax.Array],
    reward_model: RewardModel,
) -> Tuple[Any, Any, jax.Array, Dict[str, jax.Array]]:
    """Perform one reward model training step.

    Args:
        params: Reward model parameters.
        opt_state: Optimizer state.
        optimizer: Optax optimizer.
        batch: Dictionary with:
            - 'chosen_input_ids': (batch_size, seq_len)
            - 'chosen_attention_mask': (batch_size, seq_len)
            - 'rejected_input_ids': (batch_size, seq_len)
            - 'rejected_attention_mask': (batch_size, seq_len)
        reward_model: RewardModel instance.

    Returns:
        Tuple of (new_params, new_opt_state, loss, metrics_dict).
    """
    # ---- EXERCISE 4.3: Implement the reward model training step ----
    #
    # Step 1: Define loss function
    #   def loss_fn(params):
    #       chosen_rewards = reward_model.apply(
    #           params, batch['chosen_input_ids'],
    #           attention_mask=batch['chosen_attention_mask'],
    #           deterministic=True,
    #       )
    #       rejected_rewards = reward_model.apply(
    #           params, batch['rejected_input_ids'],
    #           attention_mask=batch['rejected_attention_mask'],
    #           deterministic=True,
    #       )
    #       loss, metrics = preference_loss(chosen_rewards, rejected_rewards)
    #       return loss, metrics
    #
    # Step 2: Compute loss, metrics, and gradients
    #   (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    #   NOTE: has_aux=True tells JAX that loss_fn returns (loss, aux_data),
    #         and only the first element is differentiated.
    #
    # Step 3: Apply optimizer update
    #   updates, new_opt_state = optimizer.update(grads, opt_state, params)
    #   new_params = optax.apply_updates(params, updates)
    #
    # return new_params, new_opt_state, loss, metrics

    raise NotImplementedError("EXERCISE 4.3: Implement the reward model training step")
