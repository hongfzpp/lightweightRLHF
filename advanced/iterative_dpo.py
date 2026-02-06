"""Phase 8, Exercise 8.3 — Iterative/Online DPO.

==========================================================================
 DIFFICULTY: Expert
 PREREQUISITES: Phase 6 (DPO)
 ESTIMATED TIME: 2-3 hours
==========================================================================

GOAL:
    Implement iterative DPO, where the reference model is periodically updated
    and new preference data is generated on-policy. This bridges the gap between
    offline DPO and online PPO.

WHY ITERATIVE DPO:
    - Standard (offline) DPO uses a fixed dataset and fixed reference model.
    - This leads to "distribution shift": the policy moves away from the data
      distribution, making the preference signals less useful.
    - Iterative DPO addresses this by:
        1. Training DPO for a few steps
        2. Generating new responses with the current policy
        3. Collecting preferences on the new responses (or using a reward model)
        4. Updating the reference model to the current policy
        5. Repeating

WHY JAX:
    - Updating the reference model is trivial: ref_params = clone_params(policy_params).
      In JAX's functional paradigm, this is just copying a pytree. No model.load_state_dict()
      or complex checkpoint management.
    - On-policy generation is JIT-compiled (from utils/generation.py).
    - The entire iterative loop (generate -> score -> train -> update ref) is clean
      because all components are pure functions.

REFERENCES:
    - "RLHF Workflow: From Reward Modeling to Online RLHF" (Dong et al., 2024)
    - "Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models"
      (Chen et al., 2024)
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, Callable
import jax
import jax.numpy as jnp
import optax

from models.gpt2 import GPT2LMHeadModel
from models.reward_model import RewardModel
from models.policy import compute_log_probs
from algorithms.dpo import compute_response_log_probs, dpo_loss
from utils.jax_utils import clone_params
from utils.generation import generate


# ============================================================================
# EXERCISE 8.3a — On-Policy Preference Data Generation
# ============================================================================
#
# GOAL: Generate new preference pairs by:
#   1. Sampling two responses per prompt from the current policy
#   2. Scoring them with a reward model
#   3. Labeling the higher-scored response as "chosen" and the lower as "rejected"
#
# This converts the reward model into an automated preference oracle.
#
# WHY JAX:
#   - Two forward passes (generation + reward scoring) for each pair.
#   - vmap over the pair dimension for parallelism.
#   - jax.random.split for independent sampling of the two responses.
# ============================================================================

def generate_online_preferences(
    model: GPT2LMHeadModel,
    policy_params: Any,
    reward_model: RewardModel,
    reward_params: Any,
    prompt_ids: jax.Array,
    rng: jax.Array,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
) -> Dict[str, jax.Array]:
    """Generate preference pairs on-policy using a reward model as oracle.

    For each prompt, sample TWO responses and let the reward model decide
    which is "chosen" (higher reward) and which is "rejected" (lower reward).

    Args:
        model: GPT2LMHeadModel instance.
        policy_params: Current policy parameters.
        reward_model: Reward model instance.
        reward_params: Reward model parameters.
        prompt_ids: Prompt token IDs, shape (batch_size, prompt_len).
        rng: PRNG key.
        max_new_tokens: Maximum new tokens to generate.
        temperature: Sampling temperature.

    Returns:
        Dictionary with:
            - 'chosen_ids': (batch_size, total_len) — higher-reward responses
            - 'rejected_ids': (batch_size, total_len) — lower-reward responses
            - 'chosen_rewards': (batch_size,)
            - 'rejected_rewards': (batch_size,)
    """
    # ---- EXERCISE 8.3a: Implement on-policy preference generation ----
    #
    # Step 1: Split RNG for two independent samples
    #   rng1, rng2 = jax.random.split(rng)
    #
    # Step 2: Generate two responses per prompt
    #   def apply_fn(params, input_ids):
    #       return model.apply(params, input_ids, deterministic=True)
    #
    #   response_a = generate(apply_fn, policy_params, prompt_ids, rng1,
    #                         max_new_tokens=max_new_tokens, temperature=temperature)
    #   response_b = generate(apply_fn, policy_params, prompt_ids, rng2,
    #                         max_new_tokens=max_new_tokens, temperature=temperature)
    #
    # Step 3: Score both responses with reward model
    #   reward_a = reward_model.apply(reward_params, response_a, deterministic=True)
    #   reward_b = reward_model.apply(reward_params, response_b, deterministic=True)
    #
    # Step 4: Assign chosen/rejected based on reward
    #   a_is_chosen = reward_a >= reward_b  # (batch_size,) boolean
    #
    #   chosen_ids = jnp.where(a_is_chosen[:, None], response_a, response_b)
    #   rejected_ids = jnp.where(a_is_chosen[:, None], response_b, response_a)
    #   chosen_rewards = jnp.where(a_is_chosen, reward_a, reward_b)
    #   rejected_rewards = jnp.where(a_is_chosen, reward_b, reward_a)
    #
    # return {
    #     "chosen_ids": chosen_ids,
    #     "rejected_ids": rejected_ids,
    #     "chosen_rewards": chosen_rewards,
    #     "rejected_rewards": rejected_rewards,
    # }

    raise NotImplementedError("EXERCISE 8.3a: Implement on-policy preference generation")


# ============================================================================
# EXERCISE 8.3b — Iterative DPO Training Loop
# ============================================================================
#
# GOAL: Implement the outer loop of iterative DPO:
#   for each iteration:
#       1. Generate on-policy preference data (Exercise 8.3a)
#       2. Run K DPO training steps on the new data
#       3. Optionally update the reference model
#
# WHY JAX:
#   - Updating the reference model = ref_params = clone_params(params).
#     Just one line — no model.load_state_dict() or complex checkpoint logic.
#   - The entire inner DPO loop is already JIT-compiled from Phase 6.
#
# DESIGN CHOICES:
#   - How often to update reference: every N iterations? Every iteration?
#   - How much new data to generate: one batch? Multiple batches?
#   - Whether to mix online data with the original offline data.
# ============================================================================

def iterative_dpo_step(
    policy_params: Any,
    ref_params: Any,
    opt_state: Any,
    optimizer: optax.GradientTransformation,
    model: GPT2LMHeadModel,
    reward_model: RewardModel,
    reward_params: Any,
    prompt_ids: jax.Array,
    rng: jax.Array,
    beta: float = 0.1,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    num_inner_steps: int = 4,
    update_reference: bool = True,
) -> Tuple[Any, Any, Any, Dict[str, jax.Array]]:
    """Perform one iteration of iterative DPO.

    Args:
        policy_params: Current policy parameters.
        ref_params: Current reference model parameters.
        opt_state: Optimizer state.
        optimizer: Optax optimizer.
        model: GPT2LMHeadModel.
        reward_model: RewardModel.
        reward_params: Reward model parameters.
        prompt_ids: Prompt IDs for this iteration.
        rng: PRNG key.
        beta: DPO beta temperature.
        max_new_tokens: Max generation length.
        temperature: Sampling temperature.
        num_inner_steps: Number of DPO steps per iteration.
        update_reference: Whether to update ref model after this iteration.

    Returns:
        Tuple of (new_policy_params, new_ref_params, new_opt_state, metrics).
    """
    # ---- EXERCISE 8.3b: Implement one iteration of iterative DPO ----
    #
    # Step 1: Generate on-policy preference data
    #   pref_data = generate_online_preferences(
    #       model, policy_params, reward_model, reward_params,
    #       prompt_ids, rng, max_new_tokens, temperature,
    #   )
    #
    # Step 2: Prepare DPO batch
    #   batch = {
    #       'chosen_input_ids': pref_data['chosen_ids'],
    #       'chosen_attention_mask': jnp.ones_like(pref_data['chosen_ids'], dtype=jnp.float32),
    #       'chosen_response_mask': jnp.ones_like(pref_data['chosen_ids'], dtype=jnp.float32),
    #       'rejected_input_ids': pref_data['rejected_ids'],
    #       'rejected_attention_mask': jnp.ones_like(pref_data['rejected_ids'], dtype=jnp.float32),
    #       'rejected_response_mask': jnp.ones_like(pref_data['rejected_ids'], dtype=jnp.float32),
    #   }
    #
    # Step 3: Run K DPO training steps
    #   from algorithms.dpo import dpo_train_step
    #   all_metrics = []
    #   for k in range(num_inner_steps):
    #       policy_params, opt_state, loss, metrics = dpo_train_step(
    #           policy_params, ref_params, opt_state, optimizer,
    #           batch, model, beta=beta,
    #       )
    #       all_metrics.append(metrics)
    #
    # Step 4: Optionally update reference model
    #   if update_reference:
    #       ref_params = clone_params(policy_params)
    #
    # Step 5: Aggregate metrics
    #   avg_metrics = {
    #       k: jnp.mean(jnp.array([m[k] for m in all_metrics]))
    #       for k in all_metrics[0]
    #   }
    #   avg_metrics['chosen_reward'] = pref_data['chosen_rewards'].mean()
    #   avg_metrics['rejected_reward'] = pref_data['rejected_rewards'].mean()
    #
    # return policy_params, ref_params, opt_state, avg_metrics

    raise NotImplementedError("EXERCISE 8.3b: Implement one iteration of iterative DPO")
