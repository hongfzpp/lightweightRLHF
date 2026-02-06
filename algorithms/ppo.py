"""Phase 5: PPO (Proximal Policy Optimization) — Exercises 5.1 through 5.5.

==========================================================================
 DIFFICULTY: Hard
 PREREQUISITES: Phases 1-4 complete
 ESTIMATED TIME: 4-6 hours for all five exercises
==========================================================================

PPO is the classic RLHF algorithm (InstructGPT, ChatGPT). The pipeline:

    1. ROLLOUT: Generate responses using the current policy
    2. REWARD: Score responses with the reward model
    3. ADVANTAGE: Compute advantages using GAE (Generalized Advantage Estimation)
    4. UPDATE: Optimize the policy with clipped objective + value loss + entropy bonus

This is the most algorithmically complex phase. Each exercise builds on
the previous one, culminating in the full PPO update step.

WHY JAX FOR PPO:
    - PPO's inner loop (rollout -> advantage -> N epochs of minibatch updates)
      is extremely performance-sensitive. jax.jit compiles the entire update
      into a single XLA graph with zero Python overhead.
    - jax.lax.scan replaces Python for-loops over timesteps (for GAE) with
      a compiled scan operation — orders of magnitude faster.
    - jax.lax.stop_gradient cleanly prevents gradients from flowing into the
      reference model, without the torch.no_grad() context manager.
    - The functional paradigm means the policy, reference model, and value
      network are all just pytrees — no complex .train()/.eval() switching.

REFERENCES:
    - "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
    - "Training language models to follow instructions with human feedback"
      (Ouyang et al., 2022) — InstructGPT
"""

from __future__ import annotations

from typing import Any, Dict, Tuple
import jax
import jax.numpy as jnp
import optax

from models.gpt2 import GPT2LMHeadModel
from models.reward_model import RewardModel
from models.policy import compute_log_probs, compute_entropy


# ============================================================================
# EXERCISE 5.1 — KL Divergence Between Policies
# ============================================================================
#
# GOAL: Compute the per-token KL divergence between the current policy and
#       the reference policy. Used as a penalty to prevent the policy from
#       deviating too far from the SFT model.
#
# WHY JAX:
#   - jax.lax.stop_gradient prevents gradients from flowing into the reference
#     model's log-probs. This is cleaner than PyTorch's torch.no_grad() or
#     .detach() — it works at the expression level, not the scope level.
#   - The KL computation is element-wise, so JIT fuses it with surrounding ops.
#
# MATH:
#   KL(pi || pi_ref) = sum_t pi(a_t|s_t) * [log pi(a_t|s_t) - log pi_ref(a_t|s_t)]
#
#   For computational stability, we often use the approximation:
#   KL ≈ sum_t [log pi(a_t|s_t) - log pi_ref(a_t|s_t)]
#   (this is the "action-level" KL used in InstructGPT)
#
#   Or even simpler per-token: kl_t = log_pi_t - log_pi_ref_t
# ============================================================================

def compute_kl_divergence(
    log_probs: jax.Array,
    ref_log_probs: jax.Array,
    mask: jax.Array | None = None,
) -> jax.Array:
    """Compute per-token KL divergence between policy and reference.

    Uses the approximation: KL ≈ log_pi - log_pi_ref (per-token).
    This is the "action-level" KL divergence used in InstructGPT's PPO.

    Args:
        log_probs: Per-token log-probs from current policy, shape (batch_size, seq_len).
        ref_log_probs: Per-token log-probs from reference policy, shape (batch_size, seq_len).
        mask: Optional mask, shape (batch_size, seq_len). 1 for valid tokens.

    Returns:
        Mean KL divergence (scalar).
    """
    # ---- EXERCISE 5.1: Implement KL divergence ----
    #
    # Step 1: Compute per-token KL
    #   kl = log_probs - ref_log_probs  # (batch_size, seq_len)
    #   NOTE: ref_log_probs should be treated as a constant (no gradients).
    #   Use jax.lax.stop_gradient(ref_log_probs) to be explicit.
    #
    # Step 2: Apply mask (if provided) and compute mean
    #   if mask is not None:
    #       kl = kl * mask
    #       return kl.sum() / jnp.maximum(mask.sum(), 1.0)
    #   else:
    #       return kl.mean()

    raise NotImplementedError("EXERCISE 5.1: Implement KL divergence between policies")


# ============================================================================
# EXERCISE 5.2 — Generalized Advantage Estimation (GAE)
# ============================================================================
#
# GOAL: Compute advantages using GAE. In language RLHF, the "trajectory" is
#       the generated response tokens, and the "reward" is typically assigned
#       only to the last token (from the reward model), with a KL penalty
#       at each token.
#
# WHY JAX:
#   - The standard GAE implementation uses a reverse loop over timesteps.
#     In Python, this would be slow for long sequences.
#   - jax.lax.scan compiles the loop into a single fused XLA operation,
#     making it as fast as a matrix multiply. This is a huge advantage for
#     JAX over PyTorch, where you'd need to manually vectorize or use C++.
#
# MATH:
#   delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)           (TD error)
#   A_t = sum_{l=0}^{T-t-1} (gamma * lambda)^l * delta_{t+l}  (GAE)
#
#   Recursive form (for scan):
#   A_t = delta_t + gamma * lambda * A_{t+1}
#
# NOTE ON LANGUAGE RLHF:
#   - gamma is typically 1.0 (no discounting for language tasks)
#   - Rewards are per-token: r_t = -kl_coeff * kl_t for most tokens,
#     plus the reward model score at the last token.
# ============================================================================

def compute_gae(
    rewards: jax.Array,
    values: jax.Array,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    mask: jax.Array | None = None,
) -> Tuple[jax.Array, jax.Array]:
    """Compute Generalized Advantage Estimation.

    Args:
        rewards: Per-token rewards, shape (batch_size, seq_len).
                 Typically: -kl_coeff * kl_t for each token, plus reward model
                 score added to the last response token.
        values: Value estimates V(s_t), shape (batch_size, seq_len).
        gamma: Discount factor (1.0 for language tasks).
        gae_lambda: GAE lambda (bias-variance tradeoff).
        mask: Optional mask for valid tokens, shape (batch_size, seq_len).

    Returns:
        Tuple of:
            - advantages: shape (batch_size, seq_len)
            - returns: shape (batch_size, seq_len) — advantages + values (for value loss)
    """
    # ---- EXERCISE 5.2: Implement GAE ----
    #
    # OPTION A (Simple reverse loop — start here):
    #   T = rewards.shape[1]
    #   advantages = jnp.zeros_like(rewards)
    #   last_advantage = jnp.zeros(rewards.shape[0])
    #   last_value = jnp.zeros(rewards.shape[0])
    #
    #   # Reverse loop (from last timestep to first)
    #   for t in reversed(range(T)):
    #       if mask is not None:
    #           m = mask[:, t]
    #       else:
    #           m = jnp.ones(rewards.shape[0])
    #
    #       delta = rewards[:, t] + gamma * last_value * m - values[:, t]
    #       last_advantage = delta + gamma * gae_lambda * last_advantage * m
    #       advantages = advantages.at[:, t].set(last_advantage)
    #
    #       last_value = values[:, t]
    #
    #   returns = advantages + values
    #   return advantages, returns
    #
    # OPTION B (Advanced — jax.lax.scan for speed, try after Option A works):
    #   Use jax.lax.scan to replace the Python for-loop. The scan function
    #   processes timesteps in reverse. This compiles to a single XLA op.
    #
    #   def _gae_step(carry, t_data):
    #       last_advantage, last_value = carry
    #       reward_t, value_t, mask_t = t_data
    #       delta = reward_t + gamma * last_value * mask_t - value_t
    #       advantage = delta + gamma * gae_lambda * last_advantage * mask_t
    #       return (advantage, value_t), advantage
    #
    #   # Prepare data for scan (reversed along time axis)
    #   if mask is not None:
    #       scan_data = (rewards[:, ::-1].T, values[:, ::-1].T, mask[:, ::-1].T)
    #   else:
    #       scan_data = (rewards[:, ::-1].T, values[:, ::-1].T, jnp.ones_like(rewards[:, ::-1].T))
    #   # ^ Transpose because scan iterates over the first axis
    #
    #   init_carry = (jnp.zeros(rewards.shape[0]), jnp.zeros(rewards.shape[0]))
    #   _, advantages_reversed = jax.lax.scan(_gae_step, init_carry, scan_data)
    #   advantages = advantages_reversed[::-1].T  # Reverse back and transpose
    #   returns = advantages + values
    #   return advantages, returns

    raise NotImplementedError("EXERCISE 5.2: Implement Generalized Advantage Estimation (GAE)")


# ============================================================================
# EXERCISE 5.3 — PPO Clipped Surrogate Objective
# ============================================================================
#
# GOAL: Implement the PPO clipped surrogate loss for the policy.
#
# WHY JAX:
#   - The clipped objective involves element-wise min, clip, and multiply.
#   - JIT fuses all these into a single efficient kernel.
#   - No special handling needed — it's just JAX array ops.
#
# MATH:
#   ratio = exp(log_pi - log_pi_old)
#   surr1 = ratio * advantages
#   surr2 = clip(ratio, 1 - eps, 1 + eps) * advantages
#   L_clip = -min(surr1, surr2)
# ============================================================================

def ppo_policy_loss(
    log_probs: jax.Array,
    old_log_probs: jax.Array,
    advantages: jax.Array,
    clip_eps: float = 0.2,
    mask: jax.Array | None = None,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """Compute the PPO clipped surrogate policy loss.

    Args:
        log_probs: Current policy log-probs, shape (batch_size, seq_len).
        old_log_probs: Old policy log-probs (from rollout), shape (batch_size, seq_len).
        advantages: GAE advantages, shape (batch_size, seq_len).
        clip_eps: Clipping parameter epsilon (default 0.2).
        mask: Optional token mask, shape (batch_size, seq_len).

    Returns:
        Tuple of:
            - Scalar policy loss.
            - Dict of metrics: {'clip_fraction': ..., 'approx_kl': ...}
    """
    # ---- EXERCISE 5.3: Implement PPO clipped surrogate objective ----
    #
    # Step 1: Compute importance sampling ratio
    #   ratio = jnp.exp(log_probs - old_log_probs)  # pi / pi_old
    #
    # Step 2: Compute clipped and unclipped surrogate objectives
    #   surr1 = ratio * advantages
    #   surr2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    #
    # Step 3: Take the minimum (pessimistic bound) and negate (we minimize)
    #   policy_loss = -jnp.minimum(surr1, surr2)
    #
    # Step 4: Apply mask and compute mean
    #   if mask is not None:
    #       policy_loss = (policy_loss * mask).sum() / jnp.maximum(mask.sum(), 1.0)
    #   else:
    #       policy_loss = policy_loss.mean()
    #
    # Step 5: Compute monitoring metrics
    #   clip_fraction = ((ratio - 1.0).abs() > clip_eps).float().mean()
    #   approx_kl = (old_log_probs - log_probs).mean()  # approximate KL
    #
    # return policy_loss, {"clip_fraction": clip_fraction, "approx_kl": approx_kl}

    raise NotImplementedError("EXERCISE 5.3: Implement the PPO clipped surrogate objective")


# ============================================================================
# EXERCISE 5.4 — Value Function Loss
# ============================================================================
#
# GOAL: Compute the value function loss. The value network predicts the
#       expected return at each token position. We use clipped value loss
#       (similar to the policy loss) for stability.
#
# WHY JAX:
#   - Same JIT benefits as the policy loss.
#   - Flax makes it easy to have a separate value head that shares the
#     transformer backbone with the policy. The value params are just a
#     separate leaf in the pytree.
#
# MATH:
#   L_vf = 0.5 * max(
#       (V - returns)^2,
#       (clip(V, V_old - eps, V_old + eps) - returns)^2
#   )
# ============================================================================

def value_function_loss(
    values: jax.Array,
    old_values: jax.Array,
    returns: jax.Array,
    clip_eps: float = 0.2,
    mask: jax.Array | None = None,
) -> jax.Array:
    """Compute the clipped value function loss.

    Args:
        values: Current value estimates, shape (batch_size, seq_len).
        old_values: Old value estimates (from rollout), shape (batch_size, seq_len).
        returns: GAE returns (advantages + values), shape (batch_size, seq_len).
        clip_eps: Value clipping epsilon.
        mask: Optional token mask.

    Returns:
        Scalar value function loss.
    """
    # ---- EXERCISE 5.4: Implement clipped value function loss ----
    #
    # Step 1: Unclipped value loss
    #   vf_loss1 = (values - returns) ** 2
    #
    # Step 2: Clipped value loss
    #   clipped_values = old_values + jnp.clip(values - old_values, -clip_eps, clip_eps)
    #   vf_loss2 = (clipped_values - returns) ** 2
    #
    # Step 3: Take the maximum (conservative estimate) and mean
    #   vf_loss = 0.5 * jnp.maximum(vf_loss1, vf_loss2)
    #
    # Step 4: Apply mask and compute mean
    #   if mask is not None:
    #       vf_loss = (vf_loss * mask).sum() / jnp.maximum(mask.sum(), 1.0)
    #   else:
    #       vf_loss = vf_loss.mean()
    #
    # return vf_loss

    raise NotImplementedError("EXERCISE 5.4: Implement the clipped value function loss")


# ============================================================================
# EXERCISE 5.5 — Combined PPO Loss and Update Step
# ============================================================================
#
# GOAL: Combine the policy loss, value loss, entropy bonus, and KL penalty
#       into a single PPO update step. This is the inner loop of RLHF.
#
# WHY JAX:
#   - The entire combined loss (policy + value + entropy + KL) compiles to
#     one XLA graph. JAX differentiates through all components simultaneously.
#   - jax.value_and_grad with has_aux=True handles the multiple loss terms
#     and metrics cleanly.
#
# COMBINED LOSS:
#   L = L_policy + vf_coeff * L_value - entropy_coeff * entropy + kl_coeff * KL
# ============================================================================

def ppo_update_step(
    params: Any,
    opt_state: Any,
    optimizer: optax.GradientTransformation,
    batch: Dict[str, jax.Array],
    model: GPT2LMHeadModel,
    value_head_apply_fn,
    ref_log_probs: jax.Array,
    clip_eps: float = 0.2,
    vf_coeff: float = 0.5,
    entropy_coeff: float = 0.01,
    kl_coeff: float = 0.1,
) -> Tuple[Any, Any, Dict[str, jax.Array]]:
    """Perform one PPO update step on a minibatch.

    Args:
        params: Combined policy + value head parameters.
        opt_state: Optimizer state.
        optimizer: Optax optimizer.
        batch: Dictionary with:
            - 'input_ids': (batch_size, seq_len)
            - 'attention_mask': (batch_size, seq_len)
            - 'old_log_probs': (batch_size, seq_len) — from rollout
            - 'old_values': (batch_size, seq_len) — from rollout
            - 'advantages': (batch_size, seq_len) — from GAE
            - 'returns': (batch_size, seq_len) — advantages + values
        model: GPT2LMHeadModel for the policy.
        value_head_apply_fn: Function to compute values: (params, hidden_states) -> values.
        ref_log_probs: Reference model log-probs (frozen), shape (batch_size, seq_len).
        clip_eps: PPO clipping epsilon.
        vf_coeff: Value function loss coefficient.
        entropy_coeff: Entropy bonus coefficient.
        kl_coeff: KL penalty coefficient.

    Returns:
        Tuple of (new_params, new_opt_state, metrics_dict).
    """
    # ---- EXERCISE 5.5: Implement the combined PPO update step ----
    #
    # def loss_fn(params):
    #     # 1. Forward pass through policy model
    #     logits = model.apply(params, batch['input_ids'],
    #                          attention_mask=batch['attention_mask'],
    #                          deterministic=True)
    #
    #     # 2. Compute current log-probs
    #     log_probs = compute_log_probs(logits, batch['input_ids'], batch['attention_mask'])
    #
    #     # 3. Compute value predictions (using value head)
    #     hidden_states = model.apply(params, batch['input_ids'],
    #                                 attention_mask=batch['attention_mask'],
    #                                 deterministic=True,
    #                                 method=model.get_hidden_states)
    #     values = value_head_apply_fn(params, hidden_states)
    #
    #     # 4. Policy loss (Exercise 5.3)
    #     policy_loss, policy_metrics = ppo_policy_loss(
    #         log_probs, batch['old_log_probs'], batch['advantages'],
    #         clip_eps=clip_eps, mask=batch['attention_mask'][:, 1:]
    #     )
    #
    #     # 5. Value loss (Exercise 5.4)
    #     vf_loss = value_function_loss(
    #         values, batch['old_values'], batch['returns'],
    #         clip_eps=clip_eps, mask=batch['attention_mask']
    #     )
    #
    #     # 6. Entropy bonus
    #     entropy = compute_entropy(logits, batch['attention_mask'])
    #
    #     # 7. KL penalty (Exercise 5.1)
    #     kl = compute_kl_divergence(log_probs, ref_log_probs, batch['attention_mask'][:, 1:])
    #
    #     # 8. Combined loss
    #     total_loss = policy_loss + vf_coeff * vf_loss - entropy_coeff * entropy + kl_coeff * kl
    #
    #     metrics = {
    #         "total_loss": total_loss,
    #         "policy_loss": policy_loss,
    #         "value_loss": vf_loss,
    #         "entropy": entropy,
    #         "kl": kl,
    #         **policy_metrics,
    #     }
    #     return total_loss, metrics
    #
    # # Compute loss and gradients
    # (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    #
    # # Apply optimizer
    # updates, new_opt_state = optimizer.update(grads, opt_state, params)
    # new_params = optax.apply_updates(params, updates)
    #
    # return new_params, new_opt_state, metrics

    raise NotImplementedError("EXERCISE 5.5: Implement the combined PPO update step")
