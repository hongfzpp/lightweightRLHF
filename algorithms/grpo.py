"""Phase 7: GRPO (Group Relative Policy Optimization) — Exercises 7.1 through 7.3.

==========================================================================
 DIFFICULTY: Hard
 PREREQUISITES: Phase 5 (PPO concepts) + Phase 6 (log-prob computation)
 ESTIMATED TIME: 3-4 hours
==========================================================================

GRPO (from DeepSeek-R1) is a PPO variant that eliminates the value network by
estimating advantages from a GROUP of responses per prompt. For each prompt:
    1. Sample G responses from the current policy
    2. Score each response with a reward model
    3. Normalize rewards within the group to get advantages
    4. Update the policy with PPO-clip objective using group-relative advantages

Key innovation: no value network needed! The advantage for response i is simply
its z-score within the group: A_i = (R_i - mean(R)) / std(R).

WHY GRPO (vs. PPO):
    - No value network = fewer parameters, simpler training
    - Group normalization is a natural baseline (no need to fit V(s))
    - Works especially well for tasks with verifiable rewards (math, code)
    - Cheaper per iteration than PPO (no value loss, no GAE)

WHY JAX FOR GRPO:
    - jax.vmap is PERFECT for the group dimension: sample G responses per prompt,
      compute rewards for all G in parallel, normalize within each group.
    - jax.random.split creates G independent PRNG keys for parallel sampling.
    - The entire group scoring + advantage computation + PPO update compiles
      to a single XLA graph.

REFERENCES:
    - "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via
       Reinforcement Learning" (DeepSeek, 2025)
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, Callable
import jax
import jax.numpy as jnp
import optax

from models.gpt2 import GPT2LMHeadModel
from models.reward_model import RewardModel
from models.policy import compute_log_probs


# ============================================================================
# EXERCISE 7.1 — Group Sampling and Scoring
# ============================================================================
#
# GOAL: For each prompt in a batch, sample G responses from the current policy
#       and score each with the reward model. This creates the "group" from which
#       advantages are computed.
#
# WHY JAX:
#   - jax.random.split creates G independent PRNG keys, enabling reproducible
#     parallel sampling. No global random state like torch.manual_seed().
#   - jax.vmap over the group dimension processes all G responses per prompt
#     in parallel, which the XLA compiler optimizes for the Metal GPU.
#   - The entire sampling + scoring pipeline compiles into one graph.
#
# SHAPE CONVENTION:
#   prompts: (batch_size, prompt_len)
#   responses: (batch_size, group_size, response_len)
#   rewards: (batch_size, group_size)
# ============================================================================

def group_sample_and_score(
    generate_fn: Callable,
    reward_fn: Callable,
    prompt_ids: jax.Array,
    rng: jax.Array,
    group_size: int = 8,
    max_response_len: int = 128,
) -> Tuple[jax.Array, jax.Array]:
    """Sample a group of responses per prompt and score them.

    Args:
        generate_fn: Function (rng, prompt_ids) -> generated_ids.
                     Generates one response per prompt in the batch.
        reward_fn: Function (input_ids) -> rewards.
                   Computes scalar rewards for sequences.
        prompt_ids: Prompt token IDs, shape (batch_size, prompt_len).
        rng: PRNG key for sampling.
        group_size: Number of responses to sample per prompt (G).
        max_response_len: Maximum response length.

    Returns:
        Tuple of:
            - generated_ids: (batch_size, group_size, total_len) — all generated sequences
            - rewards: (batch_size, group_size) — reward for each response
    """
    # ---- EXERCISE 7.1: Implement group sampling and scoring ----
    #
    # Step 1: Create G independent PRNG keys for sampling
    #   rngs = jax.random.split(rng, group_size)  # (G,) keys
    #
    # Step 2: Sample G responses for each prompt
    #   all_generated = []
    #   all_rewards = []
    #   for g in range(group_size):
    #       generated = generate_fn(rngs[g], prompt_ids)  # (batch_size, total_len)
    #       reward = reward_fn(generated)                  # (batch_size,)
    #       all_generated.append(generated)
    #       all_rewards.append(reward)
    #
    # Step 3: Stack into group dimension
    #   generated_ids = jnp.stack(all_generated, axis=1)  # (batch_size, G, total_len)
    #   rewards = jnp.stack(all_rewards, axis=1)           # (batch_size, G)
    #
    # return generated_ids, rewards
    #
    # NOTE: For better performance, you could vmap the generate_fn over the group
    # dimension, but the loop version is clearer for learning.

    raise NotImplementedError("EXERCISE 7.1: Implement group sampling and scoring")


# ============================================================================
# EXERCISE 7.2 — Group-Relative Advantage Estimation
# ============================================================================
#
# GOAL: Compute advantages by normalizing rewards within each group (per prompt).
#       This is the key innovation of GRPO — it replaces the value network!
#
# WHY JAX:
#   - Pure jnp operations: mean, std, division — all compile to efficient kernels.
#   - No Python loops needed — jnp operates along the group axis.
#   - vmap could further vectorize if needed (e.g., per-token advantages).
#
# MATH:
#   For each prompt i and response j in group G_i:
#       A_{i,j} = (R_{i,j} - mean(R_i)) / (std(R_i) + eps)
#
#   where R_i = {R_{i,1}, ..., R_{i,G}} are the rewards for all G responses.
#
# INTUITION:
#   - If a response scores above the group average, it gets a positive advantage.
#   - If it scores below, it gets a negative advantage.
#   - This is robust to reward scale and doesn't need a learned baseline.
# ============================================================================

def group_relative_advantage(
    rewards: jax.Array,
    eps: float = 1e-8,
) -> jax.Array:
    """Compute group-relative advantages.

    Args:
        rewards: Per-response rewards, shape (batch_size, group_size).
        eps: Small constant for numerical stability in std division.

    Returns:
        Advantages, shape (batch_size, group_size).
    """
    # ---- EXERCISE 7.2: Implement group-relative advantage ----
    #
    # Step 1: Compute per-group statistics (mean and std along group axis)
    #   group_mean = rewards.mean(axis=-1, keepdims=True)  # (batch_size, 1)
    #   group_std = rewards.std(axis=-1, keepdims=True)    # (batch_size, 1)
    #
    # Step 2: Normalize
    #   advantages = (rewards - group_mean) / (group_std + eps)  # (batch_size, group_size)
    #
    # return advantages

    raise NotImplementedError("EXERCISE 7.2: Implement group-relative advantage estimation")


# ============================================================================
# EXERCISE 7.3 — GRPO Objective and Update Step
# ============================================================================
#
# GOAL: Implement the GRPO objective — PPO-clip style loss using group-relative
#       advantages, plus KL regularization. This combines:
#       - PPO clipped surrogate objective (from Exercise 5.3)
#       - Group-relative advantages (from Exercise 7.2)
#       - KL penalty against reference model (from Exercise 5.1)
#
# WHY JAX:
#   - The complete loss (N forward passes for group members + clip objective +
#     KL penalty) compiles into a single XLA graph.
#   - jax.value_and_grad with has_aux handles the complex multi-component loss.
#
# MATH:
#   For each response j in group, at each token position t:
#       r_{j,t} = pi_theta(a_{j,t}) / pi_old(a_{j,t})
#       L_{j,t} = min(r_{j,t} * A_j, clip(r_{j,t}, 1-eps, 1+eps) * A_j)
#
#   Total: L = -(1/G) sum_j L_j - kl_coeff * KL(pi || pi_ref)
#
#   Note: A_j is the same for ALL tokens in response j (sequence-level advantage).
# ============================================================================

def grpo_loss(
    log_probs: jax.Array,
    old_log_probs: jax.Array,
    ref_log_probs: jax.Array,
    advantages: jax.Array,
    clip_eps: float = 0.2,
    kl_coeff: float = 0.05,
    mask: jax.Array | None = None,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """Compute the GRPO loss for a single group member.

    This function computes the loss for ONE response. The caller should
    loop over or vmap over group members and average.

    Args:
        log_probs: Current policy per-token log-probs, shape (batch_size, seq_len).
        old_log_probs: Old policy log-probs (from sampling), shape (batch_size, seq_len).
        ref_log_probs: Reference model log-probs, shape (batch_size, seq_len).
        advantages: Sequence-level advantages, shape (batch_size,).
                    Same value for all tokens in a response (group-relative).
        clip_eps: PPO clipping epsilon.
        kl_coeff: KL penalty coefficient.
        mask: Token mask, shape (batch_size, seq_len).

    Returns:
        Tuple of (scalar_loss, metrics_dict).
    """
    # ---- EXERCISE 7.3: Implement the GRPO loss ----
    #
    # Step 1: Expand sequence-level advantages to per-token
    #   # advantages is (batch_size,), need (batch_size, seq_len)
    #   token_advantages = advantages[:, None] * jnp.ones_like(log_probs)
    #
    # Step 2: Importance sampling ratio
    #   ratio = jnp.exp(log_probs - jax.lax.stop_gradient(old_log_probs))
    #
    # Step 3: PPO clipped objective
    #   surr1 = ratio * token_advantages
    #   surr2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * token_advantages
    #   policy_loss = -jnp.minimum(surr1, surr2)
    #
    # Step 4: KL penalty
    #   kl = log_probs - jax.lax.stop_gradient(ref_log_probs)
    #
    # Step 5: Combined loss with masking
    #   total_per_token = policy_loss + kl_coeff * kl
    #   if mask is not None:
    #       total_per_token = total_per_token * mask
    #       loss = total_per_token.sum() / jnp.maximum(mask.sum(), 1.0)
    #       kl_mean = (kl * mask).sum() / jnp.maximum(mask.sum(), 1.0)
    #   else:
    #       loss = total_per_token.mean()
    #       kl_mean = kl.mean()
    #
    # Step 6: Metrics
    #   clip_fraction = ((jnp.abs(ratio - 1.0) > clip_eps).astype(jnp.float32))
    #   if mask is not None:
    #       clip_fraction = (clip_fraction * mask).sum() / jnp.maximum(mask.sum(), 1.0)
    #   else:
    #       clip_fraction = clip_fraction.mean()
    #
    #   metrics = {
    #       "policy_loss": policy_loss.mean() if mask is None else (policy_loss * mask).sum() / jnp.maximum(mask.sum(), 1.0),
    #       "kl": kl_mean,
    #       "clip_fraction": clip_fraction,
    #   }
    #   return loss, metrics

    raise NotImplementedError("EXERCISE 7.3: Implement the GRPO loss")


def grpo_update_step(
    params: Any,
    opt_state: Any,
    optimizer: optax.GradientTransformation,
    batch: Dict[str, jax.Array],
    model: GPT2LMHeadModel,
    ref_params: Any,
    advantages: jax.Array,
    clip_eps: float = 0.2,
    kl_coeff: float = 0.05,
) -> Tuple[Any, Any, Dict[str, jax.Array]]:
    """Perform one GRPO update step.

    This processes all group members in a batch and averages the loss.

    Args:
        params: Current policy parameters.
        opt_state: Optimizer state.
        optimizer: Optax optimizer.
        batch: Dictionary with:
            - 'input_ids': (batch_size * group_size, seq_len) — flattened group
            - 'attention_mask': (batch_size * group_size, seq_len)
            - 'old_log_probs': (batch_size * group_size, seq_len)
        model: GPT2LMHeadModel instance.
        ref_params: Frozen reference model parameters.
        advantages: (batch_size * group_size,) — flattened group advantages.
        clip_eps: PPO clipping epsilon.
        kl_coeff: KL penalty coefficient.

    Returns:
        Tuple of (new_params, new_opt_state, metrics).
    """
    def loss_fn(params):
        # Forward pass with current policy
        logits = model.apply(
            params, batch['input_ids'],
            attention_mask=batch['attention_mask'],
            deterministic=True,
        )
        log_probs = compute_log_probs(logits, batch['input_ids'], batch['attention_mask'])

        # Forward pass with reference model (frozen)
        ref_logits = model.apply(
            ref_params, batch['input_ids'],
            attention_mask=batch['attention_mask'],
            deterministic=True,
        )
        ref_log_probs = compute_log_probs(ref_logits, batch['input_ids'], batch['attention_mask'])
        ref_log_probs = jax.lax.stop_gradient(ref_log_probs)

        # GRPO loss
        loss, metrics = grpo_loss(
            log_probs=log_probs,
            old_log_probs=batch['old_log_probs'],
            ref_log_probs=ref_log_probs,
            advantages=advantages,
            clip_eps=clip_eps,
            kl_coeff=kl_coeff,
            mask=batch['attention_mask'][:, 1:],  # shifted mask
        )
        return loss, metrics

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    return new_params, new_opt_state, metrics
