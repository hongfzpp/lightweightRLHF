"""Phase 6: DPO (Direct Preference Optimization) — Exercises 6.1 through 6.3.

==========================================================================
 DIFFICULTY: Medium-Hard
 PREREQUISITES: Phase 3 (SFT) + Phase 4 (Reward concepts)
 ESTIMATED TIME: 2-3 hours
==========================================================================

DPO is an elegant alternative to PPO that does NOT require a separate reward
model. Instead, it directly optimizes the policy using preference pairs
(chosen, rejected) by treating the policy itself as an implicit reward model.

Key insight: the optimal policy under a reward model r(x,y) with KL constraint
has a closed-form solution, which means you can reparameterize the preference
loss to depend only on the policy — no reward model needed.

WHY DPO (vs. PPO):
    - Simpler: no reward model training, no value network, no rollout generation
    - Stable: no clipping heuristics, no GAE hyperparameters
    - Efficient: offline training on static preference data
    - Often competitive with PPO on benchmarks

WHY JAX FOR DPO:
    - DPO requires computing log-probs under BOTH the current policy and a
      frozen reference model. In JAX, both forward passes compile into a single
      XLA graph via @jax.jit.
    - The reference model is just a frozen pytree — no .eval() or .detach() needed.
      Just don't pass it to the optimizer. JAX's functional paradigm makes this natural.
    - jax.lax.stop_gradient on the reference log-probs is explicit and clean.

REFERENCES:
    - "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
      (Rafailov et al., 2023)

MATH:
    L_DPO = -E[log sigmoid(beta * (log_ratio_chosen - log_ratio_rejected))]

    where: log_ratio = log pi_theta(y|x) - log pi_ref(y|x)
"""

from __future__ import annotations

from typing import Any, Dict, Tuple
import jax
import jax.numpy as jnp
import optax

from models.gpt2 import GPT2LMHeadModel
from models.policy import compute_log_probs, compute_sequence_log_probs
from configs.model_config import ModelConfig


# ============================================================================
# EXERCISE 6.1 — Sequence Log-Probabilities for DPO
# ============================================================================
#
# GOAL: Given a model and a (prompt + response) sequence, compute the total
#       log-probability of the response tokens only (not the prompt).
#       This is: log pi(y|x) = sum_{t in response} log pi(y_t | x, y_{<t})
#
# WHY JAX:
#   - jax.vmap can batch this over (chosen, rejected) pairs efficiently.
#   - jax.jit compiles both forward passes (policy + reference) into one graph.
#   - No need to manually manage .no_grad() scopes for the reference model.
#
# DIFFERENCE FROM Phase 3:
#   - In SFT (Phase 3), we computed loss averaged over all tokens.
#   - Here, we need the SUM of log-probs over response tokens only, because
#     DPO compares log P(whole_response | prompt) between chosen and rejected.
# ============================================================================

def compute_response_log_probs(
    model: GPT2LMHeadModel,
    params: Any,
    input_ids: jax.Array,
    attention_mask: jax.Array,
    response_mask: jax.Array,
) -> jax.Array:
    """Compute log pi(response | prompt) for each sequence in the batch.

    Args:
        model: GPT2LMHeadModel instance.
        params: Model parameters.
        input_ids: Full sequence (prompt + response), shape (batch_size, seq_len).
        attention_mask: Padding mask, shape (batch_size, seq_len).
        response_mask: 1 for response tokens, 0 for prompt/padding.
                      Shape (batch_size, seq_len).

    Returns:
        Sequence log-probs, shape (batch_size,) — sum of log-probs over response tokens.
    """
    # ---- EXERCISE 6.1: Compute response-only sequence log-probs ----
    #
    # Step 1: Forward pass to get logits
    #   logits = model.apply(params, input_ids,
    #                        attention_mask=attention_mask,
    #                        deterministic=True)
    #   # logits shape: (batch_size, seq_len, vocab_size)
    #
    # Step 2: Compute per-token log-probs (using the policy utility)
    #   token_log_probs = compute_log_probs(logits, input_ids, response_mask)
    #   # token_log_probs shape: (batch_size, seq_len - 1)
    #   # Note: compute_log_probs already shifts by 1 and applies the mask
    #
    # Step 3: Sum over sequence length to get total log P(response | prompt)
    #   seq_log_probs = token_log_probs.sum(axis=-1)  # (batch_size,)
    #   return seq_log_probs

    raise NotImplementedError("EXERCISE 6.1: Compute response-only sequence log-probabilities")


# ============================================================================
# EXERCISE 6.2 — DPO Loss
# ============================================================================
#
# GOAL: Implement the core DPO loss function.
#
# WHY JAX:
#   - The entire DPO loss (4 forward passes: policy_chosen, policy_rejected,
#     ref_chosen, ref_rejected + loss computation) compiles into ONE XLA graph.
#   - Reference model log-probs are detached via jax.lax.stop_gradient — clean
#     and explicit, unlike PyTorch's torch.no_grad() context manager.
#   - jax.nn.log_sigmoid is numerically stable.
#
# MATH:
#   log_ratio_chosen = log pi_theta(y_w | x) - log pi_ref(y_w | x)
#   log_ratio_rejected = log pi_theta(y_l | x) - log pi_ref(y_l | x)
#   L = -mean[log sigmoid(beta * (log_ratio_chosen - log_ratio_rejected))]
#
# IMPLICIT REWARD:
#   r(x, y) = beta * (log pi_theta(y|x) - log pi_ref(y|x))
#   This is the "implicit reward" — DPO effectively trains a reward model
#   and policy simultaneously!
# ============================================================================

def dpo_loss(
    policy_chosen_logps: jax.Array,
    policy_rejected_logps: jax.Array,
    ref_chosen_logps: jax.Array,
    ref_rejected_logps: jax.Array,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """Compute the DPO loss.

    Args:
        policy_chosen_logps: log pi_theta(y_w | x), shape (batch_size,).
        policy_rejected_logps: log pi_theta(y_l | x), shape (batch_size,).
        ref_chosen_logps: log pi_ref(y_w | x), shape (batch_size,).
        ref_rejected_logps: log pi_ref(y_l | x), shape (batch_size,).
        beta: DPO temperature parameter.
        label_smoothing: Optional label smoothing (0 = no smoothing).

    Returns:
        Tuple of:
            - Scalar DPO loss.
            - Dict of metrics: {
                'chosen_reward': mean implicit reward for chosen,
                'rejected_reward': mean implicit reward for rejected,
                'reward_margin': mean reward difference,
                'accuracy': fraction where chosen_reward > rejected_reward,
              }
    """
    # ---- EXERCISE 6.2: Implement the DPO loss ----
    #
    # Step 1: Compute log-ratios (policy vs reference)
    #   chosen_log_ratio = policy_chosen_logps - jax.lax.stop_gradient(ref_chosen_logps)
    #   rejected_log_ratio = policy_rejected_logps - jax.lax.stop_gradient(ref_rejected_logps)
    #
    # Step 2: Compute implicit rewards
    #   chosen_reward = beta * chosen_log_ratio
    #   rejected_reward = beta * rejected_log_ratio
    #
    # Step 3: Compute the DPO loss
    #   logits = chosen_reward - rejected_reward   # (batch_size,)
    #   if label_smoothing > 0:
    #       loss = -(1 - label_smoothing) * jax.nn.log_sigmoid(logits) \
    #              - label_smoothing * jax.nn.log_sigmoid(-logits)
    #   else:
    #       loss = -jax.nn.log_sigmoid(logits)
    #   loss = loss.mean()
    #
    # Step 4: Compute metrics
    #   accuracy = (chosen_reward > rejected_reward).astype(jnp.float32).mean()
    #   reward_margin = (chosen_reward - rejected_reward).mean()
    #
    # metrics = {
    #     "chosen_reward": chosen_reward.mean(),
    #     "rejected_reward": rejected_reward.mean(),
    #     "reward_margin": reward_margin,
    #     "accuracy": accuracy,
    # }
    # return loss, metrics

    raise NotImplementedError("EXERCISE 6.2: Implement the DPO loss")


# ============================================================================
# EXERCISE 6.3 — DPO Training Step
# ============================================================================
#
# GOAL: Combine Exercises 6.1 and 6.2 into a complete training step.
#
# WHY JAX:
#   - Four forward passes (policy chosen/rejected, ref chosen/rejected) +
#     loss computation + backward pass + optimizer update all compile into
#     a single XLA graph. Zero Python overhead between operations.
#   - The reference model params are just a frozen pytree — passed as an
#     argument but not included in the gradient computation.
# ============================================================================

def create_dpo_train_state(
    model: GPT2LMHeadModel,
    config: ModelConfig,
    learning_rate: float = 5e-6,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    rng: jax.Array = None,
) -> Tuple[Any, Any, optax.GradientTransformation]:
    """Initialize model and optimizer for DPO training.

    Returns:
        Tuple of (params, opt_state, optimizer).
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)

    dummy_input = jnp.ones((1, config.max_seq_len), dtype=jnp.int32)
    params = model.init(rng, dummy_input)

    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(learning_rate, weight_decay=weight_decay),
    )
    opt_state = optimizer.init(params)

    return params, opt_state, optimizer


def dpo_train_step(
    params: Any,
    ref_params: Any,
    opt_state: Any,
    optimizer: optax.GradientTransformation,
    batch: Dict[str, jax.Array],
    model: GPT2LMHeadModel,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> Tuple[Any, Any, jax.Array, Dict[str, jax.Array]]:
    """Perform one DPO training step.

    Args:
        params: Current policy parameters (trainable).
        ref_params: Reference model parameters (frozen, from SFT).
        opt_state: Optimizer state.
        optimizer: Optax optimizer.
        batch: Dictionary with:
            - 'chosen_input_ids': (batch_size, seq_len)
            - 'chosen_attention_mask': (batch_size, seq_len)
            - 'chosen_response_mask': (batch_size, seq_len) — 1 for response tokens
            - 'rejected_input_ids': (batch_size, seq_len)
            - 'rejected_attention_mask': (batch_size, seq_len)
            - 'rejected_response_mask': (batch_size, seq_len)
        model: GPT2LMHeadModel instance.
        beta: DPO temperature.
        label_smoothing: Label smoothing parameter.

    Returns:
        Tuple of (new_params, new_opt_state, loss, metrics).
    """
    # ---- EXERCISE 6.3: Implement the DPO training step ----
    #
    # def loss_fn(params):
    #     # Compute log-probs under current policy
    #     policy_chosen_logps = compute_response_log_probs(
    #         model, params,
    #         batch['chosen_input_ids'], batch['chosen_attention_mask'],
    #         batch['chosen_response_mask'],
    #     )
    #     policy_rejected_logps = compute_response_log_probs(
    #         model, params,
    #         batch['rejected_input_ids'], batch['rejected_attention_mask'],
    #         batch['rejected_response_mask'],
    #     )
    #
    #     # Compute log-probs under reference model (frozen)
    #     ref_chosen_logps = compute_response_log_probs(
    #         model, ref_params,
    #         batch['chosen_input_ids'], batch['chosen_attention_mask'],
    #         batch['chosen_response_mask'],
    #     )
    #     ref_rejected_logps = compute_response_log_probs(
    #         model, ref_params,
    #         batch['rejected_input_ids'], batch['rejected_attention_mask'],
    #         batch['rejected_response_mask'],
    #     )
    #     # IMPORTANT: stop gradient through reference log-probs
    #     ref_chosen_logps = jax.lax.stop_gradient(ref_chosen_logps)
    #     ref_rejected_logps = jax.lax.stop_gradient(ref_rejected_logps)
    #
    #     loss, metrics = dpo_loss(
    #         policy_chosen_logps, policy_rejected_logps,
    #         ref_chosen_logps, ref_rejected_logps,
    #         beta=beta, label_smoothing=label_smoothing,
    #     )
    #     return loss, metrics
    #
    # (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    # updates, new_opt_state = optimizer.update(grads, opt_state, params)
    # new_params = optax.apply_updates(params, updates)
    #
    # return new_params, new_opt_state, loss, metrics

    raise NotImplementedError("EXERCISE 6.3: Implement the DPO training step")
