"""PPO Trainer: orchestrates the PPO-based RLHF training loop.

This is INFRASTRUCTURE code (not an exercise). It handles:
1. Rollout: generating responses from the current policy
2. Scoring: computing rewards with the reward model
3. Advantage estimation: calling your GAE implementation
4. Updates: calling your PPO update step in minibatches

The algorithmic core (KL, GAE, clipped losses) is in algorithms/ppo.py.
"""

from __future__ import annotations

from typing import Optional, Any
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from configs.model_config import ModelConfig
from configs.ppo_config import PPOConfig
from models.gpt2 import GPT2LMHeadModel
from models.reward_model import RewardModel
from models.policy import compute_log_probs
from algorithms.ppo import compute_gae, ppo_update_step
from utils.generation import generate
from utils.logging_utils import MetricsTracker
from utils.checkpointing import save_checkpoint
from utils.jax_utils import check_backend, count_params, clone_params


class ValueHead(nn.Module):
    """Simple value head: projects hidden states to scalar values.

    This is a thin MLP on top of the shared GPT-2 backbone.
    """
    d_model: int

    @nn.compact
    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        """Predict values from hidden states.

        Args:
            hidden_states: (batch_size, seq_len, d_model)

        Returns:
            values: (batch_size, seq_len)
        """
        x = nn.Dense(features=self.d_model)(hidden_states)
        x = nn.relu(x)
        x = nn.Dense(features=1)(x)
        return x.squeeze(-1)


def train_ppo(
    policy_params: Any,
    reward_model_params: Any,
    model_config: Optional[ModelConfig] = None,
    ppo_config: Optional[PPOConfig] = None,
):
    """Run PPO-based RLHF training.

    Args:
        policy_params: Pre-trained SFT model parameters (from Phase 3).
        reward_model_params: Trained reward model parameters (from Phase 4).
        model_config: Model architecture config.
        ppo_config: PPO hyperparameters.

    Returns:
        Trained policy parameters.
    """
    if model_config is None:
        model_config = ModelConfig()
    if ppo_config is None:
        ppo_config = PPOConfig()

    check_backend()

    # Create models
    policy_model = GPT2LMHeadModel(config=model_config)
    reward_model = RewardModel(config=model_config)
    value_head = ValueHead(d_model=model_config.d_model)

    # Initialize value head
    rng = jax.random.PRNGKey(0)
    dummy_hidden = jnp.zeros((1, model_config.max_seq_len, model_config.d_model))
    value_params = value_head.init(rng, dummy_hidden)

    # Freeze reference model (just a copy of the initial policy params)
    ref_params = clone_params(policy_params)

    # Create optimizer for policy + value head
    optimizer = optax.chain(
        optax.clip_by_global_norm(ppo_config.max_grad_norm),
        optax.adamw(ppo_config.learning_rate, weight_decay=ppo_config.weight_decay),
    )
    # Combine policy and value params
    combined_params = {"policy": policy_params, "value": value_params}
    opt_state = optimizer.init(combined_params)

    print(f"Policy params: {count_params(policy_params):,}")
    print(f"Value head params: {count_params(value_params):,}")

    # Training loop
    tracker = MetricsTracker(log_dir="logs/ppo")

    for iteration in range(ppo_config.num_iterations):
        rng, rollout_rng = jax.random.split(rng)

        # ---- ROLLOUT PHASE ----
        # Generate responses using current policy
        # (Simplified: using random prompts for the skeleton)
        prompt_ids = jax.random.randint(
            rollout_rng, (ppo_config.rollout_batch_size, 8),
            0, model_config.vocab_size,
        )

        # Define apply function for generation
        def policy_apply(params, input_ids):
            return policy_model.apply(params, input_ids, deterministic=True)

        generated = generate(
            apply_fn=lambda ids: policy_apply(combined_params["policy"], ids),
            params=combined_params["policy"],
            input_ids=prompt_ids,
            rng=rollout_rng,
            max_new_tokens=ppo_config.max_response_len,
            temperature=ppo_config.temperature,
        )

        # ---- SCORING PHASE ----
        # Compute rewards with reward model
        rewards = reward_model.apply(
            reward_model_params, generated, deterministic=True,
        )

        # Compute log-probs under current policy and reference
        policy_logits = policy_model.apply(
            combined_params["policy"], generated, deterministic=True,
        )
        ref_logits = policy_model.apply(ref_params, generated, deterministic=True)

        old_log_probs = compute_log_probs(policy_logits, generated)
        ref_log_probs_val = compute_log_probs(ref_logits, generated)

        # Compute values
        hidden_states = policy_model.apply(
            combined_params["policy"], generated, deterministic=True,
            method=policy_model.get_hidden_states,
        )
        old_values = value_head.apply(combined_params["value"], hidden_states)

        # ---- ADVANTAGE ESTIMATION ----
        # Build per-token rewards: KL penalty + reward model score at last token
        seq_len = generated.shape[1] - 1  # for shifted log-probs
        per_token_rewards = jnp.zeros((generated.shape[0], seq_len))
        # Add KL penalty at each token
        kl_per_token = old_log_probs - ref_log_probs_val
        per_token_rewards = per_token_rewards - ppo_config.kl_coeff * kl_per_token
        # Add reward model score at last token
        per_token_rewards = per_token_rewards.at[:, -1].add(rewards)

        advantages, returns = compute_gae(
            per_token_rewards, old_values[:, 1:],
            gamma=ppo_config.gamma,
            gae_lambda=ppo_config.gae_lambda,
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ---- PPO UPDATE PHASE ----
        rollout_batch = {
            "input_ids": generated,
            "attention_mask": jnp.ones_like(generated, dtype=jnp.float32),
            "old_log_probs": old_log_probs,
            "old_values": old_values,
            "advantages": advantages,
            "returns": returns,
        }

        def value_apply(params, hidden):
            return value_head.apply(params["value"], hidden)

        for ppo_epoch in range(ppo_config.num_ppo_epochs):
            combined_params, opt_state, metrics = ppo_update_step(
                params=combined_params,
                opt_state=opt_state,
                optimizer=optimizer,
                batch=rollout_batch,
                model=policy_model,
                value_head_apply_fn=value_apply,
                ref_log_probs=ref_log_probs_val,
                clip_eps=ppo_config.clip_eps,
                vf_coeff=ppo_config.vf_coeff,
                entropy_coeff=ppo_config.entropy_coeff,
                kl_coeff=ppo_config.kl_coeff,
            )

        # Logging
        tracker.log(iteration, {k: float(v) for k, v in metrics.items()})
        if iteration % ppo_config.log_every_steps == 0:
            tracker.print_summary(iteration, keys=["total_loss", "policy_loss", "kl", "entropy"])

        # Checkpointing
        if iteration % ppo_config.save_every_steps == 0:
            save_checkpoint(combined_params["policy"], iteration, ppo_config.checkpoint_dir)

    save_checkpoint(combined_params["policy"], ppo_config.num_iterations, ppo_config.checkpoint_dir)
    tracker.save_csv()

    print(f"\nPPO training complete. Iterations: {ppo_config.num_iterations}")
    return combined_params["policy"]
