"""GRPO Trainer: orchestrates GRPO training.

This is INFRASTRUCTURE code (not an exercise). It calls your GRPO functions
from algorithms/grpo.py.
"""

from __future__ import annotations

from typing import Optional, Any
import jax
import jax.numpy as jnp
import optax

from configs.model_config import ModelConfig
from configs.grpo_config import GRPOConfig
from models.gpt2 import GPT2LMHeadModel
from models.reward_model import RewardModel
from models.policy import compute_log_probs
from algorithms.grpo import group_sample_and_score, group_relative_advantage, grpo_update_step
from utils.generation import generate
from utils.logging_utils import MetricsTracker
from utils.checkpointing import save_checkpoint
from utils.jax_utils import check_backend, count_params, clone_params


def train_grpo(
    policy_params: Any,
    reward_model_params: Any,
    model_config: Optional[ModelConfig] = None,
    grpo_config: Optional[GRPOConfig] = None,
):
    """Run GRPO training.

    Args:
        policy_params: Pre-trained SFT model parameters.
        reward_model_params: Trained reward model parameters.
        model_config: Model architecture config.
        grpo_config: GRPO hyperparameters.

    Returns:
        Trained policy parameters.
    """
    if model_config is None:
        model_config = ModelConfig()
    if grpo_config is None:
        grpo_config = GRPOConfig()

    check_backend()

    # Create models
    policy_model = GPT2LMHeadModel(config=model_config)
    reward_model = RewardModel(config=model_config)

    # Freeze reference model
    ref_params = clone_params(policy_params)
    params = policy_params

    # Create optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(grpo_config.max_grad_norm),
        optax.adamw(grpo_config.learning_rate, weight_decay=grpo_config.weight_decay),
    )
    opt_state = optimizer.init(params)

    print(f"Policy parameters: {count_params(params):,}")

    # Define generate and reward functions for group sampling
    def gen_fn(rng, prompt_ids):
        def apply_fn(params, ids):
            return policy_model.apply(params, ids, deterministic=True)
        return generate(
            apply_fn=apply_fn,
            params=params,
            input_ids=prompt_ids,
            rng=rng,
            max_new_tokens=grpo_config.max_response_len,
            temperature=grpo_config.temperature,
        )

    def reward_fn(input_ids):
        return reward_model.apply(
            reward_model_params, input_ids, deterministic=True,
        )

    # JIT compile update step
    @jax.jit
    def jit_update(params, opt_state, batch, advantages):
        return grpo_update_step(
            params, opt_state, optimizer, batch, policy_model,
            ref_params, advantages,
            clip_eps=grpo_config.clip_eps, kl_coeff=grpo_config.kl_coeff,
        )

    # Training loop
    tracker = MetricsTracker(log_dir="logs/grpo")
    rng = jax.random.PRNGKey(0)

    for iteration in range(grpo_config.num_iterations):
        rng, sample_rng = jax.random.split(rng)

        # Generate prompts (simplified: random for skeleton)
        prompt_ids = jax.random.randint(
            sample_rng, (grpo_config.batch_size, 8),
            0, model_config.vocab_size,
        )

        # Group sampling and scoring
        generated_ids, rewards = group_sample_and_score(
            generate_fn=gen_fn,
            reward_fn=reward_fn,
            prompt_ids=prompt_ids,
            rng=sample_rng,
            group_size=grpo_config.group_size,
            max_response_len=grpo_config.max_response_len,
        )

        # Compute group-relative advantages
        advantages = group_relative_advantage(rewards)

        # Flatten group dimension for update
        B, G = advantages.shape
        total_len = generated_ids.shape[-1]
        flat_ids = generated_ids.reshape(B * G, total_len)
        flat_advantages = advantages.reshape(B * G)
        flat_mask = jnp.ones_like(flat_ids, dtype=jnp.float32)

        # Compute old log-probs (for importance sampling ratio)
        old_logits = policy_model.apply(params, flat_ids, deterministic=True)
        old_log_probs = compute_log_probs(old_logits, flat_ids, flat_mask)

        batch = {
            "input_ids": flat_ids,
            "attention_mask": flat_mask,
            "old_log_probs": jax.lax.stop_gradient(old_log_probs),
        }

        # Update
        for _ in range(grpo_config.num_update_epochs):
            params, opt_state, metrics = jit_update(
                params, opt_state, batch, flat_advantages,
            )

        # Logging
        log_data = {k: float(v) for k, v in metrics.items()}
        log_data["mean_reward"] = float(rewards.mean())
        log_data["reward_std"] = float(rewards.std())
        tracker.log(iteration, log_data)

        if iteration % grpo_config.log_every_steps == 0:
            tracker.print_summary(
                iteration,
                keys=["policy_loss", "kl", "mean_reward"],
            )

        if iteration % grpo_config.save_every_steps == 0:
            save_checkpoint(params, iteration, grpo_config.checkpoint_dir)

    save_checkpoint(params, grpo_config.num_iterations, grpo_config.checkpoint_dir)
    tracker.save_csv()

    print(f"\nGRPO training complete. Iterations: {grpo_config.num_iterations}")
    return params
