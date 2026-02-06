"""Reward Model Trainer: orchestrates reward model training.

This is INFRASTRUCTURE code (not an exercise). It calls your reward model
loss and training step functions from algorithms/reward.py.
"""

from __future__ import annotations

from typing import Optional
import jax
import jax.numpy as jnp

from configs.model_config import ModelConfig
from configs.reward_config import RewardConfig
from models.reward_model import RewardModel
from algorithms.reward import create_reward_train_state, reward_train_step
from data.preference_dataset import preference_batch_iterator, get_synthetic_preference_data
from data.tokenizer import get_tokenizer
from utils.logging_utils import MetricsTracker
from utils.checkpointing import save_checkpoint
from utils.jax_utils import check_backend, count_params


def train_reward_model(
    model_config: Optional[ModelConfig] = None,
    reward_config: Optional[RewardConfig] = None,
    train_data=None,
    eval_data=None,
):
    """Run reward model training.

    Args:
        model_config: Model architecture config.
        reward_config: Training hyperparameters.
        train_data: List of (prompt, chosen, rejected) tuples.
        eval_data: Evaluation data.

    Returns:
        Trained reward model parameters.
    """
    if model_config is None:
        model_config = ModelConfig()
    if reward_config is None:
        reward_config = RewardConfig()

    check_backend()

    # Create model
    reward_model = RewardModel(config=model_config)

    # Initialize
    params, opt_state, optimizer = create_reward_train_state(
        reward_model=reward_model,
        config=model_config,
        learning_rate=reward_config.learning_rate,
        weight_decay=reward_config.weight_decay,
        max_grad_norm=reward_config.max_grad_norm,
    )
    n_params = count_params(params)
    print(f"Reward model parameters: {n_params:,}")

    # Data
    tokenizer = get_tokenizer()
    if train_data is None:
        train_data = get_synthetic_preference_data(n_samples=128)
    if eval_data is None:
        eval_data = get_synthetic_preference_data(n_samples=32, seed=99)

    # JIT compile
    @jax.jit
    def jit_train_step(params, opt_state, batch):
        return reward_train_step(params, opt_state, optimizer, batch, reward_model)

    # Training loop
    tracker = MetricsTracker(log_dir="logs/reward")
    global_step = 0

    for epoch in range(reward_config.num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{reward_config.num_epochs} ---")

        for batch in preference_batch_iterator(
            tokenizer, train_data,
            batch_size=reward_config.batch_size,
            max_seq_len=reward_config.max_seq_len,
            seed=epoch,
        ):
            jax_batch = {k: jnp.array(v) for k, v in batch.items()}
            params, opt_state, loss, metrics = jit_train_step(params, opt_state, jax_batch)
            global_step += 1

            tracker.log(global_step, {
                "loss": float(loss),
                "accuracy": float(metrics["accuracy"]),
                "reward_margin": float(metrics["reward_margin"]),
            })

            if global_step % reward_config.log_every_steps == 0:
                tracker.print_summary(global_step, keys=["loss", "accuracy", "reward_margin"])

            if global_step % reward_config.save_every_steps == 0:
                save_checkpoint(params, global_step, reward_config.checkpoint_dir)

    save_checkpoint(params, global_step, reward_config.checkpoint_dir)
    tracker.save_csv()

    print(f"\nReward model training complete. Final step: {global_step}")
    return params
