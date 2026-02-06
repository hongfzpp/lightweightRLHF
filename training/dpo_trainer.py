"""DPO Trainer: orchestrates DPO training.

This is INFRASTRUCTURE code (not an exercise). It handles the training loop
for DPO, calling your loss and training step functions from algorithms/dpo.py.
"""

from __future__ import annotations

from typing import Optional, Any
import jax
import jax.numpy as jnp

from configs.model_config import ModelConfig
from configs.dpo_config import DPOConfig
from models.gpt2 import GPT2LMHeadModel
from algorithms.dpo import create_dpo_train_state, dpo_train_step
from data.preference_dataset import preference_batch_iterator, get_synthetic_preference_data
from data.tokenizer import get_tokenizer
from utils.logging_utils import MetricsTracker
from utils.checkpointing import save_checkpoint
from utils.jax_utils import check_backend, count_params, clone_params


def train_dpo(
    sft_params: Optional[Any] = None,
    model_config: Optional[ModelConfig] = None,
    dpo_config: Optional[DPOConfig] = None,
    train_data=None,
    eval_data=None,
):
    """Run DPO training.

    Args:
        sft_params: Pre-trained SFT model parameters. If None, initializes fresh.
        model_config: Model architecture config.
        dpo_config: DPO hyperparameters.
        train_data: List of (prompt, chosen, rejected) tuples.
        eval_data: Evaluation data.

    Returns:
        Trained policy parameters.
    """
    if model_config is None:
        model_config = ModelConfig()
    if dpo_config is None:
        dpo_config = DPOConfig()

    check_backend()

    # Create model
    model = GPT2LMHeadModel(config=model_config)

    # Initialize or load parameters
    if sft_params is not None:
        params = sft_params
        # Create optimizer for existing params
        optimizer = optax.chain(
            optax.clip_by_global_norm(dpo_config.max_grad_norm),
            optax.adamw(dpo_config.learning_rate, weight_decay=dpo_config.weight_decay),
        )
        opt_state = optimizer.init(params)
    else:
        params, opt_state, optimizer = create_dpo_train_state(
            model, model_config,
            learning_rate=dpo_config.learning_rate,
            weight_decay=dpo_config.weight_decay,
            max_grad_norm=dpo_config.max_grad_norm,
        )

    # Freeze reference model
    ref_params = clone_params(params)
    print(f"Policy parameters: {count_params(params):,}")

    # Data
    tokenizer = get_tokenizer()
    if train_data is None:
        train_data = get_synthetic_preference_data(n_samples=128)
    if eval_data is None:
        eval_data = get_synthetic_preference_data(n_samples=32, seed=99)

    # JIT compile
    @jax.jit
    def jit_train_step(params, opt_state, batch):
        return dpo_train_step(
            params, ref_params, opt_state, optimizer, batch, model,
            beta=dpo_config.beta, label_smoothing=dpo_config.label_smoothing,
        )

    # Training loop
    tracker = MetricsTracker(log_dir="logs/dpo")
    global_step = 0

    for epoch in range(dpo_config.num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{dpo_config.num_epochs} ---")

        for batch in preference_batch_iterator(
            tokenizer, train_data,
            batch_size=dpo_config.batch_size,
            max_seq_len=dpo_config.max_seq_len,
            seed=epoch,
        ):
            # Convert to JAX arrays and add response masks
            jax_batch = {k: jnp.array(v) for k, v in batch.items()}

            # Create response masks: 1 for response tokens, 0 for prompt
            # For simplicity, use attention mask (improve with actual prompt length tracking)
            jax_batch["chosen_response_mask"] = jax_batch["chosen_attention_mask"]
            jax_batch["rejected_response_mask"] = jax_batch["rejected_attention_mask"]

            params, opt_state, loss, metrics = jit_train_step(params, opt_state, jax_batch)
            global_step += 1

            tracker.log(global_step, {
                "loss": float(loss),
                "accuracy": float(metrics["accuracy"]),
                "reward_margin": float(metrics["reward_margin"]),
                "chosen_reward": float(metrics["chosen_reward"]),
                "rejected_reward": float(metrics["rejected_reward"]),
            })

            if global_step % dpo_config.log_every_steps == 0:
                tracker.print_summary(
                    global_step,
                    keys=["loss", "accuracy", "reward_margin"],
                )

            if global_step % dpo_config.save_every_steps == 0:
                save_checkpoint(params, global_step, dpo_config.checkpoint_dir)

    save_checkpoint(params, global_step, dpo_config.checkpoint_dir)
    tracker.save_csv()

    print(f"\nDPO training complete. Final step: {global_step}")
    return params
