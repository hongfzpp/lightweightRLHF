"""SFT Trainer: orchestrates supervised fine-tuning.

This is INFRASTRUCTURE code (not an exercise). It calls the SFT loss and
training step functions you implement in algorithms/sft.py, and handles
the epoch loop, evaluation, logging, and checkpointing.
"""

from __future__ import annotations

from typing import Optional
import jax
import jax.numpy as jnp

from configs.model_config import ModelConfig
from configs.sft_config import SFTConfig
from models.gpt2 import GPT2LMHeadModel
from algorithms.sft import create_sft_train_state, sft_train_step, sft_eval_step
from data.sft_dataset import sft_batch_iterator, get_synthetic_sft_data
from data.tokenizer import get_tokenizer
from utils.logging_utils import MetricsTracker
from utils.checkpointing import save_checkpoint
from utils.jax_utils import check_backend, count_params


def train_sft(
    model_config: Optional[ModelConfig] = None,
    sft_config: Optional[SFTConfig] = None,
    train_data=None,
    eval_data=None,
):
    """Run supervised fine-tuning.

    Args:
        model_config: Model architecture configuration.
        sft_config: Training hyperparameters.
        train_data: List of (prompt, completion) tuples. If None, uses synthetic data.
        eval_data: Evaluation data. If None, uses a split of synthetic data.

    Returns:
        Trained model parameters.
    """
    # Defaults
    if model_config is None:
        model_config = ModelConfig()
    if sft_config is None:
        sft_config = SFTConfig()

    # Check device
    check_backend()

    # Create model
    model = GPT2LMHeadModel(config=model_config)

    # Initialize parameters and optimizer
    params, opt_state, optimizer = create_sft_train_state(
        model=model,
        config=model_config,
        learning_rate=sft_config.learning_rate,
        weight_decay=sft_config.weight_decay,
        max_grad_norm=sft_config.max_grad_norm,
        warmup_steps=sft_config.warmup_steps,
    )
    n_params = count_params(params)
    print(f"Model parameters: {n_params:,}")

    # Data
    tokenizer = get_tokenizer()
    if train_data is None:
        train_data = get_synthetic_sft_data(n_samples=128)
    if eval_data is None:
        eval_data = get_synthetic_sft_data(n_samples=32, seed=99)

    # JIT compile the training and eval steps
    # Note: model is captured in the closure, not passed as an argument
    @jax.jit
    def jit_train_step(params, opt_state, batch):
        return sft_train_step(params, opt_state, optimizer, batch, model)

    @jax.jit
    def jit_eval_step(params, batch):
        return sft_eval_step(params, batch, model)

    # Training loop
    tracker = MetricsTracker(log_dir="logs/sft")
    global_step = 0

    for epoch in range(sft_config.num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{sft_config.num_epochs} ---")

        for batch in sft_batch_iterator(
            tokenizer, train_data,
            batch_size=sft_config.batch_size,
            max_seq_len=sft_config.max_seq_len,
            seed=epoch,
        ):
            # Convert numpy arrays to JAX arrays
            jax_batch = {k: jnp.array(v) for k, v in batch.items()}

            # Training step
            params, opt_state, loss = jit_train_step(params, opt_state, jax_batch)
            global_step += 1

            # Logging
            tracker.log(global_step, {"train_loss": float(loss)})
            if global_step % sft_config.log_every_steps == 0:
                tracker.print_summary(global_step, keys=["train_loss"])

            # Evaluation
            if global_step % sft_config.eval_every_steps == 0:
                eval_losses = []
                for eval_batch in sft_batch_iterator(
                    tokenizer, eval_data,
                    batch_size=sft_config.batch_size,
                    max_seq_len=sft_config.max_seq_len,
                ):
                    jax_eval_batch = {k: jnp.array(v) for k, v in eval_batch.items()}
                    eval_loss = jit_eval_step(params, jax_eval_batch)
                    eval_losses.append(float(eval_loss))

                avg_eval_loss = sum(eval_losses) / max(len(eval_losses), 1)
                tracker.log(global_step, {"eval_loss": avg_eval_loss})
                print(f"  Eval loss: {avg_eval_loss:.4f}")

            # Checkpointing
            if global_step % sft_config.save_every_steps == 0:
                save_checkpoint(params, global_step, sft_config.checkpoint_dir)

    # Save final checkpoint
    save_checkpoint(params, global_step, sft_config.checkpoint_dir)
    tracker.save_csv()

    print(f"\nSFT training complete. Final step: {global_step}")
    return params
