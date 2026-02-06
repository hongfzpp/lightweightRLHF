"""Configuration for Supervised Fine-Tuning (SFT) — Stage 1 of RLHF."""

from dataclasses import dataclass


@dataclass
class SFTConfig:
    """Hyperparameters for SFT training."""
    # Optimizer
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    # Training
    batch_size: int = 16
    num_epochs: int = 3
    eval_every_steps: int = 100
    log_every_steps: int = 10

    # Data
    max_seq_len: int = 256

    # Checkpointing
    save_every_steps: int = 500
    checkpoint_dir: str = "checkpoints/sft"
