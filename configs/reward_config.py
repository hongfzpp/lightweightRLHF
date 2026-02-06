"""Configuration for Reward Model training — Stage 2 of RLHF."""

from dataclasses import dataclass


@dataclass
class RewardConfig:
    """Hyperparameters for reward model training."""
    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 50
    max_grad_norm: float = 1.0

    # Training
    batch_size: int = 8
    num_epochs: int = 2
    eval_every_steps: int = 50
    log_every_steps: int = 10

    # Data
    max_seq_len: int = 256

    # Checkpointing
    save_every_steps: int = 200
    checkpoint_dir: str = "checkpoints/reward"
