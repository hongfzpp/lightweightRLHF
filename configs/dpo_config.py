"""Configuration for DPO (Direct Preference Optimization) — Stage 3 (Option B).

DPO is a simpler alternative to PPO that does not require a separate reward
model. It directly optimizes the policy using preference pairs.
"""

from dataclasses import dataclass


@dataclass
class DPOConfig:
    """Hyperparameters for DPO training."""
    # Optimizer
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    warmup_steps: int = 50
    max_grad_norm: float = 1.0

    # DPO-specific
    beta: float = 0.1                # DPO temperature parameter
    label_smoothing: float = 0.0     # Label smoothing for DPO loss

    # Training
    batch_size: int = 4
    num_epochs: int = 3
    eval_every_steps: int = 50
    log_every_steps: int = 10

    # Data
    max_seq_len: int = 256

    # Checkpointing
    save_every_steps: int = 200
    checkpoint_dir: str = "checkpoints/dpo"
