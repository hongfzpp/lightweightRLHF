"""Configuration for GRPO (Group Relative Policy Optimization) — Stage 3 (Option C).

GRPO (from DeepSeek) eliminates the need for a value network by using
group-relative advantage estimation. For each prompt, G responses are sampled
and their rewards are normalized within the group.
"""

from dataclasses import dataclass


@dataclass
class GRPOConfig:
    """Hyperparameters for GRPO training."""
    # Optimizer
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # GRPO-specific
    group_size: int = 8              # Number of responses per prompt (G)
    clip_eps: float = 0.2            # PPO-style clipping epsilon
    kl_coeff: float = 0.05           # KL penalty coefficient
    temperature: float = 0.7         # Sampling temperature

    # Training
    num_iterations: int = 200
    batch_size: int = 4              # Number of prompts per batch
    num_update_epochs: int = 1       # Update epochs per batch
    max_response_len: int = 128
    log_every_steps: int = 1
    eval_every_steps: int = 10

    # Checkpointing
    save_every_steps: int = 50
    checkpoint_dir: str = "checkpoints/grpo"
