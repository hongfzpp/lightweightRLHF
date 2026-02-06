"""Configuration for PPO-based RLHF — Stage 3 (Option A).

PPO (Proximal Policy Optimization) is the classic RLHF approach used in
InstructGPT / ChatGPT. It requires a trained reward model and a reference
policy (the SFT model).
"""

from dataclasses import dataclass


@dataclass
class PPOConfig:
    """Hyperparameters for PPO training."""
    # Optimizer
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # PPO-specific
    clip_eps: float = 0.2             # PPO clipping epsilon
    vf_coeff: float = 0.5            # Value function loss coefficient
    entropy_coeff: float = 0.01      # Entropy bonus coefficient
    gamma: float = 1.0               # Discount factor (1.0 for language tasks)
    gae_lambda: float = 0.95         # GAE lambda
    kl_coeff: float = 0.1            # KL penalty coefficient
    kl_target: float = 6.0           # Target KL divergence (for adaptive KL)
    num_ppo_epochs: int = 4          # PPO update epochs per rollout batch
    num_minibatches: int = 4         # Number of minibatches per PPO epoch

    # Rollout
    rollout_batch_size: int = 8      # Number of prompts per rollout
    max_response_len: int = 128      # Max tokens to generate per response
    temperature: float = 0.7         # Sampling temperature during rollout

    # Training
    num_iterations: int = 200        # Number of PPO iterations
    log_every_steps: int = 1
    eval_every_steps: int = 10

    # Checkpointing
    save_every_steps: int = 50
    checkpoint_dir: str = "checkpoints/ppo"
