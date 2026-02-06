#!/usr/bin/env python3
"""Run Reward Model Training — Stage 2 of RLHF.

Usage:
    python scripts/run_reward.py

This script trains a reward model on preference data (chosen vs rejected).
The trained reward model is used by PPO and GRPO for policy optimization.

Prerequisites: Complete Exercises 4.1-4.3 (reward model).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.jax_utils import check_backend
from configs.model_config import SMALL_CONFIG
from configs.reward_config import RewardConfig
from training.reward_trainer import train_reward_model


def main():
    print("=" * 60)
    print("Stage 2: Reward Model Training")
    print("=" * 60)

    backend = check_backend()
    print()

    model_config = SMALL_CONFIG
    reward_config = RewardConfig(
        learning_rate=1e-4,
        batch_size=4,
        num_epochs=2,
        log_every_steps=5,
        eval_every_steps=50,
        save_every_steps=100,
    )

    params = train_reward_model(
        model_config=model_config,
        reward_config=reward_config,
    )

    print("\nReward model training complete! Saved to:", reward_config.checkpoint_dir)
    print("Next steps:")
    print("  - PPO:  python scripts/run_ppo.py")
    print("  - DPO:  python scripts/run_dpo.py")
    print("  - GRPO: python scripts/run_grpo.py")


if __name__ == "__main__":
    main()
