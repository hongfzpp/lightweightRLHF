#!/usr/bin/env python3
"""Run SFT (Supervised Fine-Tuning) — Stage 1 of RLHF.

Usage:
    python scripts/run_sft.py

This script trains the base GPT-2 model on instruction-following data.
The trained model becomes the starting point for PPO/DPO/GRPO.

Prerequisites: Complete Exercises 2.1-2.3 (model) and 3.1-3.3 (SFT).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.jax_utils import check_backend
from configs.model_config import SMALL_CONFIG
from configs.sft_config import SFTConfig
from training.sft_trainer import train_sft


def main():
    print("=" * 60)
    print("Stage 1: Supervised Fine-Tuning (SFT)")
    print("=" * 60)

    backend = check_backend()
    print()

    # Use small config for fast training on M4
    model_config = SMALL_CONFIG
    sft_config = SFTConfig(
        learning_rate=3e-4,
        batch_size=8,
        num_epochs=2,
        log_every_steps=5,
        eval_every_steps=50,
        save_every_steps=100,
    )

    params = train_sft(
        model_config=model_config,
        sft_config=sft_config,
    )

    print("\nSFT complete! Model saved to:", sft_config.checkpoint_dir)
    print("Next step: Run reward model training (scripts/run_reward.py)")


if __name__ == "__main__":
    main()
