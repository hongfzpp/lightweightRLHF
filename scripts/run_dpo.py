#!/usr/bin/env python3
"""Run DPO (Direct Preference Optimization) — Stage 3, Option B.

Usage:
    python scripts/run_dpo.py

DPO is a simpler alternative to PPO that does not require a reward model.
It directly optimizes the policy using preference pairs.

Prerequisites: Exercises 6.1-6.3 (DPO).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.jax_utils import check_backend
from configs.model_config import SMALL_CONFIG
from configs.dpo_config import DPOConfig
from training.dpo_trainer import train_dpo


def main():
    print("=" * 60)
    print("Stage 3B: DPO (Direct Preference Optimization)")
    print("=" * 60)

    backend = check_backend()
    print()

    model_config = SMALL_CONFIG
    dpo_config = DPOConfig(
        learning_rate=5e-6,
        beta=0.1,
        batch_size=4,
        num_epochs=2,
        log_every_steps=5,
        save_every_steps=100,
    )

    # Start from fresh model (in production, load SFT checkpoint)
    trained_params = train_dpo(
        sft_params=None,  # Will initialize fresh
        model_config=model_config,
        dpo_config=dpo_config,
    )

    print("\nDPO training complete! Saved to:", dpo_config.checkpoint_dir)


if __name__ == "__main__":
    main()
