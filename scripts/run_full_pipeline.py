#!/usr/bin/env python3
"""Run the FULL RLHF Pipeline: SFT -> Reward Model -> PPO.

Usage:
    python scripts/run_full_pipeline.py

This script runs the complete InstructGPT-style RLHF pipeline end-to-end:
    Stage 1: SFT (supervised fine-tuning)
    Stage 2: Reward model training
    Stage 3: PPO (proximal policy optimization)

Prerequisites: ALL exercises from Phases 2-5 must be implemented.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp

from utils.jax_utils import check_backend, count_params
from configs.model_config import SMALL_CONFIG
from configs.sft_config import SFTConfig
from configs.reward_config import RewardConfig
from configs.ppo_config import PPOConfig
from models.gpt2 import GPT2LMHeadModel
from models.reward_model import RewardModel
from training.sft_trainer import train_sft
from training.reward_trainer import train_reward_model
from training.ppo_trainer import train_ppo


def main():
    print("=" * 60)
    print("FULL RLHF PIPELINE: SFT -> Reward Model -> PPO")
    print("=" * 60)

    backend = check_backend()
    model_config = SMALL_CONFIG

    # =========================================================
    # STAGE 1: Supervised Fine-Tuning
    # =========================================================
    print("\n" + "=" * 60)
    print("STAGE 1: Supervised Fine-Tuning (SFT)")
    print("=" * 60)

    sft_config = SFTConfig(
        learning_rate=3e-4,
        batch_size=8,
        num_epochs=1,
        log_every_steps=10,
        eval_every_steps=50,
        save_every_steps=500,
    )

    sft_params = train_sft(
        model_config=model_config,
        sft_config=sft_config,
    )

    # =========================================================
    # STAGE 2: Reward Model Training
    # =========================================================
    print("\n" + "=" * 60)
    print("STAGE 2: Reward Model Training")
    print("=" * 60)

    reward_config = RewardConfig(
        learning_rate=1e-4,
        batch_size=4,
        num_epochs=1,
        log_every_steps=10,
        save_every_steps=500,
    )

    reward_params = train_reward_model(
        model_config=model_config,
        reward_config=reward_config,
    )

    # =========================================================
    # STAGE 3: PPO Training
    # =========================================================
    print("\n" + "=" * 60)
    print("STAGE 3: PPO-based RLHF")
    print("=" * 60)

    ppo_config = PPOConfig(
        learning_rate=1e-5,
        num_iterations=20,
        rollout_batch_size=4,
        max_response_len=32,
        num_ppo_epochs=2,
        log_every_steps=1,
        save_every_steps=50,
    )

    final_params = train_ppo(
        policy_params=sft_params,
        reward_model_params=reward_params,
        model_config=model_config,
        ppo_config=ppo_config,
    )

    # =========================================================
    # Summary
    # =========================================================
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"Model parameters: {count_params(final_params):,}")
    print(f"Checkpoints saved:")
    print(f"  SFT:    {sft_config.checkpoint_dir}")
    print(f"  Reward: {reward_config.checkpoint_dir}")
    print(f"  PPO:    {ppo_config.checkpoint_dir}")
    print()
    print("Congratulations! You've built a complete RLHF pipeline in JAX.")


if __name__ == "__main__":
    main()
