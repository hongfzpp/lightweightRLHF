#!/usr/bin/env python3
"""Run PPO-based RLHF — Stage 3, Option A.

Usage:
    python scripts/run_ppo.py

This script runs PPO training using:
    - SFT model as the initial policy and reference model
    - Trained reward model for scoring generated responses

Prerequisites: Exercises 5.1-5.5 (PPO), plus a trained SFT model and reward model.

NOTE: For a demo run, this uses fresh random models. In a real pipeline,
you'd load checkpoints from stages 1 and 2.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp

from utils.jax_utils import check_backend
from configs.model_config import SMALL_CONFIG
from configs.ppo_config import PPOConfig
from models.gpt2 import GPT2LMHeadModel
from models.reward_model import RewardModel
from training.ppo_trainer import train_ppo


def main():
    print("=" * 60)
    print("Stage 3A: PPO-based RLHF")
    print("=" * 60)

    backend = check_backend()
    print()

    model_config = SMALL_CONFIG
    ppo_config = PPOConfig(
        learning_rate=1e-5,
        num_iterations=50,
        rollout_batch_size=4,
        max_response_len=32,
        num_ppo_epochs=2,
        log_every_steps=1,
        save_every_steps=25,
    )

    # Initialize fresh models for demo (in production, load from checkpoints)
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, model_config.max_seq_len), dtype=jnp.int32)

    policy_model = GPT2LMHeadModel(config=model_config)
    policy_params = policy_model.init(rng, dummy_input)

    reward_model = RewardModel(config=model_config)
    reward_params = reward_model.init(jax.random.PRNGKey(1), dummy_input)

    print("Starting PPO training...")
    trained_params = train_ppo(
        policy_params=policy_params,
        reward_model_params=reward_params,
        model_config=model_config,
        ppo_config=ppo_config,
    )

    print("\nPPO training complete! Saved to:", ppo_config.checkpoint_dir)


if __name__ == "__main__":
    main()
