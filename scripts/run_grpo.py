#!/usr/bin/env python3
"""Run GRPO (Group Relative Policy Optimization) — Stage 3, Option C.

Usage:
    python scripts/run_grpo.py

GRPO eliminates the value network by using group-relative advantage estimation.
Requires a trained reward model for scoring group members.

Prerequisites: Exercises 7.1-7.3 (GRPO).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp

from utils.jax_utils import check_backend
from configs.model_config import SMALL_CONFIG
from configs.grpo_config import GRPOConfig
from models.gpt2 import GPT2LMHeadModel
from models.reward_model import RewardModel
from training.grpo_trainer import train_grpo


def main():
    print("=" * 60)
    print("Stage 3C: GRPO (Group Relative Policy Optimization)")
    print("=" * 60)

    backend = check_backend()
    print()

    model_config = SMALL_CONFIG
    grpo_config = GRPOConfig(
        learning_rate=1e-5,
        group_size=4,
        batch_size=2,
        num_iterations=50,
        max_response_len=32,
        log_every_steps=1,
        save_every_steps=25,
    )

    # Initialize fresh models for demo
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, model_config.max_seq_len), dtype=jnp.int32)

    policy_model = GPT2LMHeadModel(config=model_config)
    policy_params = policy_model.init(rng, dummy_input)

    reward_model = RewardModel(config=model_config)
    reward_params = reward_model.init(jax.random.PRNGKey(1), dummy_input)

    trained_params = train_grpo(
        policy_params=policy_params,
        reward_model_params=reward_params,
        model_config=model_config,
        grpo_config=grpo_config,
    )

    print("\nGRPO training complete! Saved to:", grpo_config.checkpoint_dir)


if __name__ == "__main__":
    main()
