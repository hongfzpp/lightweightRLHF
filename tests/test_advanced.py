"""Tests for Phase 8: Advanced Topics (Exercises 8.1-8.3).

Run with: pytest tests/test_advanced.py -v

NOTE: Exercise 8.1 tests require the XLA_FLAGS environment variable to be set
before importing JAX. The multi_device module handles this internally.
"""

import pytest
import os
import numpy as np


# ---------------------------------------------------------------------------
# Exercise 8.1: Multi-Device Training
# ---------------------------------------------------------------------------

class TestExercise8_1:
    """Tests for multi-device sharding."""

    def test_mesh_creation(self):
        """Should create a mesh with 4 simulated devices."""
        # Import after XLA_FLAGS is set by the module
        from advanced.multi_device import setup_mesh
        import jax

        mesh = setup_mesh(n_devices=4)
        assert mesh is not None
        assert len(mesh.devices.flat) == 4

    def test_shard_batch(self):
        """Should shard a batch across devices."""
        from advanced.multi_device import setup_mesh, shard_batch
        import jax
        import jax.numpy as jnp

        mesh = setup_mesh(n_devices=4)
        batch = {
            "input": jnp.ones((8, 4)),   # 8 examples, 4 features
            "target": jnp.zeros((8, 2)),  # 8 examples, 2 outputs
        }
        sharded = shard_batch(batch, mesh)
        assert sharded["input"].shape == (8, 4)
        assert sharded["target"].shape == (8, 2)

    def test_data_parallel_step(self):
        """Data-parallel step should produce finite loss."""
        from advanced.multi_device import setup_mesh, shard_batch, data_parallel_train_step
        import jax
        import jax.numpy as jnp

        mesh = setup_mesh(n_devices=4)
        params = {
            "W": jnp.ones((4, 2)),
            "b": jnp.zeros((2,)),
        }
        batch = {
            "input": jax.random.normal(jax.random.PRNGKey(0), (8, 4)),
            "target": jax.random.normal(jax.random.PRNGKey(1), (8, 2)),
        }
        sharded_batch = shard_batch(batch, mesh)
        new_params, loss = data_parallel_train_step(params, sharded_batch, mesh)
        assert jnp.isfinite(loss)


# ---------------------------------------------------------------------------
# Exercise 8.2: Mixed Precision
# ---------------------------------------------------------------------------

class TestExercise8_2:
    """Tests for mixed-precision training."""

    def test_loss_scaling(self):
        """Loss scaling should multiply loss by scale factor."""
        import jax.numpy as jnp
        from advanced.mixed_precision import apply_loss_scaling

        loss = jnp.array(0.5)
        scaled = apply_loss_scaling(loss, scale=32768.0)
        np.testing.assert_allclose(float(scaled), 0.5 * 32768.0, atol=1.0)

    def test_unscale_gradients(self):
        """Gradient unscaling should divide by scale and cast to float32."""
        import jax.numpy as jnp
        from advanced.mixed_precision import unscale_gradients

        grads = {"W": jnp.array([32768.0, 65536.0], dtype=jnp.float16)}
        unscaled = unscale_gradients(grads, scale=32768.0)
        assert unscaled["W"].dtype == jnp.float32
        np.testing.assert_allclose(unscaled["W"], np.array([1.0, 2.0]), atol=0.01)

    def test_mixed_precision_step(self):
        """Full mixed-precision step should produce a finite loss."""
        import jax
        import jax.numpy as jnp
        import optax
        from advanced.mixed_precision import mixed_precision_train_step

        # Simple model: y = Wx + b
        params = {
            "W": jnp.ones((3, 2), dtype=jnp.float32),
            "b": jnp.zeros((2,), dtype=jnp.float32),
        }
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(params)

        x = jax.random.normal(jax.random.PRNGKey(0), (4, 3))
        y = jax.random.normal(jax.random.PRNGKey(1), (4, 2))

        def loss_fn(fp16_params):
            # Cast data to float16 too
            x_fp16 = x.astype(jnp.float16)
            pred = x_fp16 @ fp16_params["W"] + fp16_params["b"]
            return jnp.mean((pred - y.astype(jnp.float16)) ** 2)

        new_params, new_opt_state, loss = mixed_precision_train_step(
            params, opt_state, optimizer, loss_fn,
        )
        assert jnp.isfinite(loss)
        assert new_params["W"].dtype == jnp.float32  # Master params stay float32


# ---------------------------------------------------------------------------
# Exercise 8.3: Iterative DPO
# ---------------------------------------------------------------------------

class TestExercise8_3:
    """Tests for iterative DPO (basic structure tests)."""

    def test_generate_preferences_structure(self):
        """generate_online_preferences should return correct keys."""
        import jax
        import jax.numpy as jnp
        from configs.model_config import ModelConfig
        from models.gpt2 import GPT2LMHeadModel
        from models.reward_model import RewardModel
        from advanced.iterative_dpo import generate_online_preferences

        config = ModelConfig(vocab_size=100, max_seq_len=32, n_layers=1, n_heads=2, d_model=32, d_ff=128)
        model = GPT2LMHeadModel(config=config)
        reward_model = RewardModel(config=config)

        rng = jax.random.PRNGKey(0)
        prompt_ids = jnp.ones((2, 4), dtype=jnp.int32)

        policy_params = model.init(rng, prompt_ids)
        reward_params = reward_model.init(rng, prompt_ids)

        result = generate_online_preferences(
            model, policy_params, reward_model, reward_params,
            prompt_ids, rng, max_new_tokens=8,
        )

        assert "chosen_ids" in result
        assert "rejected_ids" in result
        assert "chosen_rewards" in result
        assert "rejected_rewards" in result
        assert result["chosen_ids"].shape[0] == 2
        assert result["rejected_ids"].shape[0] == 2

    def test_chosen_reward_geq_rejected(self):
        """Chosen responses should have >= reward than rejected."""
        import jax
        import jax.numpy as jnp
        from configs.model_config import ModelConfig
        from models.gpt2 import GPT2LMHeadModel
        from models.reward_model import RewardModel
        from advanced.iterative_dpo import generate_online_preferences

        config = ModelConfig(vocab_size=100, max_seq_len=32, n_layers=1, n_heads=2, d_model=32, d_ff=128)
        model = GPT2LMHeadModel(config=config)
        reward_model = RewardModel(config=config)

        rng = jax.random.PRNGKey(42)
        prompt_ids = jnp.ones((4, 4), dtype=jnp.int32)

        policy_params = model.init(rng, prompt_ids)
        reward_params = reward_model.init(rng, prompt_ids)

        result = generate_online_preferences(
            model, policy_params, reward_model, reward_params,
            prompt_ids, rng, max_new_tokens=8,
        )

        assert (result["chosen_rewards"] >= result["rejected_rewards"]).all()
