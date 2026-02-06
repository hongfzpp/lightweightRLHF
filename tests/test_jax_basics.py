"""Tests for Phase 1: JAX Fundamentals (Exercises 1.1-1.4).

Run with: pytest tests/test_jax_basics.py -v

Each test verifies correctness of the corresponding exercise.
Tests are designed to pass once you've correctly implemented the exercise.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from utils.jax_basics import (
    linear,
    init_linear_params,
    mse_loss,
    compute_gradients,
    train_step,
    per_example_loss,
    batched_loss,
)


# ---------------------------------------------------------------------------
# Exercise 1.1: Pure Functions & Arrays
# ---------------------------------------------------------------------------

class TestExercise1_1:
    """Tests for the linear layer implementation."""

    def test_linear_shape(self):
        """Output should have shape (batch_size, out_dim)."""
        params = {
            'W': jnp.ones((3, 4)),   # out_dim=3, in_dim=4
            'b': jnp.zeros((3,)),
        }
        x = jnp.ones((2, 4))         # batch_size=2, in_dim=4
        y = linear(params, x)
        assert y.shape == (2, 3), f"Expected (2, 3), got {y.shape}"

    def test_linear_identity(self):
        """With identity W and zero b, output should equal input."""
        params = {
            'W': jnp.eye(3),
            'b': jnp.zeros((3,)),
        }
        x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y = linear(params, x)
        np.testing.assert_allclose(y, x, atol=1e-6)

    def test_linear_bias(self):
        """Bias should be added to each example."""
        params = {
            'W': jnp.zeros((2, 3)),
            'b': jnp.array([1.0, 2.0]),
        }
        x = jnp.ones((4, 3))
        y = linear(params, x)
        expected = jnp.broadcast_to(jnp.array([1.0, 2.0]), (4, 2))
        np.testing.assert_allclose(y, expected, atol=1e-6)

    def test_linear_computation(self):
        """Verify specific numerical values."""
        params = {
            'W': jnp.array([[1.0, 2.0], [3.0, 4.0]]),  # (2, 2)
            'b': jnp.array([0.5, -0.5]),
        }
        x = jnp.array([[1.0, 1.0]])  # (1, 2)
        y = linear(params, x)
        # y = [1,1] @ [[1,3],[2,4]] + [0.5, -0.5] = [3, 7] + [0.5, -0.5] = [3.5, 6.5]
        expected = jnp.array([[3.5, 6.5]])
        np.testing.assert_allclose(y, expected, atol=1e-6)

    def test_init_linear_params_shapes(self):
        """init_linear_params should produce correct shapes."""
        rng = jax.random.PRNGKey(0)
        params = init_linear_params(rng, in_dim=4, out_dim=3)
        assert params['W'].shape == (3, 4), f"W shape: {params['W'].shape}"
        assert params['b'].shape == (3,), f"b shape: {params['b'].shape}"

    def test_init_linear_params_bias_zero(self):
        """Bias should be initialized to zeros."""
        rng = jax.random.PRNGKey(0)
        params = init_linear_params(rng, in_dim=4, out_dim=3)
        np.testing.assert_allclose(params['b'], jnp.zeros(3), atol=1e-7)


# ---------------------------------------------------------------------------
# Exercise 1.2: jax.grad
# ---------------------------------------------------------------------------

class TestExercise1_2:
    """Tests for MSE loss and gradient computation."""

    def setup_method(self):
        """Set up common test data."""
        self.params = {
            'W': jnp.array([[1.0, 0.0], [0.0, 1.0]]),  # identity (2x2)
            'b': jnp.zeros((2,)),
        }
        self.x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        self.y_true = jnp.array([[1.0, 2.0], [3.0, 4.0]])  # perfect match

    def test_mse_perfect_prediction(self):
        """MSE should be 0 when predictions match targets exactly."""
        loss = mse_loss(self.params, self.x, self.y_true)
        assert loss.shape == (), f"Loss should be scalar, got shape {loss.shape}"
        np.testing.assert_allclose(float(loss), 0.0, atol=1e-6)

    def test_mse_known_value(self):
        """Verify MSE with known offset."""
        params = {
            'W': jnp.eye(2),
            'b': jnp.array([1.0, 1.0]),  # adds 1 to each output
        }
        x = jnp.array([[0.0, 0.0]])       # pred = [1, 1]
        y_true = jnp.array([[0.0, 0.0]])   # target = [0, 0]
        loss = mse_loss(params, x, y_true)
        # MSE = mean([1^2, 1^2]) = 1.0
        np.testing.assert_allclose(float(loss), 1.0, atol=1e-6)

    def test_gradients_are_pytree(self):
        """Gradients should have the same pytree structure as params."""
        loss_val, grads = compute_gradients(self.params, self.x, self.y_true)
        assert 'W' in grads, "Gradients should contain 'W'"
        assert 'b' in grads, "Gradients should contain 'b'"
        assert grads['W'].shape == self.params['W'].shape
        assert grads['b'].shape == self.params['b'].shape

    def test_gradients_zero_at_optimum(self):
        """Gradients should be zero when predictions are perfect."""
        loss_val, grads = compute_gradients(self.params, self.x, self.y_true)
        np.testing.assert_allclose(float(loss_val), 0.0, atol=1e-6)
        np.testing.assert_allclose(grads['W'], jnp.zeros_like(grads['W']), atol=1e-6)
        np.testing.assert_allclose(grads['b'], jnp.zeros_like(grads['b']), atol=1e-6)


# ---------------------------------------------------------------------------
# Exercise 1.3: jax.jit
# ---------------------------------------------------------------------------

class TestExercise1_3:
    """Tests for the JIT-compiled training step."""

    def test_train_step_reduces_loss(self):
        """A training step should reduce the loss (unless already at 0)."""
        rng = jax.random.PRNGKey(42)
        params = init_linear_params(rng, in_dim=3, out_dim=2)

        x = jax.random.normal(jax.random.PRNGKey(1), (8, 3))
        y = jax.random.normal(jax.random.PRNGKey(2), (8, 2))

        # First step
        new_params, loss1 = train_step(params, x, y)
        # Second step
        _, loss2 = train_step(new_params, x, y)

        assert float(loss2) < float(loss1), (
            f"Loss should decrease: {float(loss1):.4f} -> {float(loss2):.4f}"
        )

    def test_train_step_returns_correct_types(self):
        """train_step should return (params_dict, scalar_loss)."""
        params = {
            'W': jnp.ones((2, 3)),
            'b': jnp.zeros((2,)),
        }
        x = jnp.ones((4, 3))
        y = jnp.zeros((4, 2))

        new_params, loss = train_step(params, x, y)
        assert isinstance(new_params, dict)
        assert 'W' in new_params and 'b' in new_params
        assert loss.shape == (), f"Loss should be scalar, got {loss.shape}"

    def test_train_step_is_jittable(self):
        """train_step should be compilable with jax.jit."""
        params = {
            'W': jnp.ones((2, 3)),
            'b': jnp.zeros((2,)),
        }
        x = jnp.ones((4, 3))
        y = jnp.zeros((4, 2))

        # This should not raise an error
        jitted = jax.jit(train_step)
        new_params, loss = jitted(params, x, y)
        assert loss.shape == ()


# ---------------------------------------------------------------------------
# Exercise 1.4: jax.vmap
# ---------------------------------------------------------------------------

class TestExercise1_4:
    """Tests for vmap-based batched loss computation."""

    def setup_method(self):
        self.params = {
            'W': jnp.eye(2),
            'b': jnp.zeros((2,)),
        }

    def test_per_example_loss_shape(self):
        """per_example_loss should return a scalar for a single example."""
        x = jnp.array([1.0, 2.0])
        y = jnp.array([1.0, 2.0])
        loss = per_example_loss(self.params, x, y)
        assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"

    def test_per_example_loss_zero(self):
        """Loss should be 0 for perfect prediction."""
        x = jnp.array([1.0, 2.0])
        y = jnp.array([1.0, 2.0])
        loss = per_example_loss(self.params, x, y)
        np.testing.assert_allclose(float(loss), 0.0, atol=1e-6)

    def test_batched_loss_shape(self):
        """batched_loss should return per-example losses."""
        x_batch = jnp.ones((5, 2))
        y_batch = jnp.zeros((5, 2))
        losses = batched_loss(self.params, x_batch, y_batch)
        assert losses.shape == (5,), f"Expected (5,), got {losses.shape}"

    def test_batched_loss_values(self):
        """Verify vmapped output matches manually computed values."""
        x_batch = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        y_batch = jnp.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        losses = batched_loss(self.params, x_batch, y_batch)
        # With identity W and zero b: pred = x, loss = sum(x^2)
        expected = jnp.array([1.0, 1.0, 2.0])
        np.testing.assert_allclose(losses, expected, atol=1e-6)

    def test_batched_matches_loop(self):
        """Vmapped result should match a Python loop."""
        rng = jax.random.PRNGKey(0)
        params = init_linear_params(rng, in_dim=4, out_dim=2)
        x_batch = jax.random.normal(jax.random.PRNGKey(1), (10, 4))
        y_batch = jax.random.normal(jax.random.PRNGKey(2), (10, 2))

        # Vmapped
        vmapped_losses = batched_loss(params, x_batch, y_batch)

        # Loop
        loop_losses = jnp.array([
            per_example_loss(params, x_batch[i], y_batch[i])
            for i in range(10)
        ])

        np.testing.assert_allclose(vmapped_losses, loop_losses, atol=1e-5)
