"""Phase 1: JAX Fundamentals — Exercises 1.1 through 1.4.

==========================================================================
 DIFFICULTY: Easy
 PREREQUISITES: Basic Python, NumPy familiarity
 ESTIMATED TIME: 1-2 hours for all four exercises
==========================================================================

These exercises teach the four core JAX transforms that underpin the entire
RLHF pipeline. Complete them in order before moving to Phase 2.

WHY JAX FOR RLHF?
- Functional paradigm: explicit parameter passing means no hidden state,
  making it easy to maintain separate policy/reference/reward model copies.
- Composable transforms: jit(vmap(grad(f))) "just works" — this is exactly
  what RLHF needs (batched, compiled, differentiated loss functions).
- XLA compilation: the tight inner loops of PPO (rollout -> advantage -> update)
  compile to fused GPU kernels, eliminating Python overhead.
"""

from __future__ import annotations

from typing import Dict, Tuple
import jax
import jax.numpy as jnp


# ============================================================================
# EXERCISE 1.1 — Pure Functions & JAX Arrays
# ============================================================================
#
# GOAL: Implement a simple linear layer y = Wx + b using jax.numpy.
#
# WHY JAX (vs. NumPy):
#   - jax.numpy arrays live on the accelerator (Metal GPU on your M4).
#   - JAX enforces *pure functions*: no in-place mutation, no global state.
#   - This functional style is essential for RLHF because you'll need to
#     maintain multiple copies of model parameters (policy, reference, reward)
#     as separate pytrees — if state were hidden inside objects (like PyTorch
#     nn.Module), cloning and freezing parameters would be much harder.
#
# HINTS:
#   - Use jnp.dot() or the @ operator for matrix multiplication.
#   - Parameters are passed explicitly (not stored in self), because JAX
#     transforms (grad, jit) only work on pure functions.
#
# MATH: y = x @ W^T + b, where x: (batch, in_dim), W: (out_dim, in_dim), b: (out_dim,)
# ============================================================================

def linear(params: Dict[str, jax.Array], x: jax.Array) -> jax.Array:
    """Apply a linear transformation y = xW^T + b.

    Args:
        params: Dictionary with keys:
            'W': Weight matrix of shape (out_dim, in_dim).
            'b': Bias vector of shape (out_dim,).
        x: Input array of shape (batch_size, in_dim).

    Returns:
        Output array of shape (batch_size, out_dim).
    """
    # ---- EXERCISE 1.1: Implement the linear layer ----
    # Replace the line below with: return x @ params['W'].T + params['b']
    return jnp.dot(x, params['W'].T) + params['b']


def init_linear_params(
    rng: jax.Array,
    in_dim: int,
    out_dim: int,
) -> Dict[str, jax.Array]:
    """Initialize parameters for a linear layer.

    Args:
        rng: JAX PRNG key.
        in_dim: Input dimension.
        out_dim: Output dimension.

    Returns:
        Dictionary with 'W' and 'b' arrays.
    """
    # ---- EXERCISE 1.1b: Initialize W with random normal, b with zeros ----
    # HINT: Use jax.random.normal(rng, shape) for W
    #       Use jnp.zeros(shape) for b
    #       Scale W by 0.01 to keep initial outputs small
    return {'W': jax.random.normal(rng, (out_dim, in_dim)) * 0.01, 'b': jnp.zeros(out_dim)}


# ============================================================================
# EXERCISE 1.2 — jax.grad: Automatic Differentiation
# ============================================================================
#
# GOAL: Compute the gradient of MSE loss w.r.t. model parameters.
#
# WHY JAX (vs. PyTorch autograd):
#   - jax.grad works on *any* Python function, not just nn.Module.forward().
#   - It's composable: grad(grad(f)) gives you Hessians for free.
#   - In RLHF, you'll use grad to differentiate complex losses like PPO's
#     clipped objective, DPO's log-ratio loss, etc. The functional API means
#     the loss function is just a normal Python function taking params as input.
#   - No need for .zero_grad() / .backward() / .step() ceremony.
#
# HINTS:
#   - jax.grad(loss_fn)(params, ...) returns a pytree with the same structure
#     as params, where each leaf is the gradient of loss w.r.t. that parameter.
#   - jax.grad differentiates w.r.t. the FIRST argument by default.
#
# MATH: MSE = (1/N) * sum((y_pred - y_true)^2)
# ============================================================================

def mse_loss(
    params: Dict[str, jax.Array],
    x: jax.Array,
    y_true: jax.Array,
) -> jax.Array:
    """Compute Mean Squared Error loss using the linear model.

    Args:
        params: Linear layer parameters (from init_linear_params).
        x: Input array of shape (batch_size, in_dim).
        y_true: Target array of shape (batch_size, out_dim).

    Returns:
        Scalar MSE loss value.
    """
    # ---- EXERCISE 1.2: Implement MSE loss ----
    # Step 1: Compute predictions using the linear() function from Exercise 1.1
    # Step 2: Compute MSE = mean((y_pred - y_true)^2)
    # HINT: Use jnp.mean() and jnp.square() or ** 2
    return jnp.mean(jnp.square(y_true - linear(params, x)))


def compute_gradients(
    params: Dict[str, jax.Array],
    x: jax.Array,
    y_true: jax.Array,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """Compute loss value AND gradients w.r.t. params.

    Args:
        params: Linear layer parameters.
        x: Input array.
        y_true: Target array.

    Returns:
        Tuple of (loss_value, gradients_pytree).
    """
    # ---- EXERCISE 1.2b: Use jax.value_and_grad to get both loss and grads ----
    # HINT: jax.value_and_grad(loss_fn)(params, x, y_true) returns (loss, grads)
    #       where grads has the same pytree structure as params.
    return jax.value_and_grad(mse_loss)(params, x, y_true)


# ============================================================================
# EXERCISE 1.3 — jax.jit: Just-In-Time Compilation
# ============================================================================
#
# GOAL: JIT-compile a full training step (forward + backward + parameter update).
#
# WHY JIT COMPILATION MATTERS:
#
#   Without JIT (eager mode — how normal Python runs):
#     Each math operation (add, multiply, matmul, etc.) is sent to the GPU
#     one at a time. Between every operation, control returns to Python,
#     which decides what to do next. This back-and-forth is slow — imagine
#     a chef who reads one line of a recipe, walks to the kitchen, does that
#     one step, walks back to read the next line, and repeats.
#
#   With JIT (jax.jit):
#     JAX traces your Python function ONCE to discover the full sequence of
#     operations, then hands that entire sequence to the XLA compiler. XLA
#     produces a single optimized program that runs directly on the GPU with
#     no Python involvement. Benefits:
#
#     1. NO PYTHON OVERHEAD — the compiled function runs as native GPU code.
#        Python is only involved on the very first call (to trace & compile).
#
#     2. OPERATION FUSION — XLA merges multiple small operations into fewer
#        big ones. For example, instead of launching separate GPU kernels for
#        (a) multiply weights by inputs, (b) add bias, (c) apply ReLU, XLA
#        can fuse these into a single kernel, avoiding repeated memory reads.
#
#     3. MEMORY OPTIMIZATION — intermediate results that are only used once
#        don't need to be written to GPU memory and read back. XLA keeps them
#        in fast registers/cache instead, which can be 10-100x faster.
#
#     4. HARDWARE-SPECIFIC TUNING — XLA knows the exact hardware (e.g., GPU
#        type, number of cores, memory layout) and tailors the compiled code
#        accordingly — something Python-level code cannot do.
#
#   This matters enormously for training loops: a single training step might
#   involve hundreds of operations (forward pass, loss, backward pass, param
#   update). Without JIT, each of those hundreds of ops pays the Python
#   overhead tax. With JIT, the entire step is one compiled GPU program.
#
# WHY JAX JIT (vs. PyTorch torch.compile):
#   - Unlike torch.compile (which can hit "graph breaks" — places where it
#     falls back to slow Python), jax.jit compiles the entire function
#     reliably as long as the function is pure (no side effects).
#
# HINTS:
#   - @jax.jit decorator or jax.jit(fn) both work.
#   - The function must be PURE: no print(), no mutation, no Python randomness.
#   - All inputs/outputs must be JAX arrays or pytrees of JAX arrays.
#   - The first call is slow (compilation), subsequent calls are fast.
#
# MATH: params_new = params - lr * grad(loss)(params)
# ============================================================================

def train_step(
    params: Dict[str, jax.Array],
    x: jax.Array,
    y_true: jax.Array,
    learning_rate: float = 0.01,
) -> Tuple[Dict[str, jax.Array], jax.Array]:
    """Perform one gradient descent step.

    This function should be JIT-compilable. It computes the loss, computes
    gradients, and returns updated parameters.

    Args:
        params: Current model parameters.
        x: Input batch.
        y_true: Target batch.
        learning_rate: Step size for gradient descent.

    Returns:
        Tuple of (updated_params, loss_value).
    """
    # ---- EXERCISE 1.3: Implement a JIT-compilable training step ----
    # Step 1: Compute loss and gradients (use your compute_gradients or jax.value_and_grad)
    # Step 2: Update each parameter: param_new = param - lr * grad
    # Step 3: Return (new_params, loss)
    #
    # HINT: Use jax.tree.map to apply the update across the entire param pytree:
    #       new_params = jax.tree.map(lambda p, g: p - learning_rate * g, params, grads)
    #
    # After implementing, wrap this function with @jax.jit or call jax.jit(train_step)
    loss, grad = compute_gradients(params, x, y_true)
    new_params = jax.tree.map(lambda p, g: p -learning_rate * g, params, grad)
    return (new_params, loss)


# Create the JIT-compiled version.
# ---- EXERCISE 1.3b: JIT-compile the train_step function ----
# Uncomment the line below after implementing train_step:
train_step_jit = jax.jit(train_step)


# ============================================================================
# EXERCISE 1.4 — jax.vmap: Automatic Vectorization
# ============================================================================
#
# GOAL: Vectorize a per-example function to work over batches automatically.
#
# WHY JAX (vs. manual batching):
#   - jax.vmap transforms a function that works on a single example into one
#     that works on a batch, WITHOUT rewriting the function.
#   - This is heavily used in RLHF:
#     * DPO: vmap over (chosen, rejected) preference pairs
#     * GRPO: vmap over the group dimension (G responses per prompt)
#     * Reward model: vmap inference over multiple candidate responses
#   - The compiler can often optimize vmapped code better than hand-batched
#     code because it knows the operation is embarrassingly parallel.
#
# HINTS:
#   - vmap(fn)(batched_input) — fn takes a single example, vmap handles the batch.
#   - in_axes=(0, None) means "batch over the first arg, broadcast the second".
#   - You can vmap over specific pytree leaves using in_axes with dicts.
#
# MATH: per_example_loss(x_i, y_i) = (linear(params, x_i) - y_i)^2
#        batched_loss = vmap(per_example_loss)(x_batch, y_batch)
# ============================================================================

def per_example_loss(
    params: Dict[str, jax.Array],
    x: jax.Array,
    y_true: jax.Array,
) -> jax.Array:
    """Compute squared error for a SINGLE example (not a batch).

    Args:
        params: Linear layer parameters.
        x: Single input vector of shape (in_dim,).
        y_true: Single target vector of shape (out_dim,).

    Returns:
        Scalar squared error for this example.
    """
    # ---- EXERCISE 1.4: Implement per-example squared error ----
    # HINT: Use the linear() function but note x is now (in_dim,) not (batch, in_dim).
    #       You may need to reshape x: x[None, :] to add a batch dim, then squeeze.
    #       Or implement linear to handle 1D input directly.
    y_pred = linear(params, x)
    return jnp.sum(jnp.square(y_pred - y_true))


def batched_loss(
    params: Dict[str, jax.Array],
    x_batch: jax.Array,
    y_batch: jax.Array,
) -> jax.Array:
    """Use vmap to compute per-example losses over a batch.

    Args:
        params: Linear layer parameters (NOT batched — shared across examples).
        x_batch: Batched inputs of shape (batch_size, in_dim).
        y_batch: Batched targets of shape (batch_size, out_dim).

    Returns:
        Per-example losses of shape (batch_size,).
    """
    # ---- EXERCISE 1.4b: Use jax.vmap to vectorize per_example_loss ----
    # HINT: jax.vmap(per_example_loss, in_axes=(None, 0, 0))(params, x_batch, y_batch)
    #       in_axes=(None, 0, 0) means:
    #         - params: NOT batched (shared across all examples)
    #         - x_batch: batched along axis 0
    #         - y_batch: batched along axis 0
    return jax.vmap(per_example_loss, in_axes=(None, 0, 0))(params, x_batch, y_batch)
