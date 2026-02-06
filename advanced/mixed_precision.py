"""Phase 8, Exercise 8.2 — Mixed Precision Training (Metal-Compatible).

==========================================================================
 DIFFICULTY: Expert
 PREREQUISITES: Phase 3 (training loops), Phase 2 (model architecture)
 ESTIMATED TIME: 1-2 hours
==========================================================================

GOAL:
    Implement mixed-precision training using float16 forward pass and float32
    gradients. This reduces memory usage and can improve throughput.

    CRITICAL: Apple Metal does NOT support bfloat16. We use float16 instead,
    which requires loss scaling to avoid gradient underflow.

WHY MIXED PRECISION FOR RLHF:
    - RLHF (especially PPO) needs to keep multiple models in memory simultaneously:
      policy, reference model, value network, reward model.
    - float16 halves the memory for forward passes, letting you fit larger models
      or larger batch sizes on your M4's unified memory.
    - The M4's Neural Engine is optimized for float16 operations.

WHY JAX:
    - XLA handles dtype promotion automatically and correctly.
    - Flax modules accept param_dtype and compute_dtype arguments to control
      precision per-module.
    - Loss scaling is a simple wrapper — no AMP context managers like PyTorch.

MIXED PRECISION STRATEGY:
    - Master weights: float32 (used by optimizer for accurate updates)
    - Forward pass: float16 (faster computation, less memory)
    - Gradients: computed in float16, cast to float32 before optimizer step
    - Loss scaling: multiply loss by a large constant before backward pass,
      then divide gradients by the same constant. This prevents float16
      gradient underflow.
"""

from __future__ import annotations

from typing import Any, Tuple, Callable
import jax
import jax.numpy as jnp
import optax

from utils.jax_utils import tree_dtype_cast


# ============================================================================
# EXERCISE 8.2a — Loss Scaling for float16
# ============================================================================
#
# GOAL: Implement static loss scaling to prevent float16 gradient underflow.
#
# WHY NEEDED:
#   float16 has a narrow dynamic range (6e-8 to 65504). Small gradients can
#   underflow to zero. Loss scaling multiplies the loss by a large constant
#   (e.g., 2^15 = 32768) before the backward pass, scaling up all gradients.
#   After the backward pass, divide gradients by the same constant.
#
# WHY JAX:
#   Loss scaling is just arithmetic — no special AMP context needed.
#   jax.tree.map applies the unscaling to all gradient leaves at once.
# ============================================================================

def apply_loss_scaling(loss: jax.Array, scale: float = 32768.0) -> jax.Array:
    """Scale up the loss before backward pass (float16 stability).

    Args:
        loss: Original loss value (scalar).
        scale: Scale factor (default: 2^15).

    Returns:
        Scaled loss.
    """
    # ---- EXERCISE 8.2a: Implement loss scaling ----
    # return loss * scale
    raise NotImplementedError("EXERCISE 8.2a: Implement loss scaling")


def unscale_gradients(grads: Any, scale: float = 32768.0) -> Any:
    """Unscale gradients after backward pass.

    Args:
        grads: Gradient pytree (all leaves are float16 or float32 arrays).
        scale: The same scale factor used in apply_loss_scaling.

    Returns:
        Unscaled gradient pytree (cast to float32).
    """
    # ---- EXERCISE 8.2b: Implement gradient unscaling ----
    # Step 1: Cast gradients to float32 (for accurate optimizer update)
    # Step 2: Divide by scale
    #
    # return jax.tree.map(lambda g: g.astype(jnp.float32) / scale, grads)
    raise NotImplementedError("EXERCISE 8.2b: Implement gradient unscaling")


# ============================================================================
# EXERCISE 8.2c — Mixed Precision Training Step
# ============================================================================
#
# GOAL: Combine float16 forward pass, loss scaling, and float32 optimizer update
#       into a complete mixed-precision training step.
#
# WHY JAX:
#   - tree_dtype_cast converts an entire param pytree to float16 in one call.
#   - The optimizer always operates on the float32 "master" params.
#   - No torch.cuda.amp.autocast() context manager needed — just explicit casts.
# ============================================================================

def mixed_precision_train_step(
    master_params: Any,
    opt_state: Any,
    optimizer: optax.GradientTransformation,
    loss_fn: Callable,
    loss_scale: float = 32768.0,
) -> Tuple[Any, Any, jax.Array]:
    """Perform one mixed-precision training step.

    Args:
        master_params: Float32 master parameters (used by optimizer).
        opt_state: Optimizer state.
        optimizer: Optax optimizer.
        loss_fn: Function (params) -> loss. Will be called with float16 params.
        loss_scale: Static loss scaling factor.

    Returns:
        Tuple of (new_master_params, new_opt_state, loss_value).
    """
    # ---- EXERCISE 8.2c: Implement mixed-precision training step ----
    #
    # Step 1: Cast master params to float16 for the forward pass
    #   fp16_params = tree_dtype_cast(master_params, jnp.float16)
    #
    # Step 2: Define a scaled loss function
    #   def scaled_loss(fp16_params):
    #       loss = loss_fn(fp16_params)
    #       return apply_loss_scaling(loss, loss_scale)
    #
    # Step 3: Compute scaled loss and gradients (in float16)
    #   scaled_loss_val, fp16_grads = jax.value_and_grad(scaled_loss)(fp16_params)
    #
    # Step 4: Unscale gradients and cast to float32
    #   fp32_grads = unscale_gradients(fp16_grads, loss_scale)
    #
    # Step 5: Apply optimizer update to master (float32) params
    #   updates, new_opt_state = optimizer.update(fp32_grads, opt_state, master_params)
    #   new_master_params = optax.apply_updates(master_params, updates)
    #
    # Step 6: Recover the true (unscaled) loss value
    #   loss = scaled_loss_val / loss_scale
    #
    # return new_master_params, new_opt_state, loss

    raise NotImplementedError("EXERCISE 8.2c: Implement the mixed-precision training step")
