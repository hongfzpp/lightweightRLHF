"""Phase 8, Exercise 8.1 — Multi-Device Training with jax.sharding.

==========================================================================
 DIFFICULTY: Expert
 PREREQUISITES: Phases 1-7 complete
 ESTIMATED TIME: 2-3 hours
==========================================================================

GOAL:
    Learn JAX's modern sharding API for distributing computation across
    multiple devices. On your M4 Mac Studio (single Metal GPU), we simulate
    multi-device using CPU-based device simulation.

    The sharding API is the MODERN replacement for jax.pmap. The same code
    works on 1 Metal GPU, 8 CUDA GPUs, or a TPU pod slice.

WHY JAX (vs. PyTorch DDP / FSDP):
    - JAX's sharding is declarative: you specify WHERE data/params should
      live, and the XLA compiler automatically handles communication.
    - In PyTorch, you need DistributedDataParallel, torch.distributed,
      process groups, etc. — hundreds of lines of boilerplate.
    - JAX's approach: 3 lines to go from single-device to multi-device:
        1. Create a mesh: Mesh(devices, axis_names)
        2. Shard data: jax.device_put(data, NamedSharding(mesh, PartitionSpec('data')))
        3. JIT with sharding: jax.jit(f, in_shardings=..., out_shardings=...)
    - This is critical for RLHF because large language models MUST be distributed.

NOTE: Apple M4 has a single Metal GPU. We simulate 4 "devices" using:
    import os
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    This MUST be set before importing JAX.

SHARDING PATTERNS FOR RLHF:
    - Data parallelism: replicate model, shard data across devices
    - Tensor parallelism: shard model weights (for very large models)
    - Pipeline parallelism: shard model layers across devices
    - In practice, RLHF uses data parallelism + tensor parallelism (FSDP-like)
"""

from __future__ import annotations

import os
# !! MUST be set before importing JAX !!
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
os.environ["JAX_PLATFORM_NAME"] = "cpu"  # Use CPU backend for simulation

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from typing import Any, Tuple, Dict


# ============================================================================
# EXERCISE 8.1a — Basic Sharding: Data Parallelism
# ============================================================================
#
# GOAL: Distribute a simple computation across simulated devices using
#       JAX's sharding API. This is the foundation for scaling RLHF training.
#
# WHY JAX:
#   - NamedSharding + PartitionSpec = declarative parallelism.
#   - The XLA compiler automatically inserts all-reduce operations for gradients.
#   - Same code runs on 1 device or 1000 devices — just change the mesh.
#
# STEPS:
#   1. Create a device mesh
#   2. Shard input data across the mesh
#   3. JIT-compile a function with sharding constraints
#   4. Verify that computation is distributed
# ============================================================================

def setup_mesh(n_devices: int = 4) -> Mesh:
    """Create a device mesh for data parallelism.

    Args:
        n_devices: Number of devices (should match XLA_FLAGS setting).

    Returns:
        A JAX Mesh with axis name 'data'.
    """
    # ---- EXERCISE 8.1a: Create a device mesh ----
    #
    # Step 1: Get available devices
    #   devices = jax.devices()[:n_devices]
    #   assert len(devices) == n_devices, f"Expected {n_devices} devices, got {len(devices)}"
    #
    # Step 2: Create mesh with a single 'data' axis (data parallelism)
    #   device_array = np.array(devices).reshape(n_devices)
    #   mesh = Mesh(device_array, axis_names=('data',))
    #   return mesh

    raise NotImplementedError("EXERCISE 8.1a: Create a device mesh for data parallelism")


def shard_batch(
    batch: Dict[str, jax.Array],
    mesh: Mesh,
) -> Dict[str, jax.Array]:
    """Shard a data batch across the 'data' axis of the mesh.

    Each device gets batch_size/n_devices examples.

    Args:
        batch: Dictionary of arrays, each with batch dimension along axis 0.
        mesh: Device mesh.

    Returns:
        Sharded batch (same structure, but arrays distributed across devices).
    """
    # ---- EXERCISE 8.1b: Shard data across devices ----
    #
    # Use NamedSharding to specify that axis 0 (batch) should be split
    # across the 'data' axis of the mesh:
    #
    #   data_sharding = NamedSharding(mesh, P('data'))
    #   sharded_batch = {}
    #   for key, value in batch.items():
    #       sharded_batch[key] = jax.device_put(value, data_sharding)
    #   return sharded_batch

    raise NotImplementedError("EXERCISE 8.1b: Shard a batch across devices")


def data_parallel_train_step(
    params: Any,
    batch: Dict[str, jax.Array],
    mesh: Mesh,
) -> Tuple[Any, jax.Array]:
    """A simple data-parallel training step using sharding.

    Demonstrates how to:
    1. Replicate params across all devices
    2. Shard data across devices
    3. Let XLA handle gradient all-reduce automatically

    Args:
        params: Model parameters (will be replicated).
        batch: Sharded data batch.
        mesh: Device mesh.

    Returns:
        Tuple of (updated_params, loss).
    """
    # ---- EXERCISE 8.1c: Implement data-parallel training step ----
    #
    # Step 1: Define sharding for params (replicated = same on all devices)
    #   replicated = NamedSharding(mesh, P())  # No partitioning = replicated
    #   params = jax.device_put(params, replicated)
    #
    # Step 2: Define train function (same as single-device)
    #   def train_fn(params, batch):
    #       # Simple loss: MSE for demonstration
    #       x = batch['input']
    #       y = batch['target']
    #       pred = x @ params['W'] + params['b']
    #       loss = jnp.mean((pred - y) ** 2)
    #       return loss
    #
    # Step 3: Compute loss and gradients with sharding
    #   loss, grads = jax.value_and_grad(train_fn)(params, batch)
    #
    # Step 4: Simple SGD update (in practice, use Optax)
    #   new_params = jax.tree.map(lambda p, g: p - 0.01 * g, params, grads)
    #   return new_params, loss
    #
    # NOTE: When this is JIT-compiled, XLA automatically:
    #   - Splits the batch across devices (sharded data)
    #   - Computes local gradients on each device
    #   - All-reduces the gradients across devices
    #   - Updates the replicated params identically on all devices
    # No manual gradient synchronization needed!

    raise NotImplementedError("EXERCISE 8.1c: Implement a data-parallel training step")
