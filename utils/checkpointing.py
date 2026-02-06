"""Checkpointing utilities using Orbax.

Save and load model parameters and optimizer state at each RLHF stage.
This is infrastructure code — NOT an exercise.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import jax
import orbax.checkpoint as ocp


def save_checkpoint(
    params: Any,
    step: int,
    checkpoint_dir: str,
    opt_state: Optional[Any] = None,
) -> str:
    """Save model parameters (and optionally optimizer state) to disk.

    Args:
        params: Model parameters pytree.
        step: Current training step (used for directory naming).
        checkpoint_dir: Base directory for checkpoints.
        opt_state: Optional optimizer state to save alongside params.

    Returns:
        Path to the saved checkpoint directory.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    ckpt = {"params": params}
    if opt_state is not None:
        ckpt["opt_state"] = opt_state

    checkpointer = ocp.StandardCheckpointer()
    save_path = os.path.join(checkpoint_dir, f"step_{step}")
    checkpointer.save(save_path, ckpt)
    print(f"Checkpoint saved: {save_path}")
    return save_path


def load_checkpoint(
    checkpoint_path: str,
    target: Optional[Any] = None,
) -> Any:
    """Load a checkpoint from disk.

    Args:
        checkpoint_path: Path to the checkpoint directory.
        target: Optional target pytree structure for restoration.

    Returns:
        Restored checkpoint dictionary with 'params' (and optionally 'opt_state').
    """
    checkpointer = ocp.StandardCheckpointer()
    restored = checkpointer.restore(checkpoint_path, target)
    print(f"Checkpoint loaded: {checkpoint_path}")
    return restored


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the latest checkpoint in a directory by step number.

    Args:
        checkpoint_dir: Base checkpoint directory.

    Returns:
        Path to the latest checkpoint, or None if no checkpoints exist.
    """
    if not os.path.exists(checkpoint_dir):
        return None

    step_dirs = []
    for name in os.listdir(checkpoint_dir):
        if name.startswith("step_"):
            try:
                step = int(name.split("_")[1])
                step_dirs.append((step, os.path.join(checkpoint_dir, name)))
            except ValueError:
                continue

    if not step_dirs:
        return None

    step_dirs.sort(key=lambda x: x[0])
    return step_dirs[-1][1]
