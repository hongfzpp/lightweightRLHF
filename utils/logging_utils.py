"""Logging and metrics tracking utilities.

Simple metrics accumulator for tracking training progress across the
RLHF pipeline. No external dependencies (e.g., wandb) — just prints
and optional CSV export.

This is infrastructure code — NOT an exercise.
"""

from __future__ import annotations

import csv
import os
import time
from collections import defaultdict
from typing import Any, Dict, Optional


class MetricsTracker:
    """Accumulate and report training metrics.

    Usage:
        tracker = MetricsTracker(log_dir="logs/sft")
        for step in range(num_steps):
            loss = train_step(...)
            tracker.log(step, {"loss": float(loss), "lr": float(lr)})
            if step % 100 == 0:
                tracker.print_summary(step)
        tracker.save_csv()
    """

    def __init__(self, log_dir: Optional[str] = None):
        """Initialize the metrics tracker.

        Args:
            log_dir: Optional directory for saving CSV logs.
        """
        self.history: list[Dict[str, Any]] = []
        self.log_dir = log_dir
        self._start_time = time.time()

    def log(self, step: int, metrics: Dict[str, float]):
        """Record metrics for a given step.

        Args:
            step: Current training step.
            metrics: Dictionary of metric_name -> value.
        """
        record = {"step": step, "elapsed_s": time.time() - self._start_time}
        record.update(metrics)
        self.history.append(record)

    def get_last(self, key: str) -> Optional[float]:
        """Get the most recent value of a metric.

        Args:
            key: Metric name.

        Returns:
            Most recent value, or None if not found.
        """
        for record in reversed(self.history):
            if key in record:
                return record[key]
        return None

    def get_mean(self, key: str, last_n: int = 10) -> Optional[float]:
        """Get the mean of the last N values of a metric.

        Args:
            key: Metric name.
            last_n: Number of recent values to average.

        Returns:
            Mean value, or None if no values found.
        """
        values = [r[key] for r in self.history[-last_n:] if key in r]
        if not values:
            return None
        return sum(values) / len(values)

    def print_summary(self, step: int, keys: Optional[list[str]] = None):
        """Print a formatted summary of recent metrics.

        Args:
            step: Current step.
            keys: Optional list of metric names to print. If None, prints all.
        """
        if not self.history:
            return

        latest = self.history[-1]
        elapsed = latest.get("elapsed_s", 0)

        parts = [f"Step {step:>6d} | {elapsed:>7.1f}s"]
        target_keys = keys or [k for k in latest if k not in ("step", "elapsed_s")]

        for key in target_keys:
            val = latest.get(key)
            if val is not None:
                if isinstance(val, float):
                    parts.append(f"{key}: {val:.4f}")
                else:
                    parts.append(f"{key}: {val}")

        print(" | ".join(parts))

    def save_csv(self, filename: str = "metrics.csv"):
        """Save all metrics history to a CSV file.

        Args:
            filename: Name of the CSV file.
        """
        if not self.history:
            return

        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            path = os.path.join(self.log_dir, filename)
        else:
            path = filename

        # Collect all column names across history
        all_keys = []
        seen = set()
        for record in self.history:
            for key in record:
                if key not in seen:
                    all_keys.append(key)
                    seen.add(key)

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(self.history)

        print(f"Metrics saved to: {path}")
