"""
Metrics for evaluating Meta-RL algorithms.

Computes adaptation curves, robustness statistics, and success rates
(specific to Meta-World, where each step provides a `success` flag in info).
"""
from typing import List
import numpy as np


def episode_return(rewards: np.ndarray, length: int = None) -> float:
    """Sum of rewards over actual episode length (ignoring padding)."""
    if length is not None:
        return float(rewards[:length].sum())
    return float(rewards.sum())


def aggregate_returns(returns_per_task: List[List[float]]) -> dict:
    """
    Compute aggregate statistics across tasks.

    Args:
        returns_per_task: list (over tasks) of lists (over trajectories) of returns

    Returns:
        dict with mean, std, median, min, max, IQR
    """
    per_task_means = np.array([np.mean(r) for r in returns_per_task])
    return {
        "mean": float(per_task_means.mean()),
        "std": float(per_task_means.std()),
        "median": float(np.median(per_task_means)),
        "min": float(per_task_means.min()),
        "max": float(per_task_means.max()),
        "iqr": float(np.percentile(per_task_means, 75) - np.percentile(per_task_means, 25)),
    }


def adaptation_curve(returns_per_step: List[float]) -> dict:
    """
    Build an adaptation curve: returns at adaptation steps 0, 1, 5, 10, 20.

    Args:
        returns_per_step: list of returns ordered by adaptation step

    Returns:
        dict keyed by step index
    """
    return {f"return_at_step_{i}": r for i, r in enumerate(returns_per_step)}
