"""
Plotting utilities for Meta-RL experiments.

Reads history JSON files saved by trainers and produces:
  - learning curves (training returns over iterations)
  - adaptation gain over time (last_ep - first_ep)
  - evaluation curves on test tasks
  - per-episode adaptation curves within a meta-trial

Usage:
    python scripts/plot_results.py --history results/rl2_ml1_reach/history.json
    python scripts/plot_results.py --history results/sanity_rl2/history.json --out plots/sanity
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def smooth(y, window=5):
    if len(y) < window:
        return y
    return np.convolve(y, np.ones(window) / window, mode="valid")


def plot_training_curves(history, out_dir):
    """Training learning curves: first vs last episode return per trial."""
    if "iter" in history:
        it = history["iter"]
        first = history["first_ep"]
        last = history["last_ep"]
        gain = history.get("adapt_gain", None)
    else:
        it = history["iteration"]
        first = history["train_first_ep_return"]
        last = history["train_last_ep_return"]
        gain = [l - f for l, f in zip(last, first)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(it, first, alpha=0.3, color="tab:blue", label="first ep (raw)")
    axes[0].plot(it, last, alpha=0.3, color="tab:orange", label="last ep (raw)")
    sw = max(1, len(it) // 30)
    s_it = it[:len(smooth(first, sw))] if sw > 1 else it
    axes[0].plot(s_it, smooth(first, sw), color="tab:blue", linewidth=2, label="first ep (smooth)")
    axes[0].plot(s_it, smooth(last, sw), color="tab:orange", linewidth=2, label="last ep (smooth)")
    axes[0].set_xlabel("Meta-iteration")
    axes[0].set_ylabel("Episode return")
    axes[0].set_title("Training: first vs last episode within a trial")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.3)

    axes[1].plot(it, gain, alpha=0.3, color="tab:green")
    axes[1].plot(s_it, smooth(gain, sw), color="tab:green", linewidth=2)
    axes[1].axhline(0, color="black", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Meta-iteration")
    axes[1].set_ylabel("Adaptation gain (last - first)")
    axes[1].set_title("Within-trial adaptation gain")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_loss_curves(history, out_dir):
    """Policy/value loss + entropy."""
    if "policy_loss" not in history or len(history.get("policy_loss", [])) == 0:
        return
    it = history.get("iteration") or history.get("iter")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, title in zip(
        axes,
        ["policy_loss", "value_loss", "entropy"],
        ["Policy loss", "Value loss", "Entropy"],
    ):
        if key in history:
            ax.plot(it, history[key], alpha=0.4)
            sw = max(1, len(it) // 30)
            s_it = it[:len(smooth(history[key], sw))] if sw > 1 else it
            ax.plot(s_it, smooth(history[key], sw), linewidth=2)
            ax.set_xlabel("Meta-iteration")
            ax.set_title(title)
            ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "loss_curves.png")
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_eval_curves(history, out_dir):
    """Evaluation curves from periodic test evaluations."""
    eval_first = history.get("eval_first_ep_return_mean", [])
    eval_last = history.get("eval_last_ep_return_mean", [])
    eval_std = history.get("eval_last_ep_return_std", [])
    if not eval_first:
        return

    its = [x[0] for x in eval_first]
    first_vals = [x[1] for x in eval_first]
    last_vals = [x[1] for x in eval_last]
    std_vals = [x[1] for x in eval_std]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(its, first_vals, "o-", color="tab:blue", label="First episode")
    ax.plot(its, last_vals, "o-", color="tab:orange", label="Last episode (adapted)")
    ax.fill_between(
        its,
        np.array(last_vals) - np.array(std_vals),
        np.array(last_vals) + np.array(std_vals),
        color="tab:orange", alpha=0.2,
    )
    ax.set_xlabel("Meta-iteration")
    ax.set_ylabel("Return (mean over test tasks)")
    ax.set_title("Evaluation on held-out tasks")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "eval_curves.png")
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=str, required=True,
                        help="path to history.json file")
    parser.add_argument("--out", type=str, default=None,
                        help="output directory (default: same as history)")
    args = parser.parse_args()

    with open(args.history) as f:
        history = json.load(f)

    out_dir = args.out or os.path.dirname(args.history)
    os.makedirs(out_dir, exist_ok=True)

    plot_training_curves(history, out_dir)
    plot_loss_curves(history, out_dir)
    plot_eval_curves(history, out_dir)
    print(f"\nDone. Plots in {out_dir}")


if __name__ == "__main__":
    main()
