"""
Сводные графики для презентации.

Создаёт:
  - steps_vs_performance.png — главный график: cколько шагов потрачено
    и какой return на НОВОМ уровне
  - adaptation_curves.png — адаптация по эпизодам для Meta-RL методов
  - generalization_gap.png — PPO: train vs test уровни
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_steps_vs_performance(summary, out_path):
    """Главный график для слайдов."""
    methods, returns, steps = [], [], []

    if summary.get("ppo"):
        methods.append("PPO\n(1 уровень)")
        returns.append(summary["ppo"]["test_return_mean"])
        steps.append(summary["ppo"]["training_steps"] or 0)
    if summary.get("maml"):
        methods.append("MAML\n(50 уровней)")
        returns.append(summary["maml"]["post_adapt_return"])
        steps.append(summary["maml"]["training_steps"] or 0)
    if summary.get("rl2"):
        methods.append("RL²\n(50 уровней)")
        returns.append(summary["rl2"]["last_episode_return"])
        steps.append(summary["rl2"]["training_steps"] or 0)
    if summary.get("pearl"):
        methods.append("PEARL\n(50 уровней)")
        returns.append(summary["pearl"]["last_episode_return"])
        steps.append(summary["pearl"]["training_steps"] or 0)

    if not methods:
        print("[plot] нет данных для steps_vs_performance"); return

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    colors = ["#7f8c8d", "#9b59b6", "#3498db", "#e74c3c"]
    bars = ax1.bar(methods, returns, color=colors[:len(methods)], alpha=0.85)
    ax1.set_ylabel("Return на НОВОМ уровне", fontsize=12, color="#2c3e50")
    ax1.tick_params(axis="y", labelcolor="#2c3e50")
    ax1.set_ylim(0, max(max(returns), 0.01) * 1.25)
    ax1.grid(axis="y", alpha=0.3)

    for bar, ret in zip(bars, returns):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{ret:.2f}", ha="center", fontsize=11, fontweight="bold")

    ax2 = ax1.twinx()
    ax2.plot(range(len(methods)), [s/1e6 for s in steps],
             "ko--", markersize=12, linewidth=2)
    ax2.set_ylabel("Шагов обучения (млн)", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="black")
    for i, s in enumerate(steps):
        ax2.annotate(f"{s/1e6:.1f}M", (i, s/1e6),
                     textcoords="offset points", xytext=(10, 5), fontsize=10)

    plt.title(f"Производительность на новом уровне Procgen "
              f"({summary.get('game', '?')})\nvs затраты на обучение",
              fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")


def plot_adaptation_curves(summary, out_path):
    """Адаптация по эпизодам внутри meta-trial."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    if summary.get("rl2"):
        rets = summary["rl2"]["returns_per_episode"]
        ax.plot(range(1, len(rets)+1), rets, "o-",
                label="RL²", linewidth=2.5, markersize=10, color="#3498db")
    if summary.get("pearl"):
        rets = summary["pearl"]["returns_per_episode"]
        ax.plot(range(1, len(rets)+1), rets, "s-",
                label="PEARL", linewidth=2.5, markersize=10, color="#e74c3c")
    if summary.get("maml"):
        ax.plot([1, 2], [summary["maml"]["pre_adapt_return"],
                          summary["maml"]["post_adapt_return"]],
                "^-", label="MAML (1 grad step)", linewidth=2.5,
                markersize=10, color="#9b59b6")
    if summary.get("ppo"):
        ax.axhline(summary["ppo"]["test_return_mean"], color="#7f8c8d",
                    linestyle="--", linewidth=2,
                    label=f"PPO baseline (без адаптации)")

    ax.set_xlabel("Номер эпизода в meta-trial", fontsize=12)
    ax.set_ylabel("Return", fontsize=12)
    ax.set_title("Адаптация к новому уровню за несколько эпизодов\n"
                 "(БЕЗ обновления весов модели)", fontsize=13)
    ax.legend(fontsize=11, loc="best")
    ax.grid(alpha=0.3)
    ax.set_xticks(range(1, 6))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")


def plot_generalization_gap(summary, out_path):
    """Только PPO: сравнение train-level vs test-level."""
    if not summary.get("ppo"):
        print("[plot] нет PPO данных для generalization_gap"); return
    ppo = summary["ppo"]
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = [f"Train уровень\n(seed={summary['train_level_for_ppo']})",
              f"НОВЫЙ уровень\n(seed={summary['test_level']})"]
    values = [ppo["train_level_return"], ppo["test_return_mean"]]
    errors = [0, ppo["test_return_std"]]
    colors = ["#27ae60", "#e74c3c"]
    bars = ax.bar(labels, values, yerr=errors, capsize=10,
                   color=colors, alpha=0.85, edgecolor="black")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{v:.2f}", ha="center", fontsize=12, fontweight="bold")
    gap = values[0] - values[1]
    ax.set_title(f"Vanilla PPO: переобучается на свой уровень\n"
                 f"Generalization gap = {gap:.2f}", fontsize=13)
    ax.set_ylabel("Return", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")


def plot_training_curves(results_dir, out_path):
    """Кривые обучения для всех методов: return vs steps."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    results_dir = Path(results_dir)

    method_dirs = {
        "PPO":   list(results_dir.glob("ppo_baseline_level_*")),
        "MAML":  list(results_dir.glob("maml_train_*_levels")),
        "RL²":   list(results_dir.glob("rl2_train_*_levels")),
        "PEARL": list(results_dir.glob("pearl_train_*_levels")),
    }
    colors = {"PPO": "#7f8c8d", "MAML": "#9b59b6",
              "RL²": "#3498db", "PEARL": "#e74c3c"}

    for method, dirs in method_dirs.items():
        if not dirs:
            continue
        hist_path = dirs[0] / "history.json"
        if not hist_path.exists():
            continue
        h = json.load(open(hist_path))
        steps = np.array(h.get("step", [])) / 1e6
        if "mean_return" in h:
            rets = h["mean_return"]
        elif "post_adapt_return" in h:
            rets = h["post_adapt_return"]
        elif "last_ep" in h:
            rets = h["last_ep"]
        else:
            continue
        if len(steps) > 0:
            ax.plot(steps, rets, label=method,
                    linewidth=2, color=colors[method])

    ax.set_xlabel("Шагов обучения (млн)", fontsize=12)
    ax.set_ylabel("Mean episode return (train)", fontsize=12)
    ax.set_title("Кривые обучения", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True)
    args = p.parse_args()

    summary_path = Path(args.results_dir) / "comparison" / "summary.json"
    if not summary_path.exists():
        print(f"[ERR] {summary_path} не найден. "
              f"Сначала запусти scripts/eval_and_record.py")
        return
    summary = json.load(open(summary_path))

    out_dir = Path(args.results_dir) / "comparison"
    plot_steps_vs_performance(summary, out_dir / "steps_vs_performance.png")
    plot_adaptation_curves(summary,    out_dir / "adaptation_curves.png")
    plot_generalization_gap(summary,   out_dir / "generalization_gap.png")
    plot_training_curves(args.results_dir, out_dir / "training_curves.png")


if __name__ == "__main__":
    main()
