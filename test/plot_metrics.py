"""
Построение графиков по metrics.json для отчёта и презентации.

Создаёт:
  1. adaptation_speed.png      — кривые адаптации ep1→ep5 для каждого Meta-RL метода
  2. robustness_boxplot.png    — box plot устойчивости на 30 тестовых уровнях
  3. robustness_bar.png        — mean ± std для устойчивости
  4. generalization.png        — near / mid / far grouped bar
  5. comparison_all.png        — сводная панель 2×2 со всеми графиками
  6. summary_table.png         — таблица для слайда

Использование:
    python plot_metrics.py --input metrics.json --output plots/
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Стиль ──────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#fafafa",
    "axes.edgecolor":   "#cccccc",
    "grid.color":       "#e0e0e0",
    "grid.linewidth":   0.7,
    "font.family":      "sans-serif",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   12,
})

COLORS = {
    "ppo":   "#95a5a6",
    "maml":  "#8e44ad",
    "rl2":   "#2980b9",
    "pearl": "#c0392b",
}
LABELS = {
    "ppo":   "PPO (baseline)",
    "maml":  "MAML (FOMAML)",
    "rl2":   "RL²",
    "pearl": "PEARL",
}
MARKERS = {"ppo": "x", "maml": "^", "rl2": "o", "pearl": "s"}
METHOD_ORDER = ["ppo", "maml", "pearl", "rl2"]  # порядок: от худшего к лучшему


def get_methods(data_dict):
    return [m for m in METHOD_ORDER if m in data_dict]


# ── 1. Скорость адаптации ──────────────────────

def plot_adaptation_speed(speed, out_path):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    meta_methods = [m for m in ["maml", "pearl", "rl2"] if m in speed]

    for method in meta_methods:
        d = speed[method]
        mean = np.array(d["mean"])
        std  = np.array(d["std"])
        x = np.arange(1, len(mean) + 1)
        ax.plot(x, mean, f"{MARKERS[method]}-", label=LABELS[method],
                color=COLORS[method], linewidth=2.5, markersize=9, zorder=3)
        ax.fill_between(x, mean - std, mean + std,
                         alpha=0.12, color=COLORS[method])

    # PPO baseline горизонталь
    if "ppo" in speed:
        ppo_m = speed["ppo"]["mean"]
        ppo_s = speed["ppo"]["std"]
        max_ep = max(len(speed[m]["mean"]) for m in meta_methods) if meta_methods else 5
        ax.axhline(ppo_m, color=COLORS["ppo"], linestyle="--", linewidth=2,
                    label=f"PPO baseline ({ppo_m:.2f} ± {ppo_s:.2f})", zorder=2)
        ax.axhspan(ppo_m - ppo_s, ppo_m + ppo_s, alpha=0.06, color=COLORS["ppo"])

    ax.set_xlabel("Эпизод адаптации")
    ax.set_ylabel("Return")
    ax.set_title("Скорость адаптации к новым задачам")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.5)
    ax.set_xticks(range(1, 6))
    ax.set_xlim(0.7, 5.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [1] {out_path}")


# ── 2. Устойчивость (box plot) ─────────────────

def plot_robustness_box(rob, out_path):
    fig, ax = plt.subplots(figsize=(8, 5.5))

    methods = get_methods(rob)
    data = [rob[m]["per_level"] for m in methods]
    labels = [LABELS[m] for m in methods]

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5,
                     medianprops={"color": "black", "linewidth": 2},
                     whiskerprops={"linewidth": 1.2},
                     flierprops={"markersize": 5})
    for patch, m in zip(bp["boxes"], methods):
        patch.set_facecolor(COLORS[m])
        patch.set_alpha(0.65)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.2)

    # Подписи mean ± std
    for i, m in enumerate(methods):
        mu, sigma = rob[m]["mean"], rob[m]["std"]
        ax.text(i + 1, ax.get_ylim()[1] * 0.02 + max(rob[m]["per_level"]) + 0.3,
                f"μ={mu:.2f}, σ={sigma:.2f}",
                ha="center", fontsize=9, fontstyle="italic")

    ax.set_ylabel("Return (post-adaptation)")
    ax.set_title("")
    ax.grid(axis="y", alpha=0.4)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [2] {out_path}")


# ── 3. Устойчивость (bar) ──────────────────────

def plot_robustness_bar(rob, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    methods = get_methods(rob)
    means = [rob[m]["mean"] for m in methods]
    stds  = [rob[m]["std"]  for m in methods]
    x = np.arange(len(methods))
    colors = [COLORS[m] for m in methods]

    bars = ax.bar(x, means, yerr=stds, capsize=8, color=colors,
                   alpha=0.8, edgecolor="black", linewidth=0.8, width=0.55)
    for bar, mu, sigma in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + sigma + 0.15,
                f"{mu:.2f} ± {sigma:.2f}",
                ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in methods])
    ax.set_ylabel("Return (mean ± std)")
    ax.set_title("Устойчивость: среднее и разброс\n(меньше σ при высоком μ = лучше)")
    ax.grid(axis="y", alpha=0.4)
    ax.set_ylim(0, max(means) * 1.35)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [3] {out_path}")


# ── 4. Обобщение ───────────────────────────────

def plot_generalization(gen, out_path):
    fig, ax = plt.subplots(figsize=(10, 5.5))

    groups = ["near", "mid", "far"]
    group_labels = [
        "Близкие к train\n(Δseed ≈ 0–100)",
        "Средняя дистанция\n(Δseed ≈ 1000)",
        "Далёкие\n(Δseed ≈ 5000)",
    ]
    methods = get_methods(gen)
    n = len(methods)
    width = 0.18
    x = np.arange(len(groups))

    for i, m in enumerate(methods):
        means = [gen[m][g]["mean"] for g in groups]
        stds  = [gen[m][g]["std"]  for g in groups]
        offset = (i - n/2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, capsize=4,
                       label=LABELS[m], color=COLORS[m], alpha=0.8,
                       edgecolor="black", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=11)
    ax.set_ylabel("Return (post-adaptation)")
    ax.set_title("Обобщение: производительность vs дистанция от train-распределения\n"
                  "(хороший метод падает медленно)")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.4)
    ax.set_ylim(0, 11.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [4] {out_path}")


# ── 5. Сводная панель ──────────────────────────

def plot_all_combined(speed, rob, gen, config, out_path):
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.32, wspace=0.28)

    # ── (0,0) Адаптация ──
    ax = fig.add_subplot(gs[0, 0])
    meta = [m for m in ["maml", "pearl", "rl2"] if m in speed]
    for m in meta:
        d = speed[m]
        mn = np.array(d["mean"])
        sd = np.array(d["std"])
        xx = np.arange(1, len(mn)+1)
        ax.plot(xx, mn, f"{MARKERS[m]}-", label=LABELS[m],
                color=COLORS[m], linewidth=2, markersize=7)
        ax.fill_between(xx, mn-sd, mn+sd, alpha=0.1, color=COLORS[m])
    if "ppo" in speed:
        ax.axhline(speed["ppo"]["mean"], color=COLORS["ppo"],
                    linestyle="--", linewidth=1.8, label="PPO baseline")
    ax.set_xlabel("Эпизод адаптации")
    ax.set_ylabel("Return")
    ax.set_title("(a) Скорость адаптации")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.4)
    ax.set_xticks(range(1, 6))
    ax.set_ylim(bottom=0)

    # ── (0,1) Устойчивость (box) ──
    ax = fig.add_subplot(gs[0, 1])
    methods = get_methods(rob)
    data = [rob[m]["per_level"] for m in methods]
    bp = ax.boxplot(data, labels=[LABELS[m] for m in methods],
                     patch_artist=True, widths=0.5,
                     medianprops={"color": "black", "linewidth": 1.8})
    for patch, m in zip(bp["boxes"], methods):
        patch.set_facecolor(COLORS[m])
        patch.set_alpha(0.6)
    ax.set_ylabel("Return (post-adapt)")
    ax.set_title("(b) Устойчивость (30 тестовых уровней)")
    ax.grid(axis="y", alpha=0.4)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis='x', labelsize=9)

    # ── (1,0) Обобщение ──
    ax = fig.add_subplot(gs[1, 0])
    groups = ["near", "mid", "far"]
    gl = ["Близко", "Средне", "Далеко"]
    methods_g = get_methods(gen)
    ng = len(methods_g)
    w = 0.18
    xg = np.arange(3)
    for i, m in enumerate(methods_g):
        ms_ = [gen[m][g]["mean"] for g in groups]
        ss_ = [gen[m][g]["std"]  for g in groups]
        offset = (i - ng/2 + 0.5) * w
        ax.bar(xg + offset, ms_, w, yerr=ss_, capsize=3,
                label=LABELS[m], color=COLORS[m], alpha=0.8,
                edgecolor="black", linewidth=0.5)
    ax.set_xticks(xg)
    ax.set_xticklabels(gl)
    ax.set_ylabel("Return (post-adapt)")
    ax.set_title("(c) Обобщение (near → far)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.4)
    ax.set_ylim(0, 11.5)

    # ── (1,1) Сводная таблица ──
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    columns = ["Метод", "Adapt.\ngain", "Устойч.\n(μ ± σ)",
               "Обобщ.\nΔ(near→far)", "Шагов\nобуч."]
    rows_data = []
    steps_map = config.get("total_training_steps", {})
    for m in ["ppo", "maml", "pearl", "rl2"]:
        if m not in rob:
            continue
        # Adaptation gain
        if m in speed and isinstance(speed[m].get("mean"), list):
            mn = speed[m]["mean"]
            gain = f"{mn[-1] - mn[0]:+.2f}"
        else:
            gain = "—"
        # Robustness
        robust = f"{rob[m]['mean']:.2f} ± {rob[m]['std']:.2f}"
        # Generalization drop
        if m in gen:
            drop = gen[m]["near"]["mean"] - gen[m]["far"]["mean"]
            drop_str = f"–{drop:.2f}" if drop >= 0 else f"+{abs(drop):.2f}"
        else:
            drop_str = "—"
        # Steps
        s = steps_map.get(m, "—")
        if isinstance(s, (int, float)):
            s = f"{s/1e6:.0f}M"
        rows_data.append([LABELS[m], gain, robust, drop_str, s])

    table = ax.table(cellText=rows_data, colLabels=columns,
                      loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 2.2)
    for j in range(len(columns)):
        cell = table[(0, j)]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold", fontsize=9)
    method_order_table = [m for m in ["ppo", "maml", "pearl", "rl2"] if m in rob]
    for i, m in enumerate(method_order_table, 1):
        for j in range(len(columns)):
            table[(i, j)].set_facecolor(COLORS[m] + "22")
    ax.set_title("(d) Сводное сравнение", fontsize=13)

    fig.suptitle("Сравнительное исследование методов Meta-RL на Procgen CoinRun",
                  fontsize=15, fontweight="bold", y=0.98)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [5] {out_path}")


# ── 6. Таблица для слайда ──────────────────────

def plot_summary_table(speed, rob, gen, config, out_path):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    steps_map = config.get("total_training_steps", {})
    columns = [
        "Метод",
        "Прирост\nадаптации\n(ep1 → ep5)",
        "Устойчивость\n(μ ± σ, 30 уровней)",
        "Падение\nобобщения\n(near → far)",
        "Шагов\nобучения",
    ]

    rows = []
    for m in ["ppo", "maml", "pearl", "rl2"]:
        if m not in rob:
            continue
        if m in speed and isinstance(speed[m].get("mean"), list):
            mn = speed[m]["mean"]
            gain = f"+{mn[-1] - mn[0]:.2f}"
        else:
            gain = "—"
        robust = f"{rob[m]['mean']:.2f} ± {rob[m]['std']:.2f}"
        if m in gen:
            d = gen[m]["near"]["mean"] - gen[m]["far"]["mean"]
            drop = f"–{d:.2f}" if d >= 0 else f"+{abs(d):.2f}"
        else:
            drop = "—"
        s = steps_map.get(m, "—")
        if isinstance(s, (int, float)):
            s = f"{s/1e6:.0f}M"
        rows.append([LABELS[m], gain, robust, drop, s])

    table = ax.table(cellText=rows, colLabels=columns,
                      loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.8)

    for j in range(len(columns)):
        cell = table[(0, j)]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold", fontsize=10)

    method_order_t = [m for m in ["ppo", "maml", "pearl", "rl2"] if m in rob]
    for i, m in enumerate(method_order_t, 1):
        for j in range(len(columns)):
            table[(i, j)].set_facecolor(COLORS[m] + "22")

    # Выделим лучшие значения жирным (RL² в робастности, PEARL в адаптации near)
    # Оставлю как есть — для презентации можно кастомизировать

    plt.title("Сводное сравнение методов Meta-RL\n"
              "Procgen CoinRun, 200 train levels, 30 test levels",
              fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [6] {out_path}")


# ── 7. Дополнительно: per-trial scatter ────────

def plot_adaptation_scatter(speed, out_path):
    """Scatter plot всех отдельных триалов — показывает вариативность."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, m in zip(axes, ["maml", "pearl", "rl2"]):
        if m not in speed or "per_trial" not in speed[m]:
            ax.set_visible(False)
            continue
        trials = np.array(speed[m]["per_trial"])
        n_trials, n_eps = trials.shape
        for i in range(n_trials):
            ax.plot(range(1, n_eps+1), trials[i], "-",
                    color=COLORS[m], alpha=0.15, linewidth=1)
        mean = np.array(speed[m]["mean"])
        std  = np.array(speed[m]["std"])
        xx = np.arange(1, len(mean)+1)
        ax.plot(xx, mean, f"{MARKERS[m]}-", color=COLORS[m],
                linewidth=3, markersize=10, zorder=5, label=f"Среднее")
        ax.fill_between(xx, mean-std, mean+std, alpha=0.2, color=COLORS[m])
        ax.set_xlabel("Эпизод адаптации")
        if ax == axes[0]:
            ax.set_ylabel("Return")
        ax.set_title(LABELS[m], fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, n_eps+1))
        ax.set_ylim(-0.5, 10.5)
        ax.legend(fontsize=9)

    fig.suptitle("Адаптация по отдельным триалам (каждая линия = один тест-уровень)",
                  fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [7] {out_path}")


# ── Main ───────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="metrics.json",
                   help="Путь к metrics.json")
    p.add_argument("--output", default="plots",
                   help="Папка для сохранения графиков")
    args = p.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    with open(args.input) as f:
        data = json.load(f)

    speed  = data.get("adaptation_speed", {})
    rob    = data.get("robustness", {})
    gen    = data.get("generalization", {})
    config = data.get("config", {})

    print(f"\nМетрики загружены из {args.input}")
    print(f"Методы: {list(rob.keys())}")
    print(f"Графики сохраняются в {out}/\n")

    # Отдельные графики
    plot_adaptation_speed(speed,              out / "adaptation_speed.png")
    plot_robustness_box(rob,                  out / "robustness_boxplot.png")
    plot_robustness_bar(rob,                  out / "robustness_bar.png")
    plot_generalization(gen,                  out / "generalization.png")
    plot_adaptation_scatter(speed,            out / "adaptation_scatter.png")

    # Сводные
    plot_all_combined(speed, rob, gen, config, out / "comparison_all.png")
    plot_summary_table(speed, rob, gen, config, out / "summary_table.png")

    print(f"\nГотово! {len(list(out.glob('*.png')))} графиков в {out}/")


if __name__ == "__main__":
    main()
