"""
Экспериментальное исследование методов Meta-RL по трём критериям:

1. СКОРОСТЬ АДАПТАЦИИ — return по эпизодам внутри meta-trial
2. УСТОЙЧИВОСТЬ     — mean ± std на 20 тестовых уровнях
3. ОБОБЩЕНИЕ        — performance в зависимости от "дистанции" от train

Использование:
    python scripts/run_experiments.py \
        --results-dir results/procgen_coinrun_demo \
        --game coinrun \
        --num-train-levels 20 \
        --methods ppo rl2 maml

Создаёт:
    results/procgen_coinrun_demo/experiments/
        metrics.json           — все сырые данные
        adaptation_speed.png
        robustness.png
        generalization.png
        summary_table.png      — сводная таблица для слайда
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.procgen_env import ProcgenTaskWrapper
from algorithms.ppo_baseline.policy import PPOPolicy
from algorithms.rl2.procgen_trainer import RL2Policy


# ──────────────────────────────────────────────
# Rollout-функции
# ──────────────────────────────────────────────

def rollout_ppo(policy, env_name, level, max_steps=500):
    env = ProcgenTaskWrapper(env_name=env_name, level_seed=level)
    obs = env.reset()
    ret, steps = 0.0, 0
    done = False
    while not done and steps < max_steps:
        with torch.no_grad():
            action, _, _, _ = policy.act(torch.from_numpy(obs))
        obs, r, done, _ = env.step(int(action.item()))
        ret += r; steps += 1
    env.close()
    return ret, steps


def rollout_rl2_trial(policy, env_name, level, num_episodes=4, max_steps=400):
    """Возвращает список return по эпизодам."""
    env = ProcgenTaskWrapper(env_name=env_name, level_seed=level)
    hidden = policy.init_hidden(batch_size=1)
    prev_action = torch.zeros(1, dtype=torch.long)
    prev_reward = torch.zeros(1)
    prev_done   = torch.ones(1)
    ep_returns = []
    for _ in range(num_episodes):
        obs = env.reset()
        ret, done, steps = 0.0, False, 0
        while not done and steps < max_steps:
            with torch.no_grad():
                action, _, _, hidden = policy.act_single(
                    torch.from_numpy(obs), prev_action, prev_reward, prev_done, hidden)
            obs, r, done, _ = env.step(int(action.item()))
            prev_action = action
            prev_reward = torch.tensor([r])
            prev_done   = torch.tensor([float(done)])
            ret += r; steps += 1
        ep_returns.append(ret)
    env.close()
    return ep_returns


def rollout_maml_trial(policy_class, weights, env_name, level,
                        inner_lr=0.05, rollout_steps=64, max_steps=400):
    """Pre-adapt (ep1) и post-adapt (ep2) через один шаг градиента."""
    import torch.nn.functional as F
    from algorithms.maml.procgen_trainer import collect_short_rollout

    # Pre-adapt
    policy = policy_class(num_actions=15)
    policy.load_state_dict(weights)
    env = ProcgenTaskWrapper(env_name=env_name, level_seed=level)
    batch_pre = collect_short_rollout(env, policy, steps=rollout_steps)
    ret_pre = float(np.mean([r for r in batch_pre["returns"]]))

    # Inner step
    obs  = torch.from_numpy(batch_pre["obs"])
    acts = torch.from_numpy(batch_pre["actions"])
    rets = torch.from_numpy(batch_pre["returns"])
    new_lp, value, _ = policy.evaluate(obs, acts)
    advs = rets - value.detach()
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    loss = -(new_lp * advs).mean() + 0.5 * F.mse_loss(value, rets)
    grads = torch.autograd.grad(loss, policy.parameters())
    with torch.no_grad():
        for p, g in zip(policy.parameters(), grads):
            p.sub_(inner_lr * g)

    # Post-adapt
    batch_post = collect_short_rollout(env, policy, steps=rollout_steps)
    ret_post = float(np.mean([r for r in batch_post["returns"]]))
    env.close()
    return [ret_pre, ret_post]


# ──────────────────────────────────────────────
# Загрузка моделей
# ──────────────────────────────────────────────

def load_models(results_dir, methods, num_train_levels):
    rd = Path(results_dir)
    models = {}

    if "ppo" in methods:
        # Берём последний train-level (по умолчанию 42)
        candidates = list(rd.glob("ppo_baseline_level_*"))
        if candidates:
            ckpt = candidates[0] / "final.pt"
            if ckpt.exists():
                p = PPOPolicy(num_actions=15)
                p.load_state_dict(torch.load(ckpt, map_location="cpu"))
                p.eval()
                models["ppo"] = p
                print(f"  [OK] PPO loaded from {ckpt}")
            else:
                print(f"  [SKIP] PPO: {ckpt} не найден")

    if "rl2" in methods:
        ckpt = rd / f"rl2_train_{num_train_levels}_levels" / "final.pt"
        if ckpt.exists():
            p = RL2Policy(num_actions=15)
            p.load_state_dict(torch.load(ckpt, map_location="cpu"))
            p.eval()
            models["rl2"] = p
            print(f"  [OK] RL² loaded from {ckpt}")
        else:
            print(f"  [SKIP] RL²: {ckpt} не найден")

    if "maml" in methods:
        ckpt = rd / f"maml_train_{num_train_levels}_levels" / "final.pt"
        if ckpt.exists():
            models["maml_weights"] = torch.load(ckpt, map_location="cpu")
            print(f"  [OK] MAML weights loaded from {ckpt}")
        else:
            print(f"  [SKIP] MAML: {ckpt} не найден")

    return models


# ──────────────────────────────────────────────
# Эксперимент 1: СКОРОСТЬ АДАПТАЦИИ
# Прогоняем K meta-trials на тестовых уровнях,
# усредняем return по эпизодам
# ──────────────────────────────────────────────

def exp_adaptation_speed(models, game, test_levels, num_episodes=4,
                          num_trials_per_level=5):
    """
    Для каждого meta-RL метода: return[ep1..ep4] усреднённый по levels и trials.
    test_levels — список seed'ов тестовых уровней.
    """
    print("\n[Exp 1] Скорость адаптации...")
    results = {}

    if "rl2" in models:
        all_curves = []
        for level in test_levels:
            for _ in range(num_trials_per_level):
                curve = rollout_rl2_trial(models["rl2"], game, level,
                                           num_episodes=num_episodes)
                all_curves.append(curve)
        arr = np.array(all_curves)  # (N_trials, num_episodes)
        results["rl2"] = {
            "mean": arr.mean(axis=0).tolist(),
            "std":  arr.std(axis=0).tolist(),
            "all":  arr.tolist(),
        }
        print(f"  RL²:  ep1={arr[:,0].mean():.2f} → ep{num_episodes}={arr[:,-1].mean():.2f} "
              f"(gain={arr[:,-1].mean()-arr[:,0].mean():+.2f})")

    if "maml_weights" in models:
        all_curves = []
        for level in test_levels:
            for _ in range(num_trials_per_level):
                curve = rollout_maml_trial(PPOPolicy, models["maml_weights"],
                                            game, level)
                all_curves.append(curve)
        arr = np.array(all_curves)
        results["maml"] = {
            "mean": arr.mean(axis=0).tolist(),
            "std":  arr.std(axis=0).tolist(),
            "all":  arr.tolist(),
        }
        print(f"  MAML: pre={arr[:,0].mean():.2f} → post={arr[:,1].mean():.2f} "
              f"(gain={arr[:,1].mean()-arr[:,0].mean():+.2f})")

    if "ppo" in models:
        # PPO: один эпизод без адаптации, для baseline
        all_rets = []
        for level in test_levels:
            for _ in range(num_trials_per_level):
                ret, _ = rollout_ppo(models["ppo"], game, level)
                all_rets.append(ret)
        results["ppo"] = {
            "mean": float(np.mean(all_rets)),
            "std":  float(np.std(all_rets)),
        }
        print(f"  PPO:  test return={np.mean(all_rets):.2f} ± {np.std(all_rets):.2f}")

    return results


# ──────────────────────────────────────────────
# Эксперимент 2: УСТОЙЧИВОСТЬ
# 20 тестовых уровней, last-episode return
# ──────────────────────────────────────────────

def exp_robustness(models, game, test_levels, num_episodes=4,
                    num_trials_per_level=3):
    """
    Для каждого метода: last-episode return на каждом из test_levels.
    Смотрим среднее и std — это и есть устойчивость.
    """
    print("\n[Exp 2] Устойчивость (20 тестовых уровней)...")
    results = {}

    if "rl2" in models:
        per_level_last = []
        for level in test_levels:
            trials = [rollout_rl2_trial(models["rl2"], game, level,
                                         num_episodes=num_episodes)[-1]
                      for _ in range(num_trials_per_level)]
            per_level_last.append(np.mean(trials))
        results["rl2"] = {
            "per_level": per_level_last,
            "mean": float(np.mean(per_level_last)),
            "std":  float(np.std(per_level_last)),
        }
        print(f"  RL²:  {np.mean(per_level_last):.2f} ± {np.std(per_level_last):.2f}")

    if "maml_weights" in models:
        per_level_post = []
        for level in test_levels:
            trials = [rollout_maml_trial(PPOPolicy, models["maml_weights"],
                                          game, level)[1]
                      for _ in range(num_trials_per_level)]
            per_level_post.append(np.mean(trials))
        results["maml"] = {
            "per_level": per_level_post,
            "mean": float(np.mean(per_level_post)),
            "std":  float(np.std(per_level_post)),
        }
        print(f"  MAML: {np.mean(per_level_post):.2f} ± {np.std(per_level_post):.2f}")

    if "ppo" in models:
        per_level = []
        for level in test_levels:
            trials = [rollout_ppo(models["ppo"], game, level)[0]
                      for _ in range(num_trials_per_level)]
            per_level.append(np.mean(trials))
        results["ppo"] = {
            "per_level": per_level,
            "mean": float(np.mean(per_level)),
            "std":  float(np.std(per_level)),
        }
        print(f"  PPO:  {np.mean(per_level):.2f} ± {np.std(per_level):.2f}")

    return results


# ──────────────────────────────────────────────
# Эксперимент 3: ОБОБЩЕНИЕ
# Три группы уровней: near / mid / far от train
# ──────────────────────────────────────────────

def exp_generalization(models, game, num_train_levels,
                        num_episodes=4, trials_per_group=15):
    """
    near: seeds близко к train (num_train_levels + 0..100)
    mid:  seeds средние (num_train_levels + 1000..1100)
    far:  seeds далёкие (num_train_levels + 5000..5100)

    Для PPO дополнительно смотрим train vs near vs far.
    """
    print("\n[Exp 3] Обобщение (near / mid / far от train)...")

    rng = np.random.RandomState(42)
    groups = {
        "near": [int(num_train_levels + rng.randint(0, 100))
                  for _ in range(trials_per_group)],
        "mid":  [int(num_train_levels + 1000 + rng.randint(0, 100))
                  for _ in range(trials_per_group)],
        "far":  [int(num_train_levels + 5000 + rng.randint(0, 100))
                  for _ in range(trials_per_group)],
    }

    results = {}

    def eval_group(method_fn):
        """method_fn(level) -> scalar return"""
        group_means = {}
        for gname, levels in groups.items():
            rets = [method_fn(l) for l in levels]
            group_means[gname] = {
                "mean": float(np.mean(rets)),
                "std":  float(np.std(rets)),
                "all":  rets,
            }
        return group_means

    if "rl2" in models:
        def rl2_fn(level):
            return rollout_rl2_trial(models["rl2"], game, level,
                                      num_episodes=num_episodes)[-1]
        results["rl2"] = eval_group(rl2_fn)
        for g, v in results["rl2"].items():
            print(f"  RL²  [{g}]: {v['mean']:.2f} ± {v['std']:.2f}")

    if "maml_weights" in models:
        def maml_fn(level):
            return rollout_maml_trial(PPOPolicy, models["maml_weights"],
                                       game, level)[1]
        results["maml"] = eval_group(maml_fn)
        for g, v in results["maml"].items():
            print(f"  MAML [{g}]: {v['mean']:.2f} ± {v['std']:.2f}")

    if "ppo" in models:
        def ppo_fn(level):
            return rollout_ppo(models["ppo"], game, level)[0]
        results["ppo"] = eval_group(ppo_fn)
        for g, v in results["ppo"].items():
            print(f"  PPO  [{g}]: {v['mean']:.2f} ± {v['std']:.2f}")

    return results


# ──────────────────────────────────────────────
# Графики
# ──────────────────────────────────────────────

COLORS = {"ppo": "#7f8c8d", "maml": "#9b59b6", "rl2": "#3498db", "pearl": "#e74c3c"}
LABELS = {"ppo": "PPO (baseline)", "maml": "MAML", "rl2": "RL²", "pearl": "PEARL"}


def plot_adaptation_speed(speed_data, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    for method in ["rl2", "maml"]:
        if method not in speed_data:
            continue
        d = speed_data[method]
        mean = np.array(d["mean"])
        std  = np.array(d["std"])
        x = np.arange(1, len(mean) + 1)
        ax.plot(x, mean, "o-", label=LABELS[method],
                color=COLORS[method], linewidth=2.5, markersize=8)
        ax.fill_between(x, mean - std, mean + std,
                         alpha=0.15, color=COLORS[method])

    if "ppo" in speed_data:
        ax.axhline(speed_data["ppo"]["mean"], color=COLORS["ppo"],
                    linestyle="--", linewidth=2,
                    label=f"PPO (без адаптации, mean={speed_data['ppo']['mean']:.2f})")
        ax.fill_between(
            [1, max(4, len(speed_data.get("rl2", {}).get("mean", [4])))],
            speed_data["ppo"]["mean"] - speed_data["ppo"]["std"],
            speed_data["ppo"]["mean"] + speed_data["ppo"]["std"],
            alpha=0.08, color=COLORS["ppo"]
        )

    ax.set_xlabel("Эпизод адаптации", fontsize=12)
    ax.set_ylabel("Return", fontsize=12)
    ax.set_title("Скорость адаптации к новым задачам\n(среднее ± std по 20 тестовым уровням)",
                  fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xticks(range(1, 5))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  saved {out_path}")


def plot_robustness(rob_data, out_path):
    """Box plot: распределение return по уровням для каждого метода."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Слева: box plot
    ax = axes[0]
    method_order = [m for m in ["ppo", "maml", "rl2"] if m in rob_data]
    data_to_plot = [rob_data[m]["per_level"] for m in method_order]
    labels = [LABELS[m] for m in method_order]
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                     medianprops={"color": "black", "linewidth": 2})
    for patch, method in zip(bp["boxes"], method_order):
        patch.set_facecolor(COLORS[method])
        patch.set_alpha(0.7)
    ax.set_ylabel("Return (post-adaptation)", fontsize=12)
    ax.set_title("Устойчивость\n(распределение по 20 тестовым уровням)", fontsize=13)
    ax.grid(axis="y", alpha=0.3)

    # Справа: mean ± std как bar
    ax = axes[1]
    means = [rob_data[m]["mean"] for m in method_order]
    stds  = [rob_data[m]["std"]  for m in method_order]
    x = np.arange(len(method_order))
    bars = ax.bar(x, means, yerr=stds, capsize=8,
                   color=[COLORS[m] for m in method_order], alpha=0.8,
                   edgecolor="black", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Return (mean ± std)", fontsize=12)
    ax.set_title("Среднее и разброс на 20 уровнях\n(меньше std = лучше)", fontsize=13)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + s + 0.01,
                f"{m:.2f}±{s:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  saved {out_path}")


def plot_generalization(gen_data, out_path):
    """Как падает performance при удалении от train-распределения."""
    fig, ax = plt.subplots(figsize=(9, 5))
    group_names = ["near", "mid", "far"]
    group_labels = ["Близко к train\n(Δseed ≈ 0-100)",
                     "Средне\n(Δseed ≈ 1000)",
                     "Далеко\n(Δseed ≈ 5000)"]
    x = np.arange(len(group_names))
    width = 0.25
    method_order = [m for m in ["ppo", "maml", "rl2"] if m in gen_data]

    for i, method in enumerate(method_order):
        means = [gen_data[method][g]["mean"] for g in group_names]
        stds  = [gen_data[method][g]["std"]  for g in group_names]
        offset = (i - len(method_order)/2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=5,
                label=LABELS[method], color=COLORS[method],
                alpha=0.8, edgecolor="black", linewidth=0.6)

    ax.set_xticks(x); ax.set_xticklabels(group_labels, fontsize=11)
    ax.set_ylabel("Return (post-adaptation)", fontsize=12)
    ax.set_title("Обобщение: performance vs дистанция от train-распределения\n"
                  "(хороший метод падает медленно)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  saved {out_path}")


def plot_summary_table(speed_data, rob_data, gen_data, out_path):
    """Сводная таблица — один слайд показывает всё."""
    method_order = [m for m in ["ppo", "maml", "rl2"] if m in rob_data]

    # Собираем метрики
    rows = []
    for m in method_order:
        adaptation_gain = "—"
        if m in speed_data and "mean" in speed_data[m]:
            mn = speed_data[m]["mean"]
            if isinstance(mn, list) and len(mn) >= 2:
                adaptation_gain = f"{mn[-1] - mn[0]:+.2f}"
        robustness = f"{rob_data[m]['mean']:.2f} ± {rob_data[m]['std']:.2f}" \
                      if m in rob_data else "—"
        gen_drop = "—"
        if m in gen_data:
            near = gen_data[m]["near"]["mean"]
            far  = gen_data[m]["far"]["mean"]
            gen_drop = f"{near - far:+.2f}" if near != 0 else "—"

        train_steps = "—"
        rows.append([LABELS[m], adaptation_gain, robustness, gen_drop, train_steps])

    fig, ax = plt.subplots(figsize=(11, 3))
    ax.axis("off")
    col_labels = ["Метод", "Прирост\nадаптации\n(ep1→ep4)",
                   "Устойчивость\n(mean ± std,\n20 уровней)",
                   "Падение\nобобщения\n(near→far)",
                   "Шагов обучения"]
    table = ax.table(cellText=rows, colLabels=col_labels,
                      loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.5)
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor("#2c3e50")
        table[(0, j)].set_text_props(color="white", fontweight="bold")
    for i, m in enumerate(method_order, 1):
        for j in range(len(col_labels)):
            table[(i, j)].set_facecolor(COLORS[m] + "33")
    plt.title("Сводное сравнение методов Meta-RL", fontsize=14, fontweight="bold",
               pad=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True)
    p.add_argument("--game", default="coinrun")
    p.add_argument("--num-train-levels", type=int, default=20)
    p.add_argument("--methods", nargs="+", default=["ppo", "rl2", "maml"])
    p.add_argument("--num-test-levels", type=int, default=20,
                   help="Сколько тестовых уровней для exp2 и exp3")
    p.add_argument("--trials-per-level", type=int, default=3,
                   help="Попыток на каждый уровень (больше = точнее, но медленнее)")
    p.add_argument("--num-episodes", type=int, default=4,
                   help="Эпизодов в meta-trial для RL²/PEARL")
    args = p.parse_args()

    out_dir = Path(args.results_dir) / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Тестовые уровни для exp1 и exp2: гарантированно OOD
    rng = np.random.RandomState(0)
    test_levels_robustness = [
        int(args.num_train_levels + 1000 + rng.randint(0, 5000))
        for _ in range(args.num_test_levels)
    ]

    print(f"\n{'='*50}")
    print(f" Загружаем модели из {args.results_dir}")
    print(f"{'='*50}")
    models = load_models(args.results_dir, args.methods, args.num_train_levels)

    if not models:
        print("\n[ERR] Ни одна модель не загружена. Проверь --results-dir и --methods.")
        return

    print(f"\n{'='*50}")
    print(f" Запускаем эксперименты")
    print(f" Тестовые уровни: {args.num_test_levels}, trials/level: {args.trials_per_level}")
    print(f" Примерное время: {args.num_test_levels * args.trials_per_level * 3 // 60 + 5} мин")
    print(f"{'='*50}")

    # Exp 1: Скорость адаптации (используем первые 10 уровней)
    speed_data = exp_adaptation_speed(
        models, args.game,
        test_levels=test_levels_robustness[:10],
        num_episodes=args.num_episodes,
        num_trials_per_level=args.trials_per_level,
    )

    # Exp 2: Устойчивость (все 20 уровней)
    rob_data = exp_robustness(
        models, args.game,
        test_levels=test_levels_robustness,
        num_episodes=args.num_episodes,
        num_trials_per_level=args.trials_per_level,
    )

    # Exp 3: Обобщение (near/mid/far)
    gen_data = exp_generalization(
        models, args.game,
        num_train_levels=args.num_train_levels,
        num_episodes=args.num_episodes,
        trials_per_group=max(args.num_test_levels // 2, 5),
    )

    # Сохраняем сырые данные
    metrics = {
        "config": vars(args),
        "adaptation_speed": speed_data,
        "robustness": rob_data,
        "generalization": gen_data,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  saved {out_dir / 'metrics.json'}")

    # Графики
    print("\n[Графики]")
    plot_adaptation_speed(speed_data,       out_dir / "adaptation_speed.png")
    plot_robustness(rob_data,               out_dir / "robustness.png")
    plot_generalization(gen_data,           out_dir / "generalization.png")
    plot_summary_table(speed_data, rob_data, gen_data,
                        out_dir / "summary_table.png")

    print(f"\n{'='*50}")
    print(f" ГОТОВО. Результаты в {out_dir}/")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
