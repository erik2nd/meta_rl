"""Запуск PEARL на распределении уровней Procgen."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from algorithms.pearl.procgen_trainer import train_pearl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="coinrun")
    p.add_argument("--num-train-levels", type=int, default=50)
    p.add_argument("--total-steps", type=int, default=2_000_000)
    p.add_argument("--latent-dim", type=int, default=8)
    p.add_argument("--warmup-steps", type=int, default=10_000)
    p.add_argument("--context-size", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    out = args.output or f"results/procgen_{args.game}/pearl_train_{args.num_train_levels}_levels"

    # Импортируем твою рабочую sac_update из основного pearl.py если есть
    sac_update_fn = None
    try:
        from algorithms.pearl.pearl import sac_update_discrete as sac_update_fn
        print("[INFO] Используем sac_update_discrete из algorithms/pearl/pearl.py")
    except ImportError:
        print("[WARN] sac_update_discrete не найден в algorithms/pearl/pearl.py.")
        print("[WARN] Обучение будет идти БЕЗ обновлений (только сбор данных).")
        print("[WARN] Нужно адаптировать существующий SAC под discrete actions.")

    train_pearl(
        env_name=args.game, num_train_levels=args.num_train_levels,
        total_steps=args.total_steps, latent_dim=args.latent_dim,
        warmup_steps=args.warmup_steps, context_size=args.context_size,
        lr=args.lr, output_dir=out, seed=args.seed,
        sac_update_fn=sac_update_fn,
    )


if __name__ == "__main__":
    main()
