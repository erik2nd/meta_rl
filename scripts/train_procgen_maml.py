"""Запуск FOMAML на распределении уровней Procgen."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from algorithms.maml.procgen_trainer import train_maml_fomaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="coinrun")
    p.add_argument("--num-train-levels", type=int, default=50)
    p.add_argument("--total-steps", type=int, default=2_000_000)
    p.add_argument("--tasks-per-meta-iter", type=int, default=4)
    p.add_argument("--inner-steps", type=int, default=1)
    p.add_argument("--inner-lr", type=float, default=0.1)
    p.add_argument("--outer-lr", type=float, default=3e-4)
    p.add_argument("--rollout-steps", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    out = args.output or f"results/procgen_{args.game}/maml_train_{args.num_train_levels}_levels"
    train_maml_fomaml(
        env_name=args.game, num_train_levels=args.num_train_levels,
        total_steps=args.total_steps,
        tasks_per_meta_iter=args.tasks_per_meta_iter,
        inner_steps=args.inner_steps, inner_lr=args.inner_lr,
        outer_lr=args.outer_lr, rollout_steps=args.rollout_steps,
        output_dir=out, seed=args.seed,
    )


if __name__ == "__main__":
    main()
