"""Запуск RL² на распределении уровней Procgen."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from algorithms.rl2.procgen_trainer import train_rl2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="coinrun")
    p.add_argument("--num-train-levels", type=int, default=50)
    p.add_argument("--total-steps", type=int, default=5_000_000)
    p.add_argument("--trials-per-iter", type=int, default=4)
    p.add_argument("--episodes-per-trial", type=int, default=2)
    p.add_argument("--max-steps-per-ep", type=int, default=500)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    out = args.output or f"results/procgen_{args.game}/rl2_train_{args.num_train_levels}_levels"
    train_rl2(
        env_name=args.game, num_train_levels=args.num_train_levels,
        total_steps=args.total_steps, trials_per_iter=args.trials_per_iter,
        episodes_per_trial=args.episodes_per_trial,
        max_steps_per_ep=args.max_steps_per_ep,
        lr=args.lr, output_dir=out, seed=args.seed,
    )


if __name__ == "__main__":
    main()
