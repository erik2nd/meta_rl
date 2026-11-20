"""Запуск PPO baseline на одном уровне Procgen."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from algorithms.ppo_baseline.trainer import train_ppo_single_level


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="coinrun")
    p.add_argument("--level", type=int, default=42,
                   help="Seed уровня, на котором обучаемся")
    p.add_argument("--total-steps", type=int, default=1_000_000)
    p.add_argument("--rollout-steps", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    out = args.output or f"results/procgen_{args.game}/ppo_baseline_level_{args.level}"
    train_ppo_single_level(
        env_name=args.game, level_seed=args.level,
        total_steps=args.total_steps, rollout_steps=args.rollout_steps,
        lr=args.lr, output_dir=out, seed=args.seed,
    )


if __name__ == "__main__":
    main()
