"""
Entry point for PEARL training.

Usage:
    python scripts/train_pearl.py --config configs/pearl_config.yaml
    python scripts/train_pearl.py --config configs/pearl_config.yaml --override experiment.seed=7
"""
import argparse
import os
import sys
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from algorithms.pearl.trainer import PEARLTrainer


def parse_overrides(overrides):
    patch = {}
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Override must be key=value, got: {ov}")
        key, value = ov.split("=", 1)
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                if value.lower() in ("true", "false"):
                    value = value.lower() == "true"
        patch[key] = value
    return patch


def apply_overrides(config, overrides):
    for key, value in overrides.items():
        parts = key.split(".")
        cur = config
        for p in parts[:-1]:
            cur = cur[p]
        cur[parts[-1]] = value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    apply_overrides(config, parse_overrides(args.override))

    print("=" * 60)
    print("PEARL Meta-RL Training")
    print("=" * 60)
    print(yaml.dump(config, default_flow_style=False))
    print("=" * 60)

    trainer = PEARLTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
