"""
Entry point for MAML training.

Usage:
    python scripts/train_maml.py --config configs/maml_config.yaml
    python scripts/train_maml.py --config configs/maml_config.yaml --override seed=7
"""
import argparse
import os
import sys
import yaml

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from algorithms.maml.trainer import MAMLTrainer


def parse_overrides(overrides: list) -> dict:
    """Parse key=value overrides into a nested dict patch."""
    patch = {}
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Override must be key=value, got: {ov}")
        key, value = ov.split("=", 1)
        # Try to cast value to int/float/bool
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


def apply_overrides(config: dict, overrides: dict):
    """Apply flat key overrides to nested config (using dot notation)."""
    for key, value in overrides.items():
        parts = key.split(".")
        cur = config
        for p in parts[:-1]:
            cur = cur[p]
        cur[parts[-1]] = value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--override", nargs="*", default=[], help="key=value overrides (dot notation)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    overrides = parse_overrides(args.override)
    apply_overrides(config, overrides)

    print("=" * 60)
    print("MAML Meta-RL Training")
    print("=" * 60)
    print(yaml.dump(config, default_flow_style=False))
    print("=" * 60)

    trainer = MAMLTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
