"""
Unified logger: supports console + TensorBoard + (optional) wandb.
"""
import os
import time
from typing import Any, Dict


class Logger:
    """
    Minimal logger that writes to:
      - stdout (always)
      - TensorBoard (always, if available)
      - wandb (optional)
    """

    def __init__(
        self,
        log_dir: str,
        run_name: str,
        use_wandb: bool = False,
        wandb_project: str = "meta-rl-thesis",
        config: Dict[str, Any] = None,
    ):
        os.makedirs(log_dir, exist_ok=True)
        self.run_dir = os.path.join(log_dir, run_name)
        os.makedirs(self.run_dir, exist_ok=True)

        # TensorBoard
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(self.run_dir)
        except ImportError:
            print("[Logger] tensorboard not available, skipping.")
            self.tb_writer = None

        # wandb
        self.use_wandb = use_wandb
        if use_wandb:
            try:
                import wandb
                wandb.init(project=wandb_project, name=run_name, config=config or {})
                self.wandb = wandb
            except ImportError:
                print("[Logger] wandb not installed, skipping.")
                self.use_wandb = False
                self.wandb = None
        else:
            self.wandb = None

        self.start_time = time.time()

    def log(self, metrics: Dict[str, float], step: int):
        """Log a dict of scalars at a given step."""
        elapsed = time.time() - self.start_time
        # Console
        msg = f"[step {step:>5d} | {elapsed:7.1f}s] " + " | ".join(
            f"{k}={v:.4f}" for k, v in metrics.items()
        )
        print(msg)

        # TensorBoard
        if self.tb_writer is not None:
            for k, v in metrics.items():
                self.tb_writer.add_scalar(k, v, step)

        # wandb
        if self.use_wandb and self.wandb is not None:
            self.wandb.log(metrics, step=step)

    def close(self):
        if self.tb_writer is not None:
            self.tb_writer.close()
        if self.use_wandb and self.wandb is not None:
            self.wandb.finish()
