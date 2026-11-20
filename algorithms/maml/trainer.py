"""
MAML training loop for Meta-RL.

Performs meta-training on a distribution of tasks, periodically evaluates
on held-out test tasks, and saves checkpoints.
"""
import os
import random
import numpy as np
import torch

from envs.env_utils import MetaWorldTaskSampler
from utils.replay import collect_task_batch
from utils.logger import Logger
from utils.metrics import aggregate_returns
from .policy import GaussianMLPPolicy
from .maml import MAML


class MAMLTrainer:
    def __init__(self, config: dict):
        self.cfg = config
        self._set_seeds(config["experiment"]["seed"])

        self.device = config["experiment"]["device"]
        if self.device == "cuda" and not torch.cuda.is_available():
            print("[Trainer] CUDA requested but not available; falling back to CPU.")
            self.device = "cpu"

        # Build environment sampler
        env_cfg = config["env"]
        self.sampler = MetaWorldTaskSampler(
            benchmark=env_cfg["benchmark"],
            task_name=env_cfg["task_name"],
        )
        obs_dim, action_dim = self.sampler.get_obs_action_dims()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_episode_steps = env_cfg["max_episode_steps"]
        print(f"[Trainer] obs_dim={obs_dim}, action_dim={action_dim}")

        # Build policy
        pol_cfg = config["policy"]
        policy = GaussianMLPPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=pol_cfg["hidden_sizes"],
            activation=pol_cfg["activation"],
            init_log_std=pol_cfg["init_log_std"],
        )

        # Build MAML algorithm
        maml_cfg = config["maml"]
        self.algo = MAML(
            policy=policy,
            inner_lr=maml_cfg["inner_lr"],
            outer_lr=maml_cfg["outer_lr"],
            num_inner_steps=maml_cfg["num_inner_steps"],
            first_order=maml_cfg["first_order"],
            gamma=maml_cfg["gamma"],
            gae_lambda=maml_cfg["gae_lambda"],
            device=self.device,
        )

        # Logger
        self.logger = Logger(
            log_dir=config["experiment"]["log_dir"],
            run_name=config["experiment"]["name"],
            use_wandb=config["logging"]["use_wandb"],
            wandb_project=config["logging"]["wandb_project"],
            config=config,
        )

        # Save dir
        self.save_dir = os.path.join(
            config["experiment"]["save_dir"], config["experiment"]["name"]
        )
        os.makedirs(self.save_dir, exist_ok=True)

    def _set_seeds(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _sample_task_batch(self, split: str = "train") -> dict:
        """
        Sample one task, collect support + query trajectories.
        Support is collected with the meta-policy; query is collected
        with the adapted clone.
        """
        if split == "train":
            env, info = self.sampler.sample_train_task()
        else:
            env, info = self.sampler.sample_test_task()

        cfg = self.cfg["training"]

        # Support set: roll out current meta-policy
        support = collect_task_batch(
            env_fn=lambda: env,  # reuse same env (set_task already applied)
            policy=self.algo.policy,
            num_trajectories=cfg["trajectories_per_task"],
            max_steps=self.max_episode_steps,
            device=self.device,
        )

        # Adapt
        with torch.enable_grad():
            adapted = self.algo.meta_policy.clone()
            adapted = self.algo.adapt(adapted, support)

        # Query set: roll out adapted clone
        query = collect_task_batch(
            env_fn=lambda: env,
            policy=adapted,
            num_trajectories=cfg["eval_trajectories_per_task"],
            max_steps=self.max_episode_steps,
            device=self.device,
        )

        return {"support": support, "query": query, "info": info}

    def train(self):
        cfg = self.cfg["training"]
        log_cfg = self.cfg["logging"]

        for it in range(1, cfg["total_meta_iterations"] + 1):
            # Build a batch of task_batches (each task contributes support+query)
            task_batches = []
            for _ in range(cfg["tasks_per_meta_batch"]):
                tb = self._sample_task_batch(split="train")
                task_batches.append(tb)

            # Meta-update
            diag = self.algo.meta_update(task_batches)

            if it % log_cfg["log_every"] == 0:
                self.logger.log(
                    {
                        "train/meta_loss": diag["meta_loss"],
                        "train/mean_support_return": diag["mean_support_return"],
                        "train/mean_query_return": diag["mean_query_return"],
                    },
                    step=it,
                )

            # Periodic evaluation
            if it % cfg["eval_every_n_iterations"] == 0:
                eval_metrics = self.evaluate()
                self.logger.log(
                    {f"eval/{k}": v for k, v in eval_metrics.items()}, step=it
                )

            # Checkpoint
            if it % log_cfg["save_every"] == 0:
                self.save_checkpoint(it)

        self.save_checkpoint(cfg["total_meta_iterations"], final=True)
        self.logger.close()

    @torch.no_grad()
    def _eval_one_task(self, env):
        """Collect post-adaptation returns on one test task."""
        cfg = self.cfg["training"]

        # Pre-adaptation rollout
        pre_support = collect_task_batch(
            env_fn=lambda: env,
            policy=self.algo.policy,
            num_trajectories=cfg["trajectories_per_task"],
            max_steps=self.max_episode_steps,
            device=self.device,
        )
        pre_return = pre_support["rewards"].sum(dim=1).mean().item()

        # Adapt
        with torch.enable_grad():
            adapted = self.algo.meta_policy.clone()
            adapted = self.algo.adapt(adapted, pre_support)

        # Post-adaptation rollout
        post = collect_task_batch(
            env_fn=lambda: env,
            policy=adapted,
            num_trajectories=cfg["eval_trajectories_per_task"],
            max_steps=self.max_episode_steps,
            device=self.device,
        )
        post_return = post["rewards"].sum(dim=1).mean().item()
        return pre_return, post_return

    def evaluate(self) -> dict:
        """Evaluate on held-out test tasks."""
        cfg = self.cfg["training"]
        pre_returns, post_returns = [], []
        for _ in range(cfg["num_eval_tasks"]):
            env, _ = self.sampler.sample_test_task()
            pre, post = self._eval_one_task(env)
            pre_returns.append(pre)
            post_returns.append(post)

        return {
            "pre_adapt_return_mean": float(np.mean(pre_returns)),
            "post_adapt_return_mean": float(np.mean(post_returns)),
            "post_adapt_return_std": float(np.std(post_returns)),
            "adaptation_gain": float(np.mean(post_returns) - np.mean(pre_returns)),
        }

    def save_checkpoint(self, iteration: int, final: bool = False):
        suffix = "final" if final else f"iter_{iteration}"
        path = os.path.join(self.save_dir, f"maml_{suffix}.pt")
        torch.save(
            {
                "iteration": iteration,
                "policy_state": self.algo.policy.state_dict(),
                "optimizer_state": self.algo.outer_optimizer.state_dict(),
                "config": self.cfg,
            },
            path,
        )
        print(f"[Trainer] Saved checkpoint -> {path}")
