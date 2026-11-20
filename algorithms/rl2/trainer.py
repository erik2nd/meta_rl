"""
Training loop for RL^2.
"""
import os
import random
import numpy as np
import torch

from envs.env_utils import MetaWorldTaskSampler
from utils.logger import Logger
from .policy import RL2Policy
from .rl2 import RL2


class RL2Trainer:
    def __init__(self, config: dict):
        self.cfg = config
        self._set_seeds(config["experiment"]["seed"])

        self.device = config["experiment"]["device"]
        if self.device == "cuda" and not torch.cuda.is_available():
            print("[Trainer] CUDA unavailable, using CPU.")
            self.device = "cpu"

        # Environment sampler
        env_cfg = config["env"]
        self.sampler = MetaWorldTaskSampler(
            benchmark=env_cfg["benchmark"],
            task_name=env_cfg["task_name"],
        )
        self.max_episode_steps = env_cfg["max_episode_steps"]
        obs_dim, action_dim = self.sampler.get_obs_action_dims()
        self.obs_dim, self.action_dim = obs_dim, action_dim
        print(f"[Trainer] obs_dim={obs_dim}, action_dim={action_dim}")

        # Policy
        pol_cfg = config["policy"]
        policy = RL2Policy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_size=pol_cfg["hidden_size"],
            feature_size=pol_cfg["feature_size"],
            activation=pol_cfg["activation"],
            init_log_std=pol_cfg["init_log_std"],
        )

        # Algorithm
        rl2_cfg = config["rl2"]
        ppo_cfg = config["ppo"]
        self.algo = RL2(
            policy=policy,
            num_episodes_per_trial=rl2_cfg["num_episodes_per_trial"],
            max_episode_steps=self.max_episode_steps,
            gamma=rl2_cfg["gamma"],
            gae_lambda=rl2_cfg["gae_lambda"],
            clip_eps=ppo_cfg["clip_eps"],
            vf_coef=ppo_cfg["vf_coef"],
            ent_coef=ppo_cfg["ent_coef"],
            lr=ppo_cfg["lr"],
            num_epochs=ppo_cfg["num_epochs"],
            minibatch_size=ppo_cfg["minibatch_size"],
            max_grad_norm=ppo_cfg["max_grad_norm"],
            device=self.device,
        )

        self.logger = Logger(
            log_dir=config["experiment"]["log_dir"],
            run_name=config["experiment"]["name"],
            use_wandb=config["logging"]["use_wandb"],
            wandb_project=config["logging"]["wandb_project"],
            config=config,
        )
        self.save_dir = os.path.join(
            config["experiment"]["save_dir"], config["experiment"]["name"]
        )
        os.makedirs(self.save_dir, exist_ok=True)

        # History for plotting
        self.history = {
            "iteration": [],
            "train_first_ep_return": [],
            "train_last_ep_return": [],
            "eval_first_ep_return_mean": [],
            "eval_last_ep_return_mean": [],
            "eval_last_ep_return_std": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
        }

    def _set_seeds(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def train(self):
        cfg = self.cfg["training"]
        log_cfg = self.cfg["logging"]

        for it in range(1, cfg["total_meta_iterations"] + 1):
            # Collect a batch of trials from training tasks
            trials = []
            first_returns, last_returns = [], []
            for _ in range(cfg["trials_per_iteration"]):
                env, _ = self.sampler.sample_train_task()
                trial = self.algo.collect_trial(env)
                trials.append(trial)
                first_returns.append(trial["episode_returns"][0])
                last_returns.append(trial["episode_returns"][-1])

            # PPO update
            diag = self.algo.update(trials)

            # Log to history
            self.history["iteration"].append(it)
            self.history["train_first_ep_return"].append(float(np.mean(first_returns)))
            self.history["train_last_ep_return"].append(float(np.mean(last_returns)))
            self.history["policy_loss"].append(diag["policy_loss"])
            self.history["value_loss"].append(diag["value_loss"])
            self.history["entropy"].append(diag["entropy"])

            if it % log_cfg["log_every"] == 0:
                self.logger.log(
                    {
                        "train/policy_loss": diag["policy_loss"],
                        "train/value_loss": diag["value_loss"],
                        "train/entropy": diag["entropy"],
                        "train/first_ep_return": float(np.mean(first_returns)),
                        "train/last_ep_return": float(np.mean(last_returns)),
                        "train/adaptation_gain": float(
                            np.mean(last_returns) - np.mean(first_returns)
                        ),
                    },
                    step=it,
                )

            if it % cfg["eval_every_n_iterations"] == 0:
                eval_metrics = self.evaluate()
                self.logger.log({f"eval/{k}": v for k, v in eval_metrics.items()}, step=it)
                self.history["eval_first_ep_return_mean"].append(
                    (it, eval_metrics["first_ep_return_mean"])
                )
                self.history["eval_last_ep_return_mean"].append(
                    (it, eval_metrics["last_ep_return_mean"])
                )
                self.history["eval_last_ep_return_std"].append(
                    (it, eval_metrics["last_ep_return_std"])
                )

            if it % log_cfg["save_every"] == 0:
                self.save_checkpoint(it)

        self.save_checkpoint(cfg["total_meta_iterations"], final=True)
        self.save_history()
        self.logger.close()

    @torch.no_grad()
    def evaluate(self) -> dict:
        """Evaluate on held-out test tasks (single trial per task)."""
        cfg = self.cfg["training"]
        first_returns, last_returns, all_episode_returns = [], [], []
        for _ in range(cfg["num_eval_tasks"]):
            env, _ = self.sampler.sample_test_task()
            trial = self.algo.collect_trial(env, deterministic=False)
            first_returns.append(trial["episode_returns"][0])
            last_returns.append(trial["episode_returns"][-1])
            all_episode_returns.append(trial["episode_returns"])

        # Per-episode mean over tasks (the adaptation curve)
        K = self.algo.K
        adapt_curve = []
        for k in range(K):
            vals = [er[k] for er in all_episode_returns]
            adapt_curve.append(float(np.mean(vals)))

        return {
            "first_ep_return_mean": float(np.mean(first_returns)),
            "last_ep_return_mean": float(np.mean(last_returns)),
            "last_ep_return_std": float(np.std(last_returns)),
            "adaptation_gain": float(np.mean(last_returns) - np.mean(first_returns)),
            **{f"adapt_curve_ep{k+1}": v for k, v in enumerate(adapt_curve)},
        }

    def save_checkpoint(self, it, final=False):
        suffix = "final" if final else f"iter_{it}"
        path = os.path.join(self.save_dir, f"rl2_{suffix}.pt")
        torch.save(
            {
                "iteration": it,
                "policy_state": self.algo.policy.state_dict(),
                "optimizer_state": self.algo.optimizer.state_dict(),
                "config": self.cfg,
            },
            path,
        )
        print(f"[Trainer] Saved checkpoint -> {path}")

    def save_history(self):
        import json
        path = os.path.join(self.save_dir, "history.json")
        # Convert any numpy types
        clean = {k: v for k, v in self.history.items()}
        with open(path, "w") as f:
            json.dump(clean, f, indent=2)
        print(f"[Trainer] Saved history -> {path}")
