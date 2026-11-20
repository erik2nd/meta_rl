"""
Training loop for PEARL.

The PEARL meta-iteration structure (per Rakelly et al., 2019):
  For each meta-iteration:
    1. For each train task in a sampled batch:
       a. Clear context buffer for that task.
       b. Collect rollouts with z ~ prior, add to BOTH RL and context buffers.
       c. Repeat: sample z ~ posterior(context), collect more rollouts.
    2. For num_train_steps_per_iter:
       a. Sample a batch of train task indices.
       b. For each task: sample context + RL minibatch from its buffer.
       c. Take one PEARL gradient step.
    3. Periodically evaluate on test tasks.
"""
import json
import os
import random

import numpy as np
import torch

from envs.env_utils import MetaWorldTaskSampler
from utils.logger import Logger
from .pearl import PEARL
from .buffer import MultiTaskBuffer
from .rollout import collect_rollouts_with_z_strategy, rollout


class PEARLTrainer:
    def __init__(self, config: dict):
        self.cfg = config
        self._set_seeds(config["experiment"]["seed"])

        self.device = config["experiment"]["device"]
        if self.device == "cuda" and not torch.cuda.is_available():
            print("[Trainer] CUDA unavailable, using CPU.")
            self.device = "cpu"

        env_cfg = config["env"]
        self.sampler = MetaWorldTaskSampler(
            benchmark=env_cfg["benchmark"],
            task_name=env_cfg["task_name"],
        )
        self.max_episode_steps = env_cfg["max_episode_steps"]
        obs_dim, action_dim = self.sampler.get_obs_action_dims()
        self.obs_dim, self.action_dim = obs_dim, action_dim
        print(f"[Trainer] obs_dim={obs_dim}, action_dim={action_dim}")

        # Sample fixed sets of train/test tasks (PEARL keeps buffers per task)
        train_cfg = config["training"]
        self.num_train_tasks = train_cfg["num_train_tasks"]
        self.num_test_tasks = train_cfg["num_test_tasks"]
        # Cache task objects so we can re-instantiate same env later
        self.train_task_envs = [
            self.sampler.sample_train_task()[0] for _ in range(self.num_train_tasks)
        ]
        self.test_task_envs = [
            self.sampler.sample_test_task()[0] for _ in range(self.num_test_tasks)
        ]

        # Algorithm
        pcfg = config["policy"]
        ccfg = config["critic"]
        ecfg = config["encoder"]
        pearl_cfg = config["pearl"]
        opt_cfg = config["optim"]
        self.algo = PEARL(
            obs_dim=obs_dim,
            action_dim=action_dim,
            latent_dim=ecfg["latent_dim"],
            policy_hidden_sizes=pcfg["hidden_sizes"],
            critic_hidden_sizes=ccfg["hidden_sizes"],
            encoder_hidden_sizes=ecfg["hidden_sizes"],
            policy_activation=pcfg["activation"],
            critic_activation=ccfg["activation"],
            encoder_activation=ecfg["activation"],
            gamma=pearl_cfg["gamma"],
            tau=pearl_cfg["tau"],
            alpha=pearl_cfg["alpha"],
            kl_lambda=pearl_cfg["kl_lambda"],
            policy_lr=opt_cfg["policy_lr"],
            qf_lr=opt_cfg["qf_lr"],
            encoder_lr=opt_cfg["encoder_lr"],
            max_grad_norm=opt_cfg["max_grad_norm"],
            device=self.device,
        )
        self.context_batch_size = pearl_cfg["context_batch_size"]
        self.rl_batch_size = pearl_cfg["rl_batch_size"]

        # Buffers (one per train task)
        self.buffer = MultiTaskBuffer(
            num_tasks=self.num_train_tasks,
            obs_dim=obs_dim,
            action_dim=action_dim,
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

        self.history = {
            "iteration": [],
            "train_return": [],
            "eval_pre_return": [],
            "eval_post_return": [],
            "eval_post_return_std": [],
            "qf1_loss": [],
            "policy_loss": [],
            "kl": [],
        }

    def _set_seeds(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ----- Data collection -----

    def _collect_for_task(self, task_idx: int, num_steps_prior: int,
                         num_steps_posterior: int):
        """
        Collect rollouts for one train task: first with z ~ prior, then
        with z ~ posterior using context built from prior rollouts.
        Returns mean return across rollouts (for logging).
        """
        env = self.train_task_envs[task_idx]
        all_returns = []

        # Clear task context to ensure fresh adaptation
        self.buffer.clear_context(task_idx)

        # 1) Prior rollouts (z ~ N(0, I))
        if num_steps_prior > 0:
            prior_paths = collect_rollouts_with_z_strategy(
                env, self.algo, num_steps_prior, self.max_episode_steps,
                z_strategy="prior",
            )
            for p in prior_paths:
                self.buffer.add_path(task_idx, p, into_context=True)
                all_returns.append(p["return"])

        # 2) Posterior rollouts (z ~ q(z | c) where c = recent context)
        if num_steps_posterior > 0:
            ctx = self.buffer.sample_context(
                [task_idx],
                min(self.context_batch_size, self.buffer.ctx_buffers[task_idx].size),
                device=self.device,
            )  # [1, N, ctx_dim]
            ctx = ctx.squeeze(0)
            post_paths = collect_rollouts_with_z_strategy(
                env, self.algo, num_steps_posterior, self.max_episode_steps,
                z_strategy="posterior", context=ctx,
            )
            for p in post_paths:
                self.buffer.add_path(task_idx, p, into_context=True)
                all_returns.append(p["return"])

        return float(np.mean(all_returns)) if all_returns else 0.0

    def _warm_up(self):
        """Initial random-policy rollouts on all train tasks (z ~ prior)."""
        n = self.cfg["training"]["num_initial_steps"]
        print(f"[Trainer] Warm-up: {n} steps per task across "
              f"{self.num_train_tasks} train tasks...")
        for task_idx in range(self.num_train_tasks):
            env = self.train_task_envs[task_idx]
            paths = collect_rollouts_with_z_strategy(
                env, self.algo, n, self.max_episode_steps, z_strategy="prior",
            )
            for p in paths:
                self.buffer.add_path(task_idx, p, into_context=True)
        print("[Trainer] Warm-up complete.")

    # ----- Training -----

    def train(self):
        cfg = self.cfg["training"]
        log_cfg = self.cfg["logging"]

        self._warm_up()

        for it in range(1, cfg["total_meta_iterations"] + 1):
            # 1) Collect new data on a meta-batch of train tasks
            task_indices = random.sample(
                range(self.num_train_tasks),
                min(cfg["meta_batch_size"], self.num_train_tasks),
            )
            collect_returns = []
            for t_idx in task_indices:
                avg_ret = self._collect_for_task(
                    t_idx,
                    cfg["num_steps_prior"],
                    cfg["num_steps_posterior"],
                )
                collect_returns.append(avg_ret)
            mean_train_return = float(np.mean(collect_returns))

            # 2) Take SAC + encoder gradient steps
            losses = {"qf1_loss": [], "policy_loss": [], "kl": []}
            for _ in range(cfg["num_train_steps_per_iter"]):
                # Sample task indices for this gradient step (can resample within)
                batch_idx = random.sample(
                    range(self.num_train_tasks),
                    min(cfg["meta_batch_size"], self.num_train_tasks),
                )
                # Ensure all sampled tasks have enough data
                batch_idx = [
                    i for i in batch_idx
                    if (self.buffer.rl_buffers[i].size >= self.rl_batch_size
                        and self.buffer.ctx_buffers[i].size >= self.context_batch_size)
                ]
                if not batch_idx:
                    break

                context = self.buffer.sample_context(
                    batch_idx, self.context_batch_size, device=self.device
                )
                rl_batch = self.buffer.sample_rl_batch(
                    batch_idx, self.rl_batch_size, device=self.device
                )
                diag = self.algo.update(context, rl_batch)
                for k in losses:
                    losses[k].append(diag[k])

            mean_losses = {k: float(np.mean(v)) if v else 0.0 for k, v in losses.items()}

            self.history["iteration"].append(it)
            self.history["train_return"].append(mean_train_return)
            self.history["qf1_loss"].append(mean_losses["qf1_loss"])
            self.history["policy_loss"].append(mean_losses["policy_loss"])
            self.history["kl"].append(mean_losses["kl"])

            if it % log_cfg["log_every"] == 0:
                self.logger.log({
                    "train/mean_return": mean_train_return,
                    "train/qf1_loss": mean_losses["qf1_loss"],
                    "train/policy_loss": mean_losses["policy_loss"],
                    "train/kl": mean_losses["kl"],
                }, step=it)

            if it % cfg["eval_every_n_iterations"] == 0:
                eval_metrics = self.evaluate()
                self.logger.log(
                    {f"eval/{k}": v for k, v in eval_metrics.items()}, step=it
                )
                self.history["eval_pre_return"].append(
                    (it, eval_metrics["pre_adapt_return_mean"])
                )
                self.history["eval_post_return"].append(
                    (it, eval_metrics["post_adapt_return_mean"])
                )
                self.history["eval_post_return_std"].append(
                    (it, eval_metrics["post_adapt_return_std"])
                )

            if it % log_cfg["save_every"] == 0:
                self.save_checkpoint(it)

        self.save_checkpoint(cfg["total_meta_iterations"], final=True)
        self.save_history()
        self.logger.close()

    # ----- Evaluation -----

    @torch.no_grad()
    def evaluate(self) -> dict:
        """
        Evaluate on held-out test tasks.

        For each test task:
          - Pre-adaptation: roll out with z ~ prior, record return.
          - Adapt: use those rollouts as context, sample z ~ posterior.
          - Post-adaptation: roll out with the adapted z, record return.
        """
        cfg = self.cfg["training"]
        pre_returns, post_returns = [], []

        for env in self.test_task_envs:
            # Pre-adaptation: z ~ prior
            z_prior = self.algo.sample_z_prior(num_tasks=1).squeeze(0)
            pre_rets = []
            ctx_buf_obs, ctx_buf_act, ctx_buf_r, ctx_buf_next = [], [], [], []
            for _ in range(cfg["num_eval_trajectories"]):
                path = rollout(env, self.algo, z_prior, self.max_episode_steps)
                pre_rets.append(path["return"])
                ctx_buf_obs.append(path["obs"])
                ctx_buf_act.append(path["actions"])
                ctx_buf_r.append(path["rewards"])
                ctx_buf_next.append(path["next_obs"])
            pre_returns.append(float(np.mean(pre_rets)))

            # Build context from pre-adapt rollouts
            ctx_obs = torch.as_tensor(np.concatenate(ctx_buf_obs), device=self.device)
            ctx_act = torch.as_tensor(np.concatenate(ctx_buf_act), device=self.device)
            ctx_r = torch.as_tensor(np.concatenate(ctx_buf_r), device=self.device).unsqueeze(-1)
            ctx_next = torch.as_tensor(np.concatenate(ctx_buf_next), device=self.device)
            ctx = torch.cat([ctx_obs, ctx_act, ctx_r, ctx_next], dim=-1)  # [N, ctx_dim]

            # Adapt: sample z ~ posterior
            mu, sigma_sq = self.algo.infer_posterior(ctx.unsqueeze(0))
            z_post = self.algo.sample_z(mu, sigma_sq).squeeze(0)

            # Post-adaptation
            post_rets = []
            for _ in range(cfg["num_eval_trajectories"]):
                path = rollout(env, self.algo, z_post, self.max_episode_steps)
                post_rets.append(path["return"])
            post_returns.append(float(np.mean(post_rets)))

        return {
            "pre_adapt_return_mean": float(np.mean(pre_returns)),
            "post_adapt_return_mean": float(np.mean(post_returns)),
            "post_adapt_return_std": float(np.std(post_returns)),
            "adaptation_gain": float(np.mean(post_returns) - np.mean(pre_returns)),
        }

    # ----- Checkpointing -----

    def save_checkpoint(self, it, final=False):
        suffix = "final" if final else f"iter_{it}"
        path = os.path.join(self.save_dir, f"pearl_{suffix}.pt")
        torch.save({
            "iteration": it,
            "policy_state": self.algo.policy.state_dict(),
            "qf1_state": self.algo.qf1.state_dict(),
            "qf2_state": self.algo.qf2.state_dict(),
            "encoder_state": self.algo.encoder.state_dict(),
            "config": self.cfg,
        }, path)
        print(f"[Trainer] Saved checkpoint -> {path}")

    def save_history(self):
        path = os.path.join(self.save_dir, "history.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"[Trainer] Saved history -> {path}")
