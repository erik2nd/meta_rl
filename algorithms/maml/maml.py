"""
Model-Agnostic Meta-Learning (MAML) for Reinforcement Learning.

Reference: Finn et al., "Model-Agnostic Meta-Learning for Fast Adaptation
of Deep Networks", ICML 2017. https://arxiv.org/abs/1703.03400

Implementation uses learn2learn for cloning policies with differentiable
inner-loop updates. The inner-loop loss is a vanilla policy-gradient
(REINFORCE-style) loss with GAE advantages; the outer-loop loss is the
same form, computed on post-adaptation trajectories.
"""
from typing import Tuple
import torch
import torch.nn as nn
import learn2learn as l2l

from .policy import GaussianMLPPolicy


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation.

    Args:
        rewards: [T] tensor of rewards
        values: [T+1] tensor of value estimates (last is bootstrap)
        dones:   [T] tensor of done flags (1 if episode ended)
        gamma, gae_lambda: GAE hyperparameters

    Returns:
        advantages: [T]
        returns:    [T] (= advantages + values[:-1])
    """
    T = rewards.size(0)
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0
    for t in reversed(range(T)):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * not_done - values[t]
        last_gae = delta + gamma * gae_lambda * not_done * last_gae
        advantages[t] = last_gae
    returns = advantages + values[:-1]
    return advantages, returns


def policy_gradient_loss(
    policy: GaussianMLPPolicy,
    obs: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
) -> torch.Tensor:
    """
    Vanilla policy gradient loss: -E[log pi(a|s) * A].

    Advantages are normalized for stability. We do NOT use a value-function
    baseline in the inner loop (linear baseline / no baseline is the
    standard MAML-RL setup; can be added later).
    """
    log_probs = policy.log_prob(obs, actions)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    loss = -(log_probs * advantages.detach()).mean()
    return loss


class MAML:
    """
    MAML wrapper around a Gaussian policy.

    Key methods:
      - adapt(): one or more inner-loop gradient steps on a cloned policy.
      - meta_update(): outer-loop gradient step on the meta-parameters.
    """

    def __init__(
        self,
        policy: GaussianMLPPolicy,
        inner_lr: float = 0.05,
        outer_lr: float = 3e-4,
        num_inner_steps: int = 1,
        first_order: bool = False,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = "cpu",
    ):
        self.device = device
        self.policy = policy.to(device)
        # learn2learn MAML wrapper: handles differentiable clone() and adapt()
        self.meta_policy = l2l.algorithms.MAML(
            self.policy,
            lr=inner_lr,
            first_order=first_order,
            allow_unused=True,
        )
        self.outer_optimizer = torch.optim.Adam(
            self.meta_policy.parameters(), lr=outer_lr
        )

        self.num_inner_steps = num_inner_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def compute_advantages(self, batch: dict) -> torch.Tensor:
        """
        Compute GAE-style advantages WITHOUT a learned value function.
        We use a simple per-trajectory return baseline (mean discounted return)
        as a substitute. This keeps the inner loop fully differentiable and
        avoids training a separate critic, which is the original MAML-RL setup.

        batch keys: obs [N,T,D], actions [N,T,A], rewards [N,T], dones [N,T]
        """
        rewards = batch["rewards"]  # [N, T]
        N, T = rewards.shape

        # Discounted returns per trajectory
        returns = torch.zeros_like(rewards)
        running = torch.zeros(N, device=rewards.device)
        for t in reversed(range(T)):
            running = rewards[:, t] + self.gamma * running * (1.0 - batch["dones"][:, t])
            returns[:, t] = running

        # Baseline = mean return at each timestep across trajectories
        baseline = returns.mean(dim=0, keepdim=True)  # [1, T]
        advantages = returns - baseline
        return advantages.reshape(-1)  # [N*T]

    def adapt(
        self,
        clone: nn.Module,
        support_batch: dict,
    ) -> nn.Module:
        """
        Inner loop: take `num_inner_steps` gradient steps on `clone`
        using the support batch. Returns the adapted clone.
        """
        for _ in range(self.num_inner_steps):
            obs = support_batch["obs"].reshape(-1, support_batch["obs"].size(-1))
            actions = support_batch["actions"].reshape(-1, support_batch["actions"].size(-1))
            advantages = self.compute_advantages(support_batch)

            loss = policy_gradient_loss(clone, obs, actions, advantages)
            clone.adapt(loss)
        return clone

    def meta_update(self, task_batches: list) -> dict:
        """
        Outer loop: for each task, clone policy, adapt on support set,
        compute loss on query set, sum, backpropagate through inner loop.

        Args:
            task_batches: list of dicts with keys
                "support": {obs, actions, rewards, dones}
                "query":   {obs, actions, rewards, dones}

        Returns:
            dict with diagnostics (meta_loss, mean_support_return, etc.)
        """
        self.outer_optimizer.zero_grad()

        meta_loss = 0.0
        support_returns = []
        query_returns = []

        for task in task_batches:
            # Clone policy for differentiable adaptation
            clone = self.meta_policy.clone()

            # Inner loop adaptation on support set
            clone = self.adapt(clone, task["support"])

            # Compute query loss with adapted clone
            q_obs = task["query"]["obs"].reshape(-1, task["query"]["obs"].size(-1))
            q_actions = task["query"]["actions"].reshape(
                -1, task["query"]["actions"].size(-1)
            )
            q_advantages = self.compute_advantages(task["query"])
            query_loss = policy_gradient_loss(clone, q_obs, q_actions, q_advantages)

            meta_loss = meta_loss + query_loss

            # Diagnostics
            support_returns.append(task["support"]["rewards"].sum(dim=1).mean().item())
            query_returns.append(task["query"]["rewards"].sum(dim=1).mean().item())

        meta_loss = meta_loss / len(task_batches)
        meta_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.meta_policy.parameters(), max_norm=1.0)
        self.outer_optimizer.step()

        return {
            "meta_loss": meta_loss.item(),
            "mean_support_return": sum(support_returns) / len(support_returns),
            "mean_query_return": sum(query_returns) / len(query_returns),
        }

    def get_adapted_policy(self, support_batch: dict) -> nn.Module:
        """For evaluation: produce an adapted clone for a single task."""
        clone = self.meta_policy.clone()
        clone = self.adapt(clone, support_batch)
        return clone
