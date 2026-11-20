"""
Recurrent PPO loss and advantage computation for RL^2.

The outer-loop optimizer is just PPO. The trick: we treat a full meta-trial
(K episodes in one task) as a single sequence and let the recurrent policy
handle adaptation internally.
"""
from typing import Tuple
import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generalized Advantage Estimation over a batch of sequences.

    Args:
        rewards: [B, T]
        values:  [B, T+1]  (last column = bootstrap value)
        dones:   [B, T]
        gamma, gae_lambda: GAE hyperparameters

    Returns:
        advantages: [B, T]
        returns:    [B, T]   (= advantages + values[:, :-1])
    """
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(B, device=rewards.device)
    for t in reversed(range(T)):
        not_done = 1.0 - dones[:, t]
        delta = rewards[:, t] + gamma * values[:, t + 1] * not_done - values[:, t]
        last_gae = delta + gamma * gae_lambda * not_done * last_gae
        advantages[:, t] = last_gae
    returns = advantages + values[:, :-1]
    return advantages, returns


def ppo_loss(
    policy,
    batch: dict,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute PPO loss on a batch of meta-trials.

    batch keys (all tensors):
        obs:           [B, T, obs_dim]
        actions:       [B, T, action_dim]
        prev_actions:  [B, T, action_dim]
        prev_rewards:  [B, T, 1]
        prev_dones:    [B, T, 1]
        old_log_probs: [B, T]
        advantages:    [B, T]
        returns:       [B, T]
        initial_hidden:[B, hidden_size]
    """
    dist, values = policy.forward_sequence(
        batch["obs"],
        batch["prev_actions"],
        batch["prev_rewards"],
        batch["prev_dones"],
        batch["initial_hidden"],
    )
    new_log_probs = dist.log_prob(batch["actions"]).sum(dim=-1)  # [B, T]
    entropy = dist.entropy().sum(dim=-1).mean()

    # Normalize advantages
    advantages = batch["advantages"]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # PPO clipped objective
    ratio = torch.exp(new_log_probs - batch["old_log_probs"])
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # Value loss
    value_loss = 0.5 * (values - batch["returns"]).pow(2).mean()

    total = policy_loss + vf_coef * value_loss - ent_coef * entropy

    info = {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.item(),
        "mean_ratio": ratio.mean().item(),
        "clip_fraction": ((ratio - 1.0).abs() > clip_eps).float().mean().item(),
    }
    return total, info
