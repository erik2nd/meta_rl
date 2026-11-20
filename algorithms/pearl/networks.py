"""
Networks for PEARL: Gaussian policy and twin Q-functions, both conditioned
on the task embedding z (concatenated with state/state-action input).
"""
import math
import torch
import torch.nn as nn
from torch.distributions import Normal


def _mlp(in_dim: int, out_dim: int, hidden_sizes: list, activation: str):
    act_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}[activation]
    layers = []
    prev = in_dim
    for h in hidden_sizes:
        layers += [nn.Linear(prev, h), act_fn()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class TanhGaussianPolicy(nn.Module):
    """
    SAC-style squashed Gaussian policy, conditioned on (state, z).

    Outputs a tanh-squashed Gaussian over actions, plus the corrected
    log-probability (accounting for the change of variables).
    """

    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int,
        hidden_sizes: list,
        activation: str = "relu",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        act_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}[activation]
        layers = []
        prev = obs_dim + latent_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), act_fn()]
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev, action_dim)
        self.log_std_head = nn.Linear(prev, action_dim)

    def forward(self, obs: torch.Tensor, z: torch.Tensor):
        """
        Args:
            obs: [B, obs_dim]
            z:   [B, latent_dim]

        Returns:
            dist: pre-squash Normal
            mean, log_std
        """
        x = torch.cat([obs, z], dim=-1)
        h = self.trunk(x)
        mean = self.mean_head(h)
        log_std = torch.clamp(self.log_std_head(h), self.LOG_STD_MIN, self.LOG_STD_MAX)
        return Normal(mean, log_std.exp()), mean, log_std

    def sample(self, obs: torch.Tensor, z: torch.Tensor, deterministic: bool = False):
        """
        Sample tanh-squashed action with reparameterization.

        Returns:
            action: [B, action_dim]   (in [-1, 1])
            log_prob: [B]             (corrected log-probability)
        """
        dist, mean, log_std = self.forward(obs, z)
        if deterministic:
            u = mean
        else:
            u = dist.rsample()

        action = torch.tanh(u)
        # log-prob with tanh correction:
        # log pi(a|s) = log N(u|mu, sigma) - sum log(1 - tanh(u)^2)
        log_prob = dist.log_prob(u).sum(dim=-1)
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
        return action, log_prob


class QNetwork(nn.Module):
    """Q(s, a, z) network."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int,
        hidden_sizes: list,
        activation: str = "relu",
    ):
        super().__init__()
        self.net = _mlp(
            in_dim=obs_dim + action_dim + latent_dim,
            out_dim=1,
            hidden_sizes=hidden_sizes,
            activation=activation,
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action, z], dim=-1)
        return self.net(x).squeeze(-1)
