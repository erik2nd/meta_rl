"""
Gaussian MLP policy for continuous action spaces.
Returns a Normal distribution over actions.
"""
import torch
import torch.nn as nn
from torch.distributions import Normal


class GaussianMLPPolicy(nn.Module):
    """
    Multi-layer perceptron Gaussian policy.

    Output: mean and (state-independent) log_std of a diagonal Gaussian.
    Used in MAML and other meta-RL algorithms for continuous control.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: list,
        activation: str = "tanh",
        init_log_std: float = -0.5,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Build MLP trunk
        act_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}[activation]
        layers = []
        prev_size = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(act_fn())
            prev_size = h
        self.trunk = nn.Sequential(*layers)

        # Mean head and learnable log_std (state-independent — standard for MAML)
        self.mean_head = nn.Linear(prev_size, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * init_log_std)

    def forward(self, obs: torch.Tensor) -> Normal:
        """Return a Normal distribution over actions."""
        features = self.trunk(obs)
        mean = self.mean_head(features)
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """Sample (or take mean) action; return action + log_prob."""
        dist = self.forward(obs)
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Log-probability of given actions under the policy."""
        dist = self.forward(obs)
        return dist.log_prob(actions).sum(dim=-1)
