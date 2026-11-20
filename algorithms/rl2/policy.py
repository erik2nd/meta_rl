"""
GRU-based recurrent policy for RL^2.

Input at each timestep: concat(state, prev_action, prev_reward, prev_done)
The hidden state encodes the agent's belief about the current task and
is preserved across episodes within one meta-trial.

Reference: Duan et al., "RL^2: Fast Reinforcement Learning via Slow
Reinforcement Learning", 2016. https://arxiv.org/abs/1611.02779
"""
from typing import Tuple
import torch
import torch.nn as nn
from torch.distributions import Normal


class RL2Policy(nn.Module):
    """
    Recurrent actor-critic for RL^2.

    Architecture:
        input -> MLP feature embedding -> GRU -> (mean head, value head)

    The actor outputs a diagonal Gaussian over continuous actions.
    The critic shares the GRU trunk (common for recurrent PPO).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_size: int = 128,
        feature_size: int = 64,
        activation: str = "tanh",
        init_log_std: float = -0.5,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size

        # RL^2 input: state + prev_action + prev_reward + prev_done
        input_dim = obs_dim + action_dim + 1 + 1

        act_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}[activation]

        # Feature embedding before recurrence
        self.embed = nn.Sequential(
            nn.Linear(input_dim, feature_size),
            act_fn(),
        )

        # Recurrence: holds the "task belief"
        self.gru = nn.GRUCell(feature_size, hidden_size)

        # Heads
        self.mean_head = nn.Linear(hidden_size, action_dim)
        self.value_head = nn.Linear(hidden_size, 1)
        self.log_std = nn.Parameter(torch.ones(action_dim) * init_log_std)

    def initial_hidden(self, batch_size: int = 1, device: str = "cpu") -> torch.Tensor:
        """Zero initial hidden state (reset at the start of each trial)."""
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def step(
        self,
        obs: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        prev_done: torch.Tensor,
        hidden: torch.Tensor,
    ) -> Tuple[Normal, torch.Tensor, torch.Tensor]:
        """
        One-step forward (for rollout). All inputs have batch dimension.

        Args:
            obs: [B, obs_dim]
            prev_action: [B, action_dim]
            prev_reward: [B, 1]
            prev_done: [B, 1]
            hidden: [B, hidden_size]

        Returns:
            dist: Normal distribution over actions
            value: [B, 1]
            new_hidden: [B, hidden_size]
        """
        x = torch.cat([obs, prev_action, prev_reward, prev_done], dim=-1)
        features = self.embed(x)
        new_hidden = self.gru(features, hidden)
        mean = self.mean_head(new_hidden)
        std = torch.exp(self.log_std).expand_as(mean)
        value = self.value_head(new_hidden)
        return Normal(mean, std), value, new_hidden

    def forward_sequence(
        self,
        obs_seq: torch.Tensor,
        prev_action_seq: torch.Tensor,
        prev_reward_seq: torch.Tensor,
        prev_done_seq: torch.Tensor,
        initial_hidden: torch.Tensor,
    ) -> Tuple[Normal, torch.Tensor]:
        """
        Forward over a full sequence (for PPO loss computation).

        Args:
            obs_seq:       [B, T, obs_dim]
            prev_action_seq: [B, T, action_dim]
            prev_reward_seq: [B, T, 1]
            prev_done_seq:   [B, T, 1]
            initial_hidden:  [B, hidden_size]

        Returns:
            dist: Normal over actions, batch_shape = [B, T]
            values: [B, T]
        """
        B, T, _ = obs_seq.shape
        x = torch.cat([obs_seq, prev_action_seq, prev_reward_seq, prev_done_seq], dim=-1)
        features = self.embed(x.reshape(B * T, -1)).reshape(B, T, -1)

        # Unroll GRU manually (so we can backprop through it correctly)
        hidden = initial_hidden
        hiddens = []
        for t in range(T):
            hidden = self.gru(features[:, t], hidden)
            hiddens.append(hidden)
        hidden_seq = torch.stack(hiddens, dim=1)  # [B, T, H]

        mean = self.mean_head(hidden_seq)
        std = torch.exp(self.log_std).expand_as(mean)
        values = self.value_head(hidden_seq).squeeze(-1)
        return Normal(mean, std), values
