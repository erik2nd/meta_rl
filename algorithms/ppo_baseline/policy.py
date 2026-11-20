"""Vanilla PPO policy для Procgen (discrete actions)."""
import torch
import torch.nn as nn
from torch.distributions import Categorical
from algorithms.shared.cnn import NatureCNN


class PPOPolicy(nn.Module):
    def __init__(self, num_actions=15, feat_dim=256):
        super().__init__()
        self.encoder = NatureCNN(out_dim=feat_dim)
        self.actor = nn.Linear(feat_dim, num_actions)
        self.critic = nn.Linear(feat_dim, 1)

    def forward(self, obs):
        feat = self.encoder(obs)
        logits = self.actor(feat)
        value = self.critic(feat).squeeze(-1)
        return logits, value

    def act(self, obs, deterministic=False):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.probs.argmax(-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), value, dist.entropy()

    def evaluate(self, obs, actions):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), value, dist.entropy()
