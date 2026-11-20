"""
PEARL policy и Q-network, обусловленные на z.
"""
import torch
import torch.nn as nn
from torch.distributions import Categorical
from algorithms.shared.cnn import NatureCNN


class PEARLPolicy(nn.Module):
    """Discrete actor pi(a | s, z). Адаптировано для Procgen."""
    def __init__(self, num_actions=15, latent_dim=8, feat_dim=256, hidden=256):
        super().__init__()
        self.encoder = NatureCNN(out_dim=feat_dim)
        self.head = nn.Sequential(
            nn.Linear(feat_dim + latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, obs, z):
        feat = self.encoder(obs)
        if z.dim() == 1:
            z = z.unsqueeze(0).expand(feat.shape[0], -1)
        elif z.shape[0] == 1 and feat.shape[0] > 1:
            z = z.expand(feat.shape[0], -1)
        x = torch.cat([feat, z], dim=-1)
        return self.head(x)

    def act(self, obs, z, deterministic=False):
        logits = self.forward(obs, z)
        dist = Categorical(logits=logits)
        action = dist.probs.argmax(-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy()


class PEARLQNetwork(nn.Module):
    """Q(s, a, z) для дискретных действий (выход = вектор Q-values на все a)."""
    def __init__(self, num_actions=15, latent_dim=8, feat_dim=256, hidden=256):
        super().__init__()
        self.encoder = NatureCNN(out_dim=feat_dim)
        self.head = nn.Sequential(
            nn.Linear(feat_dim + latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, obs, z):
        feat = self.encoder(obs)
        if z.dim() == 1:
            z = z.unsqueeze(0).expand(feat.shape[0], -1)
        elif z.shape[0] == 1 and feat.shape[0] > 1:
            z = z.expand(feat.shape[0], -1)
        x = torch.cat([feat, z], dim=-1)
        return self.head(x)
