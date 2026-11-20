"""
PEARL context encoder для Procgen.

Контекст c = {(o_i, a_i, r_i, o'_i)} -> вектор z ~ q_phi(z|c).
Используем product-of-Gaussians по транзишенам (см. оригинальный PEARL paper).

NB: это адаптация твоей существующей algorithms/pearl/encoder.py
под image-наблюдения. Если у тебя уже есть рабочий MLP-encoder для
MetaWorld, замени там self.obs_proj на NatureCNN.
"""
import torch
import torch.nn as nn
from algorithms.shared.cnn import NatureCNN


class ContextEncoder(nn.Module):
    """
    q_phi(z | c): принимает batch транзишенов одной задачи,
    возвращает Gaussian posterior (mean, var) на z.
    """
    def __init__(self, num_actions=15, latent_dim=8, hidden=128, feat_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_actions = num_actions

        # Кодируем obs и next_obs одним CNN
        self.obs_cnn = NatureCNN(out_dim=feat_dim)
        # Per-transition: [feat(o), onehot(a), r, feat(o')] -> (mu_i, var_i)
        trans_dim = feat_dim + num_actions + 1 + feat_dim
        self.trans_mlp = nn.Sequential(
            nn.Linear(trans_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * latent_dim),  # mu and log_var
        )

    def encode_transitions(self, obs, actions, rewards, next_obs):
        """
        obs, next_obs: (N, 64, 64, 3) uint8
        actions: (N,) long
        rewards: (N,) float
        returns: per-transition (mu, var), each (N, latent_dim)
        """
        f1 = self.obs_cnn(obs)
        f2 = self.obs_cnn(next_obs)
        onehot = torch.zeros(actions.shape[0], self.num_actions, device=f1.device)
        onehot.scatter_(1, actions.long().unsqueeze(1), 1.0)
        r = rewards.float().unsqueeze(-1)
        x = torch.cat([f1, onehot, r, f2], dim=-1)
        out = self.trans_mlp(x)
        mu, log_var = out.chunk(2, dim=-1)
        var = torch.exp(log_var).clamp(min=1e-7)
        return mu, var

    def forward(self, obs, actions, rewards, next_obs):
        """Product-of-Gaussians по транзишенам -> posterior на z."""
        mu_i, var_i = self.encode_transitions(obs, actions, rewards, next_obs)
        # Product of Gaussians: precision-weighted average
        prec_i = 1.0 / var_i
        var_post = 1.0 / prec_i.sum(dim=0, keepdim=True)
        mu_post = var_post * (mu_i * prec_i).sum(dim=0, keepdim=True)
        return mu_post.squeeze(0), var_post.squeeze(0)

    def sample(self, mu, var):
        eps = torch.randn_like(mu)
        return mu + torch.sqrt(var) * eps

    def sample_prior(self, batch_size=1, device="cpu"):
        return torch.randn(batch_size, self.latent_dim, device=device)
