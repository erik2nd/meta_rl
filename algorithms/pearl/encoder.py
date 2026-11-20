"""
Probabilistic context encoder q_phi(z | c) for PEARL.

The encoder takes a set of transitions c = {(s_i, a_i, r_i, s'_i)} from
a task and produces a Gaussian posterior over the latent task embedding z.

Permutation invariance is achieved via "product of Gaussian factors":
each transition produces its own Gaussian factor N(mu_i, sigma_i^2);
the joint posterior is the product of these factors (renormalized).

Reference: Rakelly et al., 2019. arXiv:1903.08254 (Section 4.2).
"""
from typing import Tuple
import torch
import torch.nn as nn


class ContextEncoder(nn.Module):
    """
    Per-transition MLP that outputs (mu, log_sigma_sq) for each transition.
    Combine into a single Gaussian via product-of-experts (multiplying
    factors), yielding a permutation-invariant posterior.

    Input per transition: (s, a, r, s')  -> dim = 2*obs_dim + action_dim + 1
    Output: latent_dim mean and log_var
    """

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

        input_dim = 2 * obs_dim + action_dim + 1  # (s, a, r, s')
        act_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}[activation]

        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), act_fn()]
            prev = h
        # Output: 2 * latent_dim (mu and log_sigma_sq concatenated)
        layers.append(nn.Linear(prev, 2 * latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            context: [B, N, 2*obs_dim + action_dim + 1]
                B = batch of tasks, N = transitions per task

        Returns:
            mu: [B, latent_dim]
            sigma_sq: [B, latent_dim]  (variance of posterior, post-aggregation)
        """
        B, N, _ = context.shape
        flat = context.reshape(B * N, -1)
        out = self.net(flat).reshape(B, N, 2 * self.latent_dim)

        mus = out[..., :self.latent_dim]            # [B, N, latent_dim]
        log_sigma_sq = out[..., self.latent_dim:]   # [B, N, latent_dim]
        # Numerically stable: clamp log variance
        log_sigma_sq = torch.clamp(log_sigma_sq, -10.0, 2.0)
        sigma_sq = torch.exp(log_sigma_sq)          # per-transition variances

        # Product-of-experts aggregation:
        # If each factor is N(mu_i, sigma_i^2), the product is
        # N(mu, sigma^2) where
        #   1/sigma^2 = sum_i 1/sigma_i^2
        #   mu / sigma^2 = sum_i mu_i / sigma_i^2
        # We multiply also by the prior N(0, 1) (one extra factor).
        precisions = 1.0 / sigma_sq                 # [B, N, latent_dim]
        precision_sum = precisions.sum(dim=1) + 1.0  # +1 for unit prior
        sigma_sq_post = 1.0 / precision_sum          # [B, latent_dim]
        mu_times_prec = (mus * precisions).sum(dim=1)  # [B, latent_dim]
        mu_post = mu_times_prec * sigma_sq_post      # [B, latent_dim]

        return mu_post, sigma_sq_post

    @staticmethod
    def kl_to_unit_prior(mu: torch.Tensor, sigma_sq: torch.Tensor) -> torch.Tensor:
        """
        KL divergence D_KL( N(mu, sigma_sq) || N(0, I) ) per batch element.
        Returns a [B] tensor.
        """
        # KL = 0.5 * sum( sigma_sq + mu^2 - 1 - log(sigma_sq) ) over latent dims
        return 0.5 * (sigma_sq + mu.pow(2) - 1.0 - torch.log(sigma_sq + 1e-8)).sum(dim=-1)

    @staticmethod
    def sample_z(mu: torch.Tensor, sigma_sq: torch.Tensor) -> torch.Tensor:
        """Reparameterized sample z ~ N(mu, sigma_sq)."""
        eps = torch.randn_like(mu)
        return mu + eps * torch.sqrt(sigma_sq + 1e-8)
