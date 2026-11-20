"""
PEARL: Probabilistic Embeddings for Actor-critic RL.

Combines SAC (off-policy actor-critic) with a probabilistic context
encoder. The encoder's parameters are trained jointly with the critic
(through the Q-loss gradient), regularized by a KL term to a unit prior.

Reference: Rakelly et al., 2019. arXiv:1903.08254
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ContextEncoder
from .networks import TanhGaussianPolicy, QNetwork


class PEARL:
    """
    Holds policy, twin Q-networks, target Q-networks, and context encoder.

    Key methods:
      - infer_posterior(context): returns (mu, sigma_sq) for tasks.
      - sample_z(mu, sigma_sq): reparameterized sample.
      - update(task_indices, buffer): one PEARL gradient step.
      - get_action(obs, z): for environment rollouts.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int,
        policy_hidden_sizes: list,
        critic_hidden_sizes: list,
        encoder_hidden_sizes: list,
        policy_activation: str = "relu",
        critic_activation: str = "relu",
        encoder_activation: str = "relu",
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        kl_lambda: float = 0.1,
        policy_lr: float = 3e-4,
        qf_lr: float = 3e-4,
        encoder_lr: float = 3e-4,
        max_grad_norm: float = 1.0,
        device: str = "cpu",
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.kl_lambda = kl_lambda
        self.latent_dim = latent_dim
        self.max_grad_norm = max_grad_norm

        # Encoder
        self.encoder = ContextEncoder(
            obs_dim=obs_dim,
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_sizes=encoder_hidden_sizes,
            activation=encoder_activation,
        ).to(device)

        # Policy
        self.policy = TanhGaussianPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_sizes=policy_hidden_sizes,
            activation=policy_activation,
        ).to(device)

        # Twin Q-networks and their targets
        self.qf1 = QNetwork(obs_dim, action_dim, latent_dim,
                            critic_hidden_sizes, critic_activation).to(device)
        self.qf2 = QNetwork(obs_dim, action_dim, latent_dim,
                            critic_hidden_sizes, critic_activation).to(device)
        self.target_qf1 = copy.deepcopy(self.qf1)
        self.target_qf2 = copy.deepcopy(self.qf2)
        for p in self.target_qf1.parameters():
            p.requires_grad = False
        for p in self.target_qf2.parameters():
            p.requires_grad = False

        # Optimizers
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.qf_optim = torch.optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=qf_lr
        )
        self.encoder_optim = torch.optim.Adam(self.encoder.parameters(), lr=encoder_lr)

    # ----- Inference helpers -----

    def infer_posterior(self, context: torch.Tensor):
        """context: [T, N, ctx_dim] -> (mu, sigma_sq) each [T, latent_dim]."""
        return self.encoder(context)

    def sample_z(self, mu: torch.Tensor, sigma_sq: torch.Tensor) -> torch.Tensor:
        return ContextEncoder.sample_z(mu, sigma_sq)

    @torch.no_grad()
    def get_action(self, obs, z, deterministic: bool = False):
        """For environment rollouts. obs: np array; z: [latent_dim] tensor."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        z_t = z.unsqueeze(0) if z.dim() == 1 else z
        action, _ = self.policy.sample(obs_t, z_t, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def sample_z_prior(self, num_tasks: int = 1) -> torch.Tensor:
        """Sample z from the unit Gaussian prior (used for exploration)."""
        return torch.randn(num_tasks, self.latent_dim, device=self.device)

    # ----- Soft target updates -----

    def _soft_update(self):
        with torch.no_grad():
            for p, tp in zip(self.qf1.parameters(), self.target_qf1.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
            for p, tp in zip(self.qf2.parameters(), self.target_qf2.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

    # ----- Main update -----

    def update(self, context: torch.Tensor, rl_batch: dict) -> dict:
        """
        One PEARL gradient step.

        Args:
            context: [T, N, ctx_dim]      transitions for posterior inference
            rl_batch: dict with keys obs, actions, rewards, next_obs, dones
                      each of shape [T, B, ...]

        Returns: dict with diagnostics.
        """
        T, B = rl_batch["obs"].shape[:2]
        device = self.device

        # 1) Infer posterior q(z | c) and sample z
        mu, sigma_sq = self.infer_posterior(context)         # [T, latent_dim]
        z = self.sample_z(mu, sigma_sq)                       # [T, latent_dim]
        # Broadcast z to per-transition
        z_expanded = z.unsqueeze(1).expand(T, B, self.latent_dim).reshape(T * B, -1)

        # Flatten rl_batch to [T*B, ...]
        obs = rl_batch["obs"].reshape(T * B, -1)
        actions = rl_batch["actions"].reshape(T * B, -1)
        rewards = rl_batch["rewards"].reshape(T * B)
        next_obs = rl_batch["next_obs"].reshape(T * B, -1)
        dones = rl_batch["dones"].reshape(T * B)

        # 2) Q-value targets: use target Q-networks and current policy on next_obs
        with torch.no_grad():
            next_action, next_log_prob = self.policy.sample(next_obs, z_expanded)
            q1_next = self.target_qf1(next_obs, next_action, z_expanded)
            q2_next = self.target_qf2(next_obs, next_action, z_expanded)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            q_target = rewards + self.gamma * (1 - dones) * min_q_next

        # 3) Critic loss (NOTE: encoder's gradients also flow from here)
        # We detach z for the actor loss but NOT for the critic loss
        # (encoder is trained through the Bellman residual).
        q1_pred = self.qf1(obs, actions, z_expanded)
        q2_pred = self.qf2(obs, actions, z_expanded)
        qf1_loss = F.mse_loss(q1_pred, q_target)
        qf2_loss = F.mse_loss(q2_pred, q_target)
        qf_loss = qf1_loss + qf2_loss

        # 4) KL regularization on posterior
        kl = ContextEncoder.kl_to_unit_prior(mu, sigma_sq).mean()
        kl_loss = self.kl_lambda * kl

        # 5) Encoder gradients = qf gradients + kl gradients
        self.encoder_optim.zero_grad()
        self.qf_optim.zero_grad()
        (qf_loss + kl_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(
            list(self.qf1.parameters()) + list(self.qf2.parameters()),
            self.max_grad_norm,
        )
        self.encoder_optim.step()
        self.qf_optim.step()

        # 6) Actor loss (detach z so encoder isn't pulled toward policy)
        z_detached = z.detach().unsqueeze(1).expand(T, B, self.latent_dim).reshape(T * B, -1)
        new_action, new_log_prob = self.policy.sample(obs, z_detached)
        q1_new = self.qf1(obs, new_action, z_detached)
        q2_new = self.qf2(obs, new_action, z_detached)
        min_q_new = torch.min(q1_new, q2_new)
        policy_loss = (self.alpha * new_log_prob - min_q_new).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy_optim.step()

        # 7) Soft target update
        self._soft_update()

        return {
            "qf1_loss": qf1_loss.item(),
            "qf2_loss": qf2_loss.item(),
            "policy_loss": policy_loss.item(),
            "kl": kl.item(),
            "z_mu_norm": mu.norm(dim=-1).mean().item(),
            "z_sigma_mean": sigma_sq.sqrt().mean().item(),
            "mean_q": q1_pred.mean().item(),
            "log_prob_mean": new_log_prob.mean().item(),
        }
