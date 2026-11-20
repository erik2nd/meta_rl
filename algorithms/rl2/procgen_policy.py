"""
RL² policy для Procgen.

Архитектура: CNN(obs) || one_hot(prev_action) || prev_reward || done
             -> GRU -> actor & critic heads

Hidden state НЕ сбрасывается между эпизодами в рамках одного meta-trial.
"""
import torch
import torch.nn as nn
from torch.distributions import Categorical
from algorithms.shared.cnn import NatureCNN


class RL2Policy(nn.Module):
    def __init__(self, num_actions=15, feat_dim=256, gru_hidden=256):
        super().__init__()
        self.num_actions = num_actions
        self.encoder = NatureCNN(out_dim=feat_dim)

        # На вход GRU: encoded obs + onehot(prev_action) + prev_reward + done
        gru_input_dim = feat_dim + num_actions + 1 + 1
        self.gru = nn.GRU(gru_input_dim, gru_hidden, batch_first=True)
        self.gru_hidden = gru_hidden

        self.actor = nn.Linear(gru_hidden, num_actions)
        self.critic = nn.Linear(gru_hidden, 1)

    def init_hidden(self, batch_size=1, device="cpu"):
        return torch.zeros(1, batch_size, self.gru_hidden, device=device)

    def _step_input(self, obs, prev_action, prev_reward, prev_done):
        """Один таймстеп: возвращает feature vector для GRU."""
        feat = self.encoder(obs)  # (B, feat_dim)
        onehot = torch.zeros(feat.shape[0], self.num_actions, device=feat.device)
        onehot.scatter_(1, prev_action.long().unsqueeze(1), 1.0)
        prev_r = prev_reward.float().unsqueeze(-1) if prev_reward.dim() == 1 \
                 else prev_reward.float()
        prev_d = prev_done.float().unsqueeze(-1) if prev_done.dim() == 1 \
                 else prev_done.float()
        return torch.cat([feat, onehot, prev_r, prev_d], dim=-1)

    def act_single(self, obs, prev_action, prev_reward, prev_done, hidden):
        """Один шаг — для rollout. obs: (1,64,64,3) или (64,64,3)."""
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        x = self._step_input(obs, prev_action, prev_reward, prev_done)
        x = x.unsqueeze(1)  # (B, T=1, D)
        out, new_hidden = self.gru(x, hidden)
        out = out.squeeze(1)  # (B, gru_hidden)
        logits = self.actor(out)
        value = self.critic(out).squeeze(-1)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value, new_hidden

    def evaluate_sequence(self, obs_seq, prev_act_seq, prev_rew_seq,
                          prev_done_seq, action_seq, initial_hidden):
        """
        Forward по всей последовательности — для loss compute.
        obs_seq: (B, T, 64, 64, 3) и т.д.
        """
        B, T = obs_seq.shape[:2]
        flat_obs = obs_seq.reshape(B * T, *obs_seq.shape[2:])
        flat_feat = self.encoder(flat_obs).reshape(B, T, -1)

        onehot = torch.zeros(B, T, self.num_actions, device=flat_feat.device)
        onehot.scatter_(2, prev_act_seq.long().unsqueeze(-1), 1.0)
        prev_r = prev_rew_seq.float().unsqueeze(-1)
        prev_d = prev_done_seq.float().unsqueeze(-1)
        x = torch.cat([flat_feat, onehot, prev_r, prev_d], dim=-1)

        out, _ = self.gru(x, initial_hidden)
        logits = self.actor(out)
        values = self.critic(out).squeeze(-1)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(action_seq)
        entropy = dist.entropy()
        return log_probs, values, entropy
