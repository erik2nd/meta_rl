"""
RL^2 algorithm: meta-trial rollouts + recurrent PPO update.

A "meta-trial" is K consecutive episodes in the same task. The recurrent
policy's hidden state is preserved across episodes within a trial,
allowing the agent to "adapt" implicitly through its memory.
"""
import numpy as np
import torch

from .policy import RL2Policy
from .ppo import compute_gae, ppo_loss


class RL2:
    """
    Wraps an RL2Policy with PPO outer-loop training.

    Key methods:
      - collect_trial(): roll out K episodes in one task, preserving hidden state.
      - update(): PPO update over a batch of trials.
    """

    def __init__(
        self,
        policy: RL2Policy,
        num_episodes_per_trial: int = 2,
        max_episode_steps: int = 150,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        lr: float = 3e-4,
        num_epochs: int = 4,
        minibatch_size: int = 64,
        max_grad_norm: float = 1.0,
        device: str = "cpu",
    ):
        self.device = device
        self.policy = policy.to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.K = num_episodes_per_trial
        self.T = max_episode_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size
        self.max_grad_norm = max_grad_norm

    @torch.no_grad()
    def collect_trial(self, env, deterministic: bool = False) -> dict:
        """
        Collect K episodes in a single task, preserving GRU hidden state.

        Returns a dict with the full meta-trial flattened to a sequence
        of length T_total = K * max_episode_steps (with padding for short eps):
            obs:          [T_total, obs_dim]
            actions:      [T_total, action_dim]
            prev_actions: [T_total, action_dim]
            prev_rewards: [T_total, 1]
            prev_dones:   [T_total, 1]
            rewards:      [T_total]
            dones:        [T_total]   (episode-level dones; trial-level reset handled separately)
            log_probs:    [T_total]
            values:       [T_total]
            episode_returns: list of K floats (raw episode returns, in order)
        """
        device = self.device
        obs_dim = self.policy.obs_dim
        action_dim = self.policy.action_dim

        all_obs = []
        all_actions = []
        all_prev_actions = []
        all_prev_rewards = []
        all_prev_dones = []
        all_rewards = []
        all_dones = []
        all_log_probs = []
        all_values = []
        episode_returns = []

        hidden = self.policy.initial_hidden(batch_size=1, device=device)
        prev_action = torch.zeros(1, action_dim, device=device)
        prev_reward = torch.zeros(1, 1, device=device)
        prev_done = torch.zeros(1, 1, device=device)

        for ep_idx in range(self.K):
            obs = _env_reset(env)
            ep_return = 0.0
            ep_steps = 0

            for t in range(self.T):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                dist, value, hidden = self.policy.step(
                    obs_t, prev_action, prev_reward, prev_done, hidden
                )
                if deterministic:
                    action = dist.mean
                else:
                    action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)

                action_np = action.squeeze(0).cpu().numpy()
                next_obs, reward, done = _env_step(env, action_np)

                all_obs.append(obs)
                all_actions.append(action_np)
                all_prev_actions.append(prev_action.squeeze(0).cpu().numpy())
                all_prev_rewards.append(prev_reward.squeeze(0).cpu().numpy())
                all_prev_dones.append(prev_done.squeeze(0).cpu().numpy())
                all_rewards.append(float(reward))
                all_dones.append(float(done))
                all_log_probs.append(log_prob.item())
                all_values.append(value.squeeze().item())

                ep_return += float(reward)
                ep_steps += 1

                # Update prev_* for next step
                prev_action = action
                prev_reward = torch.tensor([[float(reward)]], device=device)
                prev_done = torch.tensor([[float(done)]], device=device)

                obs = next_obs
                if done:
                    break

            # Pad to T if episode ended early
            pad = self.T - ep_steps
            for _ in range(pad):
                all_obs.append(np.zeros(obs_dim, dtype=np.float32))
                all_actions.append(np.zeros(action_dim, dtype=np.float32))
                all_prev_actions.append(np.zeros(action_dim, dtype=np.float32))
                all_prev_rewards.append(np.zeros(1, dtype=np.float32))
                all_prev_dones.append(np.ones(1, dtype=np.float32))  # padding marked done
                all_rewards.append(0.0)
                all_dones.append(1.0)
                all_log_probs.append(0.0)
                all_values.append(0.0)

            episode_returns.append(ep_return)

        # Bootstrap value for GAE
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        _, last_value, _ = self.policy.step(
            obs_t, prev_action, prev_reward, prev_done, hidden
        )
        last_value = last_value.squeeze().item()

        return {
            "obs": np.array(all_obs, dtype=np.float32),
            "actions": np.array(all_actions, dtype=np.float32),
            "prev_actions": np.array(all_prev_actions, dtype=np.float32),
            "prev_rewards": np.array(all_prev_rewards, dtype=np.float32),
            "prev_dones": np.array(all_prev_dones, dtype=np.float32),
            "rewards": np.array(all_rewards, dtype=np.float32),
            "dones": np.array(all_dones, dtype=np.float32),
            "log_probs": np.array(all_log_probs, dtype=np.float32),
            "values": np.array(all_values, dtype=np.float32),
            "last_value": last_value,
            "episode_returns": episode_returns,
        }

    def _stack_trials(self, trials: list) -> dict:
        """Stack a list of trial dicts into batched tensors with GAE."""
        device = self.device
        B = len(trials)
        T_total = trials[0]["obs"].shape[0]

        obs = torch.as_tensor(np.stack([tr["obs"] for tr in trials]), device=device)
        actions = torch.as_tensor(np.stack([tr["actions"] for tr in trials]), device=device)
        prev_actions = torch.as_tensor(
            np.stack([tr["prev_actions"] for tr in trials]), device=device
        )
        prev_rewards = torch.as_tensor(
            np.stack([tr["prev_rewards"] for tr in trials]), device=device
        )
        prev_dones = torch.as_tensor(
            np.stack([tr["prev_dones"] for tr in trials]), device=device
        )
        rewards = torch.as_tensor(np.stack([tr["rewards"] for tr in trials]), device=device)
        dones = torch.as_tensor(np.stack([tr["dones"] for tr in trials]), device=device)
        old_log_probs = torch.as_tensor(
            np.stack([tr["log_probs"] for tr in trials]), device=device
        )
        values = torch.as_tensor(np.stack([tr["values"] for tr in trials]), device=device)
        last_values = torch.as_tensor(
            np.array([tr["last_value"] for tr in trials], dtype=np.float32),
            device=device,
        ).unsqueeze(1)

        # Append bootstrap for GAE
        values_extended = torch.cat([values, last_values], dim=1)  # [B, T+1]
        advantages, returns = compute_gae(
            rewards, values_extended, dones, self.gamma, self.gae_lambda
        )

        # Initial hidden state for outer-loop forward pass
        initial_hidden = self.policy.initial_hidden(batch_size=B, device=device)

        return {
            "obs": obs,
            "actions": actions,
            "prev_actions": prev_actions,
            "prev_rewards": prev_rewards,
            "prev_dones": prev_dones,
            "old_log_probs": old_log_probs,
            "advantages": advantages,
            "returns": returns,
            "initial_hidden": initial_hidden,
        }

    def update(self, trials: list) -> dict:
        """
        PPO update over a batch of meta-trials.

        Performs `num_epochs` passes. With recurrent policies, we keep each
        trial as one minibatch unit (cannot split within a trial without
        losing the GRU state). For simplicity we use full-batch per epoch.
        """
        batch = self._stack_trials(trials)
        diag_acc = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                    "mean_ratio": 0.0, "clip_fraction": 0.0}

        for epoch in range(self.num_epochs):
            loss, info = ppo_loss(
                self.policy, batch,
                clip_eps=self.clip_eps,
                vf_coef=self.vf_coef,
                ent_coef=self.ent_coef,
            )
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            for k in diag_acc:
                diag_acc[k] += info[k] / self.num_epochs

        diag_acc["total_loss"] = loss.item()
        return diag_acc


# --- gym/gymnasium compatibility helpers ---

def _env_reset(env):
    out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        return out[0]
    return out


def _env_step(env, action):
    out = env.step(action)
    if len(out) == 5:
        next_obs, reward, terminated, truncated, _ = out
        return next_obs, reward, bool(terminated or truncated)
    next_obs, reward, done, _ = out
    return next_obs, reward, bool(done)
