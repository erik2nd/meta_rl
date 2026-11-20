"""
Trajectory collection for PEARL.

Key difference from MAML/RL^2: rollouts are conditioned on a sampled
task embedding z. During warm-up and "prior" rollouts, z is sampled from
the unit prior N(0, I). During "posterior" rollouts, z is sampled from
q_phi(z | c) using the recently collected transitions.
"""
import numpy as np
import torch


@torch.no_grad()
def rollout(env, algo, z: torch.Tensor, max_steps: int, deterministic: bool = False) -> dict:
    """
    Roll out one episode conditioned on a fixed z.

    Returns:
        path dict with arrays:
            obs:      [T, obs_dim]
            actions:  [T, action_dim]
            rewards:  [T]
            next_obs: [T, obs_dim]
            dones:    [T]
            length:   T
    """
    obs = _env_reset(env)
    obs_buf, act_buf, rew_buf, next_buf, done_buf = [], [], [], [], []

    for t in range(max_steps):
        action = algo.get_action(obs, z, deterministic=deterministic)
        next_obs, reward, done = _env_step(env, action)
        obs_buf.append(np.asarray(obs, dtype=np.float32))
        act_buf.append(np.asarray(action, dtype=np.float32))
        rew_buf.append(float(reward))
        next_buf.append(np.asarray(next_obs, dtype=np.float32))
        done_buf.append(float(done))
        obs = next_obs
        if done:
            break

    return {
        "obs": np.asarray(obs_buf, dtype=np.float32),
        "actions": np.asarray(act_buf, dtype=np.float32),
        "rewards": np.asarray(rew_buf, dtype=np.float32),
        "next_obs": np.asarray(next_buf, dtype=np.float32),
        "dones": np.asarray(done_buf, dtype=np.float32),
        "length": len(rew_buf),
        "return": float(np.sum(rew_buf)),
    }


def collect_rollouts_with_z_strategy(
    env,
    algo,
    num_steps: int,
    max_episode_steps: int,
    z_strategy: str = "prior",
    context: torch.Tensor = None,
) -> list:
    """
    Collect rollouts until `num_steps` env steps are gathered.

    z_strategy:
        - "prior":     z ~ N(0, I) at the start of each rollout
        - "posterior": z ~ q_phi(z | context). Must provide `context`.
        - "fixed":     use a pre-computed z (must be passed as `context` here).
    """
    device = algo.device
    paths = []
    steps_collected = 0

    while steps_collected < num_steps:
        if z_strategy == "prior":
            z = algo.sample_z_prior(num_tasks=1).squeeze(0)
        elif z_strategy == "posterior":
            assert context is not None, "Need context for posterior"
            mu, sigma_sq = algo.infer_posterior(context.unsqueeze(0))
            z = algo.sample_z(mu, sigma_sq).squeeze(0)
        elif z_strategy == "fixed":
            z = context  # caller passes z directly
        else:
            raise ValueError(f"Unknown z_strategy: {z_strategy}")

        path = rollout(env, algo, z, max_episode_steps)
        paths.append(path)
        steps_collected += path["length"]

    return paths


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
