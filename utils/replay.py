"""
Trajectory collection utilities.

Rolls out a policy in an environment, collects (obs, action, reward, done)
tuples, and packs them into a batch suitable for MAML/RL?/PEARL.
"""
import numpy as np
import torch


@torch.no_grad()
def collect_trajectory(env, policy, max_steps: int, device: str = "cpu") -> dict:
    """
    Roll out one episode in `env` using `policy`.

    Returns dict with tensors of shape [T, ...] for one trajectory:
        obs: [T, obs_dim]
        actions: [T, action_dim]
        rewards: [T]
        dones: [T]
    """
    obs_list, action_list, reward_list, done_list = [], [], [], []

    obs, _ = env.reset() if _is_gymnasium_api(env) else (env.reset(), {})

    for t in range(max_steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action, _ = policy.get_action(obs_t, deterministic=False)
        action_np = action.squeeze(0).cpu().numpy()

        step_out = env.step(action_np)
        # Handle both gym and gymnasium step API
        if len(step_out) == 5:
            next_obs, reward, terminated, truncated, _ = step_out
            done = terminated or truncated
        else:
            next_obs, reward, done, _ = step_out

        obs_list.append(obs)
        action_list.append(action_np)
        reward_list.append(float(reward))
        done_list.append(float(done))

        obs = next_obs
        if done:
            break

    # Pad to max_steps so all trajectories in a batch have same length
    T = len(obs_list)
    obs_dim = len(obs_list[0])
    action_dim = len(action_list[0])

    if T < max_steps:
        pad = max_steps - T
        obs_list += [np.zeros(obs_dim, dtype=np.float32)] * pad
        action_list += [np.zeros(action_dim, dtype=np.float32)] * pad
        reward_list += [0.0] * pad
        done_list += [1.0] * pad  # mark padding as done

    return {
        "obs": torch.as_tensor(np.array(obs_list), dtype=torch.float32, device=device),
        "actions": torch.as_tensor(np.array(action_list), dtype=torch.float32, device=device),
        "rewards": torch.as_tensor(np.array(reward_list), dtype=torch.float32, device=device),
        "dones": torch.as_tensor(np.array(done_list), dtype=torch.float32, device=device),
        "length": T,
    }


def collect_task_batch(env_fn, policy, num_trajectories: int, max_steps: int, device: str = "cpu") -> dict:
    """
    Collect multiple trajectories from the SAME task. `env_fn` is a zero-arg
    callable that returns a fresh env instance (or the same env reset).

    Returns dict of stacked tensors:
        obs: [N, T, obs_dim]
        actions: [N, T, action_dim]
        rewards: [N, T]
        dones: [N, T]
    """
    trajs = []
    for _ in range(num_trajectories):
        env = env_fn()
        traj = collect_trajectory(env, policy, max_steps, device)
        trajs.append(traj)

    batch = {
        "obs": torch.stack([t["obs"] for t in trajs]),
        "actions": torch.stack([t["actions"] for t in trajs]),
        "rewards": torch.stack([t["rewards"] for t in trajs]),
        "dones": torch.stack([t["dones"] for t in trajs]),
        "lengths": [t["length"] for t in trajs],
    }
    return batch


def _is_gymnasium_api(env) -> bool:
    """Heuristic: check whether env.reset() returns (obs, info) tuple."""
    # Meta-World currently uses old gym API (reset returns obs only),
    # but newer versions may switch to gymnasium. We auto-detect.
    try:
        sig = env.reset.__doc__ or ""
        return "info" in sig.lower()
    except Exception:
        return False
