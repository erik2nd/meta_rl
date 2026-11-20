"""
Replay buffers for PEARL.

Each task gets its own RL buffer (for SAC updates) and its own context
buffer (for sampling transitions used by the context encoder).

Both are simple fixed-size circular buffers.
"""
import numpy as np
import torch


class ReplayBuffer:
    """Simple circular replay buffer for transitions (s, a, r, s', done)."""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

        self.size = 0
        self.ptr = 0

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_path(self, path: dict):
        """Bulk-add a trajectory (dict of np arrays of shape [T, ...])."""
        T = len(path["rewards"])
        for t in range(T):
            self.add(
                path["obs"][t],
                path["actions"][t],
                path["rewards"][t],
                path["next_obs"][t],
                path["dones"][t],
            )

    def sample(self, batch_size: int, device: str = "cpu") -> dict:
        if self.size == 0:
            raise RuntimeError("Sampling from empty buffer")
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[idx], device=device),
            "actions": torch.as_tensor(self.actions[idx], device=device),
            "rewards": torch.as_tensor(self.rewards[idx], device=device),
            "next_obs": torch.as_tensor(self.next_obs[idx], device=device),
            "dones": torch.as_tensor(self.dones[idx], device=device),
        }


class MultiTaskBuffer:
    """
    Holds one ReplayBuffer per task (for SAC training) and one context
    buffer per task (smaller, recent transitions for encoder).
    """

    def __init__(
        self,
        num_tasks: int,
        obs_dim: int,
        action_dim: int,
        rl_capacity: int = 1_000_000,
        ctx_capacity: int = 100_000,
    ):
        self.num_tasks = num_tasks
        self.rl_buffers = [
            ReplayBuffer(rl_capacity, obs_dim, action_dim) for _ in range(num_tasks)
        ]
        self.ctx_buffers = [
            ReplayBuffer(ctx_capacity, obs_dim, action_dim) for _ in range(num_tasks)
        ]

    def add_path(self, task_idx: int, path: dict, into_context: bool = True):
        self.rl_buffers[task_idx].add_path(path)
        if into_context:
            self.ctx_buffers[task_idx].add_path(path)

    def clear_context(self, task_idx: int):
        self.ctx_buffers[task_idx] = ReplayBuffer(
            self.ctx_buffers[task_idx].capacity,
            self.rl_buffers[task_idx].obs.shape[1],
            self.rl_buffers[task_idx].actions.shape[1],
        )

    def sample_rl_batch(self, task_indices: list, batch_size: int, device: str = "cpu"):
        """Sample one minibatch per task; stack along task dim."""
        batches = [
            self.rl_buffers[i].sample(batch_size, device=device) for i in task_indices
        ]
        # Stack -> [num_tasks, batch_size, ...]
        out = {}
        for key in batches[0]:
            out[key] = torch.stack([b[key] for b in batches], dim=0)
        return out

    def sample_context(self, task_indices: list, batch_size: int, device: str = "cpu"):
        """
        Sample a context set per task: [num_tasks, batch_size, ctx_dim].
        Context features are (s, a, r, s').
        """
        batches = [
            self.ctx_buffers[i].sample(batch_size, device=device) for i in task_indices
        ]
        contexts = []
        for b in batches:
            ctx = torch.cat(
                [b["obs"], b["actions"], b["rewards"].unsqueeze(-1), b["next_obs"]],
                dim=-1,
            )
            contexts.append(ctx)
        return torch.stack(contexts, dim=0)  # [T, N, ctx_dim]
