"""
PEARL trainer (discrete) — каркас. 

ВАЖНО: полный SAC-цикл уже есть в твоей algorithms/pearl/pearl.py для
MetaWorld (continuous actions). Для Procgen нужно:
  1) Заменить TanhGaussianPolicy -> PEARLPolicy (Categorical, см. networks.py).
  2) Заменить obs encoder в Q и в context_encoder на NatureCNN — уже сделано.
  3) Discrete SAC: target = min(Q1, Q2) - alpha*log_pi (по всем действиям, sum по dist.probs).

Этот файл реализует ТОЛЬКО loop сбора данных и логирования, чтобы получить
обученную модель, сохранить history.json и checkpoints. Если у тебя в repo
algorithms/pearl/pearl.py уже отлажен — импортируй оттуда обновлённую функцию
sac_update(...) и подставь сюда.
"""
import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from algorithms.pearl.procgen_encoder import ContextEncoder
from algorithms.pearl.procgen_networks import PEARLPolicy, PEARLQNetwork


class MultiTaskBuffer:
    """Отдельный кольцевой буфер на каждую задачу (level seed)."""
    def __init__(self, capacity=10_000):
        self.capacity = capacity
        self.buffers = defaultdict(list)

    def add(self, task_id, transition):
        buf = self.buffers[task_id]
        if len(buf) >= self.capacity:
            buf.pop(0)
        buf.append(transition)

    def sample_context(self, task_id, n=100):
        buf = self.buffers[task_id]
        if len(buf) == 0:
            return None
        idx = np.random.choice(len(buf), min(n, len(buf)), replace=False)
        return [buf[i] for i in idx]

    def sample_batch(self, task_id, n=256):
        return self.sample_context(task_id, n)

    def __len__(self):
        return sum(len(b) for b in self.buffers.values())


def stack_transitions(transitions, device="cpu"):
    obs = torch.from_numpy(np.stack([t["obs"] for t in transitions])).to(device)
    acts = torch.tensor([t["act"] for t in transitions], dtype=torch.long, device=device)
    rews = torch.tensor([t["rew"] for t in transitions], dtype=torch.float32, device=device)
    next_obs = torch.from_numpy(np.stack([t["next_obs"] for t in transitions])).to(device)
    dones = torch.tensor([t["done"] for t in transitions], dtype=torch.float32, device=device)
    return obs, acts, rews, next_obs, dones


def collect_rollout_with_z(env, policy, z, max_steps=500):
    """Один эпизод с фиксированным z."""
    transitions = []
    obs = env.reset()
    ep_ret = 0.0
    for _ in range(max_steps):
        obs_t = torch.from_numpy(obs)
        with torch.no_grad():
            action, _, _ = policy.act(obs_t, z)
        a = int(action.item())
        next_obs, reward, done, _ = env.step(a)
        transitions.append({"obs": obs, "act": a, "rew": reward,
                            "next_obs": next_obs, "done": float(done)})
        ep_ret += reward
        obs = next_obs
        if done:
            break
    return transitions, ep_ret


def train_pearl(env_name="coinrun", num_train_levels=50,
                total_steps=2_000_000, latent_dim=8,
                warmup_steps=10_000, context_size=100,
                lr=3e-4, output_dir="results/pearl",
                seed=0, log_every=5, ckpt_every=50,
                sac_update_fn=None):
    """
    sac_update_fn: callable(policy, q1, q2, target_q1, target_q2, encoder,
                            batch, context, optimizers, alpha) -> dict of losses
                  ВАЖНО: подставь сюда свою функцию из algorithms/pearl/pearl.py
                  Если None — будут только rollouts без обновления (для отладки).
    """
    from envs.procgen_env import ProcgenTaskSampler

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/checkpoints").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    sampler = ProcgenTaskSampler(env_name=env_name,
                                  num_train_levels=num_train_levels, seed=seed)
    policy = PEARLPolicy(num_actions=15, latent_dim=latent_dim)
    q1 = PEARLQNetwork(num_actions=15, latent_dim=latent_dim)
    q2 = PEARLQNetwork(num_actions=15, latent_dim=latent_dim)
    target_q1 = PEARLQNetwork(num_actions=15, latent_dim=latent_dim)
    target_q2 = PEARLQNetwork(num_actions=15, latent_dim=latent_dim)
    target_q1.load_state_dict(q1.state_dict())
    target_q2.load_state_dict(q2.state_dict())
    encoder = ContextEncoder(num_actions=15, latent_dim=latent_dim)

    optimizers = {
        "policy": torch.optim.Adam(policy.parameters(), lr=lr),
        "q1": torch.optim.Adam(q1.parameters(), lr=lr),
        "q2": torch.optim.Adam(q2.parameters(), lr=lr),
        "encoder": torch.optim.Adam(encoder.parameters(), lr=lr),
    }
    alpha = 0.1  # фиксированная энтропия для упрощения

    buffer = MultiTaskBuffer(capacity=10_000)
    history = {"step": [], "mean_return": [], "kl": [],
               "q_loss": [], "pi_loss": [], "wallclock": []}
    steps_done = 0
    iteration = 0
    t_start = time.time()

    while steps_done < total_steps:
        # 1. Сэмплируем задачу, делаем prior + posterior rollouts
        level = sampler.sample_train_task()
        env = sampler.make_env(level)

        # Prior rollout (z ~ N(0,I))
        z_prior = encoder.sample_prior(batch_size=1).squeeze(0)
        with torch.no_grad():
            trans_prior, ret_prior = collect_rollout_with_z(env, policy, z_prior)
        for t in trans_prior:
            buffer.add(level, t)
        steps_done += len(trans_prior)

        # Posterior rollout (z из накопленного контекста)
        ctx = buffer.sample_context(level, n=context_size)
        if ctx is not None and len(ctx) >= 10:
            obs_c, act_c, rew_c, next_obs_c, _ = stack_transitions(ctx)
            with torch.no_grad():
                mu, var = encoder(obs_c, act_c, rew_c, next_obs_c)
                z_post = encoder.sample(mu, var)
            trans_post, ret_post = collect_rollout_with_z(env, policy, z_post)
            for t in trans_post:
                buffer.add(level, t)
            steps_done += len(trans_post)
        else:
            ret_post = ret_prior

        env.close()

        # 2. SAC update (несколько шагов на разных задачах)
        if steps_done > warmup_steps and sac_update_fn is not None:
            for _ in range(10):  # gradient steps per env step
                # Sample task uniformly
                tasks_with_data = [t for t, b in buffer.buffers.items() if len(b) >= 50]
                if not tasks_with_data:
                    break
                task = np.random.choice(tasks_with_data)
                batch = buffer.sample_batch(task, n=256)
                context = buffer.sample_context(task, n=context_size)
                losses = sac_update_fn(policy, q1, q2, target_q1, target_q2,
                                       encoder, batch, context, optimizers, alpha)
        else:
            losses = {"q_loss": 0.0, "pi_loss": 0.0, "kl": 0.0}

        # Логирование
        if iteration % log_every == 0:
            elapsed = time.time() - t_start
            print(f"[PEARL] iter={iteration:4d} steps={steps_done:8d} "
                  f"ret_prior={ret_prior:.2f} ret_post={ret_post:.2f} "
                  f"elapsed={elapsed/60:.1f}min")
            history["step"].append(steps_done)
            history["mean_return"].append((ret_prior + ret_post) / 2)
            history["kl"].append(losses.get("kl", 0.0))
            history["q_loss"].append(losses.get("q_loss", 0.0))
            history["pi_loss"].append(losses.get("pi_loss", 0.0))
            history["wallclock"].append(elapsed)

        if iteration % ckpt_every == 0 and iteration > 0:
            torch.save({
                "policy": policy.state_dict(),
                "encoder": encoder.state_dict(),
                "q1": q1.state_dict(), "q2": q2.state_dict(),
            }, f"{output_dir}/checkpoints/step_{steps_done}.pt")
        iteration += 1

    torch.save({
        "policy": policy.state_dict(),
        "encoder": encoder.state_dict(),
        "q1": q1.state_dict(), "q2": q2.state_dict(),
    }, f"{output_dir}/final.pt")
    with open(f"{output_dir}/history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(f"{output_dir}/config.json", "w") as f:
        json.dump({
            "method": "pearl", "env_name": env_name,
            "num_train_levels": num_train_levels,
            "total_steps": total_steps, "latent_dim": latent_dim,
            "lr": lr, "seed": seed, "final_steps": steps_done,
        }, f, indent=2)
    print(f"[PEARL] Done. Saved to {output_dir}")
