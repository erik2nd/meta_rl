"""
PPO trainer для baseline на ОДНОМ уровне Procgen.

Цель: показать, что vanilla PPO выучит конкретный уровень, но не сможет
играть на новом. Это создаёт контраст с Meta-RL методами.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from algorithms.ppo_baseline.policy import PPOPolicy


def compute_gae(rewards, values, dones, gamma=0.999, lam=0.95):
    """Generalized Advantage Estimation."""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = values[t + 1] if t + 1 < T else 0.0
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + np.array(values[:T], dtype=np.float32)
    return advantages, returns


def collect_rollout(env, policy, rollout_steps=128, device="cpu"):
    """Один rollout фиксированной длины. Если эпизод закончился — сбрасываем."""
    obs_buf, act_buf, logp_buf = [], [], []
    rew_buf, val_buf, done_buf = [], [], []
    ep_returns = []

    obs = env.reset()
    ep_ret = 0.0
    for _ in range(rollout_steps):
        obs_t = torch.from_numpy(obs).to(device)
        with torch.no_grad():
            action, logp, value, _ = policy.act(obs_t)
        a = int(action.item())
        next_obs, reward, done, _ = env.step(a)

        obs_buf.append(obs)
        act_buf.append(a)
        logp_buf.append(logp.item())
        rew_buf.append(reward)
        val_buf.append(value.item())
        done_buf.append(float(done))
        ep_ret += reward

        if done:
            ep_returns.append(ep_ret)
            ep_ret = 0.0
            obs = env.reset()
        else:
            obs = next_obs

    # bootstrap value
    obs_t = torch.from_numpy(obs).to(device)
    with torch.no_grad():
        _, _, last_value, _ = policy.act(obs_t)
    val_buf.append(last_value.item())

    advs, rets = compute_gae(rew_buf, val_buf, done_buf)
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)

    batch = {
        "obs": np.stack(obs_buf),
        "actions": np.array(act_buf, dtype=np.int64),
        "old_logp": np.array(logp_buf, dtype=np.float32),
        "advantages": advs,
        "returns": rets,
    }
    return batch, ep_returns


def ppo_update(policy, optimizer, batch, clip=0.2, vf_coef=0.5,
               ent_coef=0.01, epochs=4, minibatch=64, device="cpu"):
    """Несколько эпох SGD по rollout-буферу."""
    obs    = torch.from_numpy(batch["obs"]).to(device)
    acts   = torch.from_numpy(batch["actions"]).to(device)
    oldlp  = torch.from_numpy(batch["old_logp"]).to(device)
    advs   = torch.from_numpy(batch["advantages"]).to(device)
    rets   = torch.from_numpy(batch["returns"]).to(device)

    N = obs.shape[0]
    idx = np.arange(N)
    losses = {"pi": 0.0, "v": 0.0, "ent": 0.0, "n": 0}

    for _ in range(epochs):
        np.random.shuffle(idx)
        for start in range(0, N, minibatch):
            mb = idx[start:start + minibatch]
            new_lp, value, ent = policy.evaluate(obs[mb], acts[mb])
            ratio = torch.exp(new_lp - oldlp[mb])
            surr1 = ratio * advs[mb]
            surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * advs[mb]
            pi_loss = -torch.min(surr1, surr2).mean()
            v_loss  = F.mse_loss(value, rets[mb])
            ent_b   = ent.mean()
            loss = pi_loss + vf_coef * v_loss - ent_coef * ent_b

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

            losses["pi"]  += pi_loss.item()
            losses["v"]   += v_loss.item()
            losses["ent"] += ent_b.item()
            losses["n"]   += 1
    for k in ("pi", "v", "ent"):
        losses[k] /= max(losses["n"], 1)
    return losses


def train_ppo_single_level(env_name="coinrun", level_seed=42,
                            total_steps=1_000_000, rollout_steps=128,
                            lr=5e-4, output_dir="results/ppo_baseline",
                            seed=0, log_every=10, ckpt_every=100):
    from envs.procgen_env import ProcgenTaskWrapper

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/checkpoints").mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = ProcgenTaskWrapper(env_name=env_name, level_seed=level_seed)
    policy = PPOPolicy(num_actions=env.action_space.n)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    history = {"step": [], "mean_return": [], "pi_loss": [],
               "v_loss": [], "entropy": [], "wallclock": []}
    steps_done = 0
    iteration = 0
    t_start = time.time()
    running_returns = []

    while steps_done < total_steps:
        batch, ep_returns = collect_rollout(env, policy, rollout_steps)
        losses = ppo_update(policy, optimizer, batch)

        steps_done += rollout_steps
        running_returns.extend(ep_returns)
        if len(running_returns) > 100:
            running_returns = running_returns[-100:]

        if iteration % log_every == 0:
            mean_r = float(np.mean(running_returns)) if running_returns else 0.0
            elapsed = time.time() - t_start
            print(f"[PPO] iter={iteration:5d} steps={steps_done:8d} "
                  f"return={mean_r:6.2f} pi_loss={losses['pi']:.3f} "
                  f"v_loss={losses['v']:.3f} ent={losses['ent']:.3f} "
                  f"elapsed={elapsed/60:.1f}min")
            history["step"].append(steps_done)
            history["mean_return"].append(mean_r)
            history["pi_loss"].append(losses["pi"])
            history["v_loss"].append(losses["v"])
            history["entropy"].append(losses["ent"])
            history["wallclock"].append(elapsed)

        if iteration % ckpt_every == 0 and iteration > 0:
            torch.save(policy.state_dict(),
                       f"{output_dir}/checkpoints/step_{steps_done}.pt")
        iteration += 1

    torch.save(policy.state_dict(), f"{output_dir}/final.pt")
    with open(f"{output_dir}/history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(f"{output_dir}/config.json", "w") as f:
        json.dump({
            "method": "ppo_baseline", "env_name": env_name,
            "level_seed": level_seed, "total_steps": total_steps,
            "rollout_steps": rollout_steps, "lr": lr, "seed": seed,
            "final_steps": steps_done,
        }, f, indent=2)
    env.close()
    print(f"[PPO] Done. Saved to {output_dir}")
