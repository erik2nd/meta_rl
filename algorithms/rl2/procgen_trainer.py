"""
RL² trainer для Procgen.

Главная идея: один MEAT-TRIAL = K эпизодов на ОДНОЙ задаче (level seed) ПОДРЯД
с НЕ-сбрасывающимся hidden state. Затем sample новую задачу и новый trial.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from algorithms.rl2.procgen_policy import RL2Policy


def collect_meta_trial(env, policy, episodes_per_trial=2, max_steps_per_ep=500,
                       device="cpu"):
    """
    Один meta-trial = несколько эпизодов на одной задаче.
    hidden state переносится между эпизодами!
    """
    obs_buf, act_buf, rew_buf, done_buf = [], [], [], []
    prev_act_buf, prev_rew_buf, prev_done_buf = [], [], []
    val_buf, logp_buf = [], []
    ep_returns = []

    hidden = policy.init_hidden(batch_size=1, device=device)
    prev_action = torch.zeros(1, dtype=torch.long, device=device)
    prev_reward = torch.zeros(1, device=device)
    prev_done = torch.ones(1, device=device)  # first step "после" done

    for ep in range(episodes_per_trial):
        obs = env.reset()
        ep_ret = 0.0
        for t in range(max_steps_per_ep):
            obs_t = torch.from_numpy(obs).to(device)
            with torch.no_grad():
                action, logp, value, hidden = policy.act_single(
                    obs_t, prev_action, prev_reward, prev_done, hidden
                )
            a = int(action.item())
            next_obs, reward, done, _ = env.step(a)

            obs_buf.append(obs)
            act_buf.append(a)
            rew_buf.append(reward)
            done_buf.append(float(done))
            prev_act_buf.append(int(prev_action.item()))
            prev_rew_buf.append(float(prev_reward.item()))
            prev_done_buf.append(float(prev_done.item()))
            val_buf.append(value.item())
            logp_buf.append(logp.item())

            prev_action = action
            prev_reward = torch.tensor([reward], device=device)
            prev_done = torch.tensor([float(done)], device=device)
            ep_ret += reward
            obs = next_obs
            if done:
                break
        ep_returns.append(ep_ret)
        # Hidden state НЕ сбрасываем — это и есть адаптация!

    # bootstrap
    obs_t = torch.from_numpy(obs).to(device)
    with torch.no_grad():
        _, _, last_val, _ = policy.act_single(
            obs_t, prev_action, prev_reward, prev_done, hidden
        )
    val_buf.append(last_val.item())

    return {
        "obs": np.stack(obs_buf),
        "actions": np.array(act_buf, dtype=np.int64),
        "rewards": np.array(rew_buf, dtype=np.float32),
        "dones": np.array(done_buf, dtype=np.float32),
        "prev_actions": np.array(prev_act_buf, dtype=np.int64),
        "prev_rewards": np.array(prev_rew_buf, dtype=np.float32),
        "prev_dones": np.array(prev_done_buf, dtype=np.float32),
        "values": np.array(val_buf, dtype=np.float32),
        "old_logp": np.array(logp_buf, dtype=np.float32),
    }, ep_returns


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    T = len(rewards)
    advs = np.zeros(T, dtype=np.float32)
    last = 0.0
    for t in reversed(range(T)):
        nv = values[t + 1] if t + 1 < T else 0.0
        nt = 1.0 - dones[t]
        delta = rewards[t] + gamma * nv * nt - values[t]
        last = delta + gamma * lam * nt * last
        advs[t] = last
    rets = advs + values[:T]
    return advs, rets


def rl2_ppo_update(policy, optimizer, trials, clip=0.2, vf_coef=0.5,
                   ent_coef=0.01, epochs=4, device="cpu"):
    """
    Каждый trial — отдельная последовательность. Обновляем PPO loss по всем.
    """
    losses = {"pi": 0.0, "v": 0.0, "ent": 0.0, "n": 0}
    for _ in range(epochs):
        for trial in trials:
            advs, rets = compute_gae(trial["rewards"], trial["values"], trial["dones"])
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)

            obs = torch.from_numpy(trial["obs"]).to(device).unsqueeze(0)  # B=1
            acts = torch.from_numpy(trial["actions"]).to(device).unsqueeze(0)
            prev_acts = torch.from_numpy(trial["prev_actions"]).to(device).unsqueeze(0)
            prev_rews = torch.from_numpy(trial["prev_rewards"]).to(device).unsqueeze(0)
            prev_dones = torch.from_numpy(trial["prev_dones"]).to(device).unsqueeze(0)
            old_lp = torch.from_numpy(trial["old_logp"]).to(device)
            advs_t = torch.from_numpy(advs).to(device)
            rets_t = torch.from_numpy(rets).to(device)

            h0 = policy.init_hidden(batch_size=1, device=device)
            new_lp, value, ent = policy.evaluate_sequence(
                obs, prev_acts, prev_rews, prev_dones, acts, h0
            )
            new_lp = new_lp.squeeze(0)
            value = value.squeeze(0)
            ent = ent.squeeze(0)

            ratio = torch.exp(new_lp - old_lp)
            surr1 = ratio * advs_t
            surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * advs_t
            pi_loss = -torch.min(surr1, surr2).mean()
            v_loss = F.mse_loss(value, rets_t)
            ent_loss = ent.mean()
            loss = pi_loss + vf_coef * v_loss - ent_coef * ent_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

            losses["pi"] += pi_loss.item()
            losses["v"] += v_loss.item()
            losses["ent"] += ent_loss.item()
            losses["n"] += 1
    for k in ("pi", "v", "ent"):
        losses[k] /= max(losses["n"], 1)
    return losses


def train_rl2(env_name="coinrun", num_train_levels=50,
              total_steps=5_000_000, trials_per_iter=4,
              episodes_per_trial=2, max_steps_per_ep=500,
              lr=5e-4, output_dir="results/rl2",
              seed=0, log_every=5, ckpt_every=50):
    from envs.procgen_env import ProcgenTaskSampler

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/checkpoints").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    sampler = ProcgenTaskSampler(env_name=env_name,
                                  num_train_levels=num_train_levels, seed=seed)
    policy = RL2Policy(num_actions=15)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    history = {"step": [], "mean_return": [],
               "first_ep": [], "last_ep": [], "adaptation_gain": [],
               "pi_loss": [], "v_loss": [], "entropy": [], "wallclock": []}
    steps_done = 0
    iteration = 0
    t_start = time.time()

    while steps_done < total_steps:
        trials = []
        all_returns = []
        first_ep_rets = []
        last_ep_rets = []
        for _ in range(trials_per_iter):
            level = sampler.sample_train_task()
            env = sampler.make_env(level)
            trial_data, ep_returns = collect_meta_trial(
                env, policy, episodes_per_trial=episodes_per_trial,
                max_steps_per_ep=max_steps_per_ep
            )
            env.close()
            trials.append(trial_data)
            all_returns.extend(ep_returns)
            if len(ep_returns) >= 1:
                first_ep_rets.append(ep_returns[0])
                last_ep_rets.append(ep_returns[-1])
            steps_done += len(trial_data["actions"])

        losses = rl2_ppo_update(policy, optimizer, trials)

        if iteration % log_every == 0:
            mean_r = float(np.mean(all_returns)) if all_returns else 0.0
            fe = float(np.mean(first_ep_rets)) if first_ep_rets else 0.0
            le = float(np.mean(last_ep_rets)) if last_ep_rets else 0.0
            gain = le - fe
            elapsed = time.time() - t_start
            print(f"[RL2] iter={iteration:4d} steps={steps_done:8d} "
                  f"ret={mean_r:6.2f} first_ep={fe:.2f} last_ep={le:.2f} "
                  f"gain={gain:+.2f} elapsed={elapsed/60:.1f}min")
            history["step"].append(steps_done)
            history["mean_return"].append(mean_r)
            history["first_ep"].append(fe)
            history["last_ep"].append(le)
            history["adaptation_gain"].append(gain)
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
            "method": "rl2", "env_name": env_name,
            "num_train_levels": num_train_levels,
            "total_steps": total_steps,
            "trials_per_iter": trials_per_iter,
            "episodes_per_trial": episodes_per_trial,
            "lr": lr, "seed": seed, "final_steps": steps_done,
        }, f, indent=2)
    print(f"[RL2] Done. Saved to {output_dir}")
