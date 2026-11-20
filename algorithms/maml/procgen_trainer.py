"""
MAML trainer для Procgen.

ВАЖНОЕ ЗАМЕЧАНИЕ:
На CPU MAML c CNN практически неработоспособен из-за second-order градиентов
по большой сети. Рекомендуется:
  - Использовать FOMAML (first-order MAML): игнорируем вторую производную.
  - Или вообще пропустить MAML на Procgen и ограничиться текстовым сравнением.

Этот файл реализует FOMAML с PPO во внутреннем цикле.
Использует learn2learn для удобства, как в твоём существующем maml.py.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    import learn2learn as l2l
    _HAS_L2L = True
except ImportError:
    _HAS_L2L = False

from algorithms.ppo_baseline.policy import PPOPolicy


def collect_short_rollout(env, policy, steps=64, device="cpu"):
    """Короткий rollout для внутреннего цикла MAML."""
    obs_buf, act_buf, rew_buf, val_buf, logp_buf, done_buf = [], [], [], [], [], []
    obs = env.reset()
    for _ in range(steps):
        obs_t = torch.from_numpy(obs).to(device)
        action, logp, value, _ = policy.act(obs_t)
        a = int(action.item())
        next_obs, reward, done, _ = env.step(a)
        obs_buf.append(obs)
        act_buf.append(a)
        rew_buf.append(reward)
        val_buf.append(value.item())
        logp_buf.append(logp.item())
        done_buf.append(float(done))
        obs = next_obs if not done else env.reset()

    rets = []
    R = 0.0
    for r, d in zip(reversed(rew_buf), reversed(done_buf)):
        R = r + 0.99 * R * (1 - d)
        rets.insert(0, R)
    return {
        "obs": np.stack(obs_buf),
        "actions": np.array(act_buf, dtype=np.int64),
        "returns": np.array(rets, dtype=np.float32),
        "old_logp": np.array(logp_buf, dtype=np.float32),
    }


def inner_update(policy, batch, lr=0.1, device="cpu"):
    """Один шаг градиентного спуска для адаптации к задаче."""
    obs = torch.from_numpy(batch["obs"]).to(device)
    acts = torch.from_numpy(batch["actions"]).to(device)
    rets = torch.from_numpy(batch["returns"]).to(device)

    new_lp, value, _ = policy.evaluate(obs, acts)
    advs = rets - value.detach()
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    loss = -(new_lp * advs).mean() + 0.5 * F.mse_loss(value, rets)

    grads = torch.autograd.grad(loss, policy.parameters(),
                                 create_graph=False, allow_unused=True)
    new_state = {}
    for (name, p), g in zip(policy.named_parameters(), grads):
        if g is None:
            new_state[name] = p
        else:
            new_state[name] = p - lr * g
    return new_state, loss.item()


def train_maml_fomaml(env_name="coinrun", num_train_levels=50,
                      total_steps=2_000_000, tasks_per_meta_iter=4,
                      inner_steps=1, inner_lr=0.1, outer_lr=3e-4,
                      rollout_steps=64,
                      output_dir="results/maml", seed=0,
                      log_every=5, ckpt_every=50):
    from envs.procgen_env import ProcgenTaskSampler

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/checkpoints").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if not _HAS_L2L:
        print("[MAML] WARNING: learn2learn не установлен — используем "
              "ручную реализацию FOMAML.")

    sampler = ProcgenTaskSampler(env_name=env_name,
                                  num_train_levels=num_train_levels, seed=seed)
    policy = PPOPolicy(num_actions=15)
    optimizer = torch.optim.Adam(policy.parameters(), lr=outer_lr)

    history = {"step": [], "pre_adapt_return": [], "post_adapt_return": [],
               "adaptation_gain": [], "meta_loss": [], "wallclock": []}
    steps_done = 0
    iteration = 0
    t_start = time.time()

    while steps_done < total_steps:
        meta_grads = [torch.zeros_like(p) for p in policy.parameters()]
        pre_rets, post_rets = [], []
        meta_loss_total = 0.0

        for _ in range(tasks_per_meta_iter):
            level = sampler.sample_train_task()
            env = sampler.make_env(level)

            # 1. Pre-adaptation rollout
            pre_batch = collect_short_rollout(env, policy, steps=rollout_steps)
            pre_rets.append(float(pre_batch["returns"][0]))

            # 2. Inner loop — adapt copy of policy
            adapted_policy = PPOPolicy(num_actions=15)
            adapted_policy.load_state_dict(policy.state_dict())
            for _ in range(inner_steps):
                adapt_batch = collect_short_rollout(env, adapted_policy,
                                                     steps=rollout_steps)
                obs = torch.from_numpy(adapt_batch["obs"])
                acts = torch.from_numpy(adapt_batch["actions"])
                rets = torch.from_numpy(adapt_batch["returns"])
                new_lp, value, _ = adapted_policy.evaluate(obs, acts)
                advs = rets - value.detach()
                advs = (advs - advs.mean()) / (advs.std() + 1e-8)
                in_loss = -(new_lp * advs).mean() + 0.5 * F.mse_loss(value, rets)
                grads = torch.autograd.grad(in_loss,
                                             adapted_policy.parameters())
                with torch.no_grad():
                    for p, g in zip(adapted_policy.parameters(), grads):
                        p.sub_(inner_lr * g)

            # 3. Post-adaptation rollout + outer loss
            post_batch = collect_short_rollout(env, adapted_policy,
                                                steps=rollout_steps)
            post_rets.append(float(post_batch["returns"][0]))
            obs = torch.from_numpy(post_batch["obs"])
            acts = torch.from_numpy(post_batch["actions"])
            rets = torch.from_numpy(post_batch["returns"])
            new_lp, value, _ = adapted_policy.evaluate(obs, acts)
            advs = rets - value.detach()
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            outer_loss = -(new_lp * advs).mean() + 0.5 * F.mse_loss(value, rets)
            meta_loss_total += outer_loss.item()

            # FOMAML: используем grad от adapted_policy на тех же параметрах
            grads = torch.autograd.grad(outer_loss,
                                         adapted_policy.parameters())
            for mg, g in zip(meta_grads, grads):
                mg += g / tasks_per_meta_iter

            steps_done += 2 * rollout_steps  # pre + post
            env.close()

        # Apply meta-gradient to original policy
        optimizer.zero_grad()
        for p, mg in zip(policy.parameters(), meta_grads):
            p.grad = mg.clone()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()

        if iteration % log_every == 0:
            pre_m = float(np.mean(pre_rets))
            post_m = float(np.mean(post_rets))
            elapsed = time.time() - t_start
            print(f"[MAML] iter={iteration:4d} steps={steps_done:8d} "
                  f"pre={pre_m:.2f} post={post_m:.2f} gain={post_m-pre_m:+.2f} "
                  f"loss={meta_loss_total/tasks_per_meta_iter:.3f} "
                  f"elapsed={elapsed/60:.1f}min")
            history["step"].append(steps_done)
            history["pre_adapt_return"].append(pre_m)
            history["post_adapt_return"].append(post_m)
            history["adaptation_gain"].append(post_m - pre_m)
            history["meta_loss"].append(meta_loss_total / tasks_per_meta_iter)
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
            "method": "maml_fomaml", "env_name": env_name,
            "num_train_levels": num_train_levels,
            "total_steps": total_steps,
            "inner_lr": inner_lr, "outer_lr": outer_lr,
            "inner_steps": inner_steps, "seed": seed,
            "final_steps": steps_done,
        }, f, indent=2)
    print(f"[MAML] Done. Saved to {output_dir}")
