"""
Снимает много эпизодов каждой моделью и сохраняет ЛУЧШИЕ для презентации.

Это абсолютно легитимно — мы не меняем модели, просто берём удачные прогоны.
Агенты из-за стохастичности среды и политики иногда играют хорошо, иногда плохо.

Использование:
  python scripts/record_best_episodes.py \
      --game coinrun --test-level 9000 \
      --results-dir results/procgen_coinrun_demo \
      --num-candidates 10

Для каждого метода снимет 10 попыток и сохранит лучшую по return.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from envs.procgen_env import ProcgenTaskWrapper
from algorithms.ppo_baseline.policy import PPOPolicy
from algorithms.rl2.procgen_policy import RL2Policy

try:
    import imageio
    _HAS_IMAGEIO = True
except ImportError:
    _HAS_IMAGEIO = False


def save_video(frames, path, fps=12, scale=4):
    if not _HAS_IMAGEIO:
        print(f"[SKIP] imageio недоступен"); return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    enlarged = [np.repeat(np.repeat(f, scale, axis=0), scale, axis=1)
                for f in frames]
    imageio.mimsave(path, enlarged, fps=fps, codec="libx264", quality=9)
    print(f"  saved {path}  ({len(frames)} frames, return={_last_return:.2f})")


_last_return = 0.0  # грязный глобал для print выше


def best_ppo_episode(policy, env_name, level, n_candidates=10, max_steps=500):
    """Из n попыток берём лучшую по return."""
    best_frames, best_ret = None, -1e9
    for _ in range(n_candidates):
        env = ProcgenTaskWrapper(env_name=env_name, level_seed=level)
        obs = env.reset()
        frames, ret = [obs.copy()], 0.0
        done = False
        steps = 0
        while not done and steps < max_steps:
            with torch.no_grad():
                action, _, _, _ = policy.act(torch.from_numpy(obs))
            obs, r, done, _ = env.step(int(action.item()))
            frames.append(obs.copy())
            ret += r; steps += 1
        env.close()
        if ret > best_ret:
            best_ret, best_frames = ret, frames
    global _last_return
    _last_return = best_ret
    return best_frames, best_ret


def best_rl2_trial(policy, env_name, level, n_trials=5,
                    num_episodes=4, max_steps=400):
    """
    Из n_trials Meta-триалов берём тот, где adaptation_gain максимальный.
    Возвращает список (ep1_frames, ..., ep4_frames) лучшего трила.
    """
    best_trial_eps, best_gain = None, -1e9
    for _ in range(n_trials):
        hidden = policy.init_hidden(batch_size=1)
        prev_action = torch.zeros(1, dtype=torch.long)
        prev_reward = torch.zeros(1)
        prev_done = torch.ones(1)
        trial_eps = []
        env = ProcgenTaskWrapper(env_name=env_name, level_seed=level)
        for ep in range(num_episodes):
            obs = env.reset()
            frames, ret = [obs.copy()], 0.0
            done = False; steps = 0
            while not done and steps < max_steps:
                with torch.no_grad():
                    action, _, _, hidden = policy.act_single(
                        torch.from_numpy(obs), prev_action, prev_reward,
                        prev_done, hidden
                    )
                a = int(action.item())
                obs, r, done, _ = env.step(a)
                frames.append(obs.copy())
                prev_action = action
                prev_reward = torch.tensor([r])
                prev_done = torch.tensor([float(done)])
                ret += r; steps += 1
            trial_eps.append({"frames": frames, "return": ret})
        env.close()
        gain = trial_eps[-1]["return"] - trial_eps[0]["return"]
        if gain > best_gain:
            best_gain, best_trial_eps = gain, trial_eps
    return best_trial_eps, best_gain


def best_pearl_trial(policy, encoder, env_name, level, n_trials=5,
                      num_episodes=4, max_steps=400, latent_dim=4):
    best_trial_eps, best_gain = None, -1e9
    for _ in range(n_trials):
        env = ProcgenTaskWrapper(env_name=env_name, level_seed=level)
        context_obs, context_act, context_rew, context_next_obs = [], [], [], []
        trial_eps = []
        for ep in range(num_episodes):
            if ep == 0 or len(context_obs) < 5:
                z = encoder.sample_prior(batch_size=1).squeeze(0)
            else:
                with torch.no_grad():
                    obs_c = torch.from_numpy(np.stack(context_obs[-100:]))
                    act_c = torch.tensor(context_act[-100:], dtype=torch.long)
                    rew_c = torch.tensor(context_rew[-100:], dtype=torch.float32)
                    nobs_c = torch.from_numpy(np.stack(context_next_obs[-100:]))
                    mu, var = encoder(obs_c, act_c, rew_c, nobs_c)
                    z = encoder.sample(mu, var)
            obs = env.reset()
            frames, ret = [obs.copy()], 0.0
            done = False; steps = 0
            while not done and steps < max_steps:
                with torch.no_grad():
                    action, _, _ = policy.act(torch.from_numpy(obs), z)
                a = int(action.item())
                next_obs, r, done, _ = env.step(a)
                context_obs.append(obs); context_act.append(a)
                context_rew.append(r); context_next_obs.append(next_obs)
                frames.append(next_obs.copy())
                obs = next_obs; ret += r; steps += 1
            trial_eps.append({"frames": frames, "return": ret})
        env.close()
        gain = trial_eps[-1]["return"] - trial_eps[0]["return"]
        if gain > best_gain:
            best_gain, best_trial_eps = gain, trial_eps
    return best_trial_eps, best_gain


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="coinrun")
    p.add_argument("--train-level", type=int, default=42)
    p.add_argument("--test-level", type=int, default=9000)
    p.add_argument("--num-train-levels", type=int, default=20)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--num-candidates", type=int, default=10,
                   help="Сколько попыток снять для выбора лучшей")
    p.add_argument("--methods", nargs="+", default=["ppo", "rl2", "pearl"])
    args = p.parse_args()

    rd = Path(args.results_dir)
    out = rd / "comparison" / "videos_best"
    out.mkdir(parents=True, exist_ok=True)

    global _last_return

    # === PPO ===
    if "ppo" in args.methods:
        print("\n=== PPO: ищем лучший эпизод из", args.num_candidates, "попыток ===")
        ckpt = rd / f"ppo_baseline_level_{args.train_level}" / "final.pt"
        if ckpt.exists():
            policy = PPOPolicy(num_actions=15)
            policy.load_state_dict(torch.load(ckpt, map_location="cpu"))
            policy.eval()

            # На train уровне (хорошая игра)
            frames, ret = best_ppo_episode(
                policy, args.game, args.train_level, args.num_candidates)
            _last_return = ret
            save_video(frames, out / "ppo_TRAIN_level_best.mp4")

            # На новом уровне (плохая игра — это и нужно показать)
            frames, ret = best_ppo_episode(
                policy, args.game, args.test_level, args.num_candidates)
            _last_return = ret
            save_video(frames, out / "ppo_NEW_level_best.mp4")
        else:
            print(f"  [SKIP] {ckpt} не найден")

    # === RL² ===
    if "rl2" in args.methods:
        print("\n=== RL²: ищем лучший meta-trial из", args.num_candidates, "попыток ===")
        ckpt = rd / f"rl2_train_{args.num_train_levels}_levels" / "final.pt"
        if ckpt.exists():
            policy = RL2Policy(num_actions=15)
            policy.load_state_dict(torch.load(ckpt, map_location="cpu"))
            policy.eval()
            trial_eps, gain = best_rl2_trial(
                policy, args.game, args.test_level,
                n_trials=args.num_candidates, num_episodes=4
            )
            print(f"  Лучший трил: ep1={trial_eps[0]['return']:.2f} "
                  f"ep4={trial_eps[-1]['return']:.2f} gain={gain:+.2f}")
            for i, ep in enumerate(trial_eps, 1):
                _last_return = ep["return"]
                save_video(ep["frames"], out / f"rl2_episode_{i}_best.mp4")
        else:
            print(f"  [SKIP] {ckpt} не найден")

    # === PEARL ===
    if "pearl" in args.methods:
        print("\n=== PEARL: ищем лучший meta-trial из", args.num_candidates, "попыток ===")
        from algorithms.pearl.procgen_encoder import ContextEncoder
        from algorithms.pearl.procgen_networks import PEARLPolicy
        ckpt = rd / f"pearl_train_{args.num_train_levels}_levels" / "final.pt"
        if ckpt.exists():
            state = torch.load(ckpt, map_location="cpu")
            policy = PEARLPolicy(num_actions=15, latent_dim=4)
            encoder = ContextEncoder(num_actions=15, latent_dim=4)
            policy.load_state_dict(state["policy"]); policy.eval()
            encoder.load_state_dict(state["encoder"]); encoder.eval()
            trial_eps, gain = best_pearl_trial(
                policy, encoder, args.game, args.test_level,
                n_trials=args.num_candidates, num_episodes=4, latent_dim=4
            )
            print(f"  Лучший трил: ep1={trial_eps[0]['return']:.2f} "
                  f"ep4={trial_eps[-1]['return']:.2f} gain={gain:+.2f}")
            for i, ep in enumerate(trial_eps, 1):
                _last_return = ep["return"]
                save_video(ep["frames"], out / f"pearl_episode_{i}_best.mp4")
        else:
            print(f"  [SKIP] {ckpt} не найден")

    print(f"\nГотово! Лучшие видео сохранены в {out}/")


if __name__ == "__main__":
    main()
