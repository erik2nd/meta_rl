"""
Оценка всех обученных моделей и запись видео.

Что делает:
  1. Загружает чекпоинты PPO, RL², PEARL, MAML.
  2. Прогоняет каждую модель на тест-уровне (новый seed для Meta-RL,
     и НА ТОМ ЖЕ уровне на котором учили — для vanilla PPO).
  3. Сохраняет mp4 для каждой модели.
  4. Для Meta-RL методов записывает несколько эпизодов в одном триле,
     чтобы показать адаптацию.
  5. Сохраняет summary.json с returns и training steps.

Использование:
  python scripts/eval_and_record.py \
      --game coinrun --test-level 9000 \
      --results-dir results/procgen_coinrun
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
    print("[WARN] imageio не установлен. pip install imageio imageio-ffmpeg")


def save_video(frames, path, fps=15):
    if not _HAS_IMAGEIO:
        print(f"[SKIP] {path}: imageio недоступен")
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Procgen возвращает uint8 (64,64,3) — увеличим для нагляднее
    enlarged = []
    for f in frames:
        # nearest-neighbor 4x scale (64 -> 256)
        e = np.repeat(np.repeat(f, 4, axis=0), 4, axis=1)
        enlarged.append(e)
    imageio.mimsave(path, enlarged, fps=fps, codec="libx264", quality=8)
    print(f"  saved {path} ({len(frames)} frames)")


def play_ppo_episode(policy, env, max_steps=500, deterministic=False):
    """Один эпизод vanilla PPO. Возвращает frames, return, steps."""
    obs = env.reset()
    frames, ret, steps = [obs.copy()], 0.0, 0
    done = False
    while not done and steps < max_steps:
        with torch.no_grad():
            obs_t = torch.from_numpy(obs)
            action, _, _, _ = policy.act(obs_t, deterministic=deterministic)
        obs, r, done, _ = env.step(int(action.item()))
        frames.append(obs.copy())
        ret += r
        steps += 1
    return frames, ret, steps


def play_rl2_trial(policy, env, num_episodes=4, max_steps=500):
    """
    Один meta-trial RL² на одной задаче: несколько эпизодов с переносом hidden.
    Возвращает список (frames_per_ep, return_per_ep).
    """
    hidden = policy.init_hidden(batch_size=1)
    prev_action = torch.zeros(1, dtype=torch.long)
    prev_reward = torch.zeros(1)
    prev_done = torch.ones(1)

    episodes = []
    for ep in range(num_episodes):
        obs = env.reset()
        frames, ret, steps = [obs.copy()], 0.0, 0
        done = False
        while not done and steps < max_steps:
            with torch.no_grad():
                obs_t = torch.from_numpy(obs)
                action, _, _, hidden = policy.act_single(
                    obs_t, prev_action, prev_reward, prev_done, hidden
                )
            a = int(action.item())
            obs, r, done, _ = env.step(a)
            frames.append(obs.copy())
            prev_action = action
            prev_reward = torch.tensor([r])
            prev_done = torch.tensor([float(done)])
            ret += r
            steps += 1
        episodes.append({"frames": frames, "return": ret, "steps": steps})
    return episodes


def play_pearl_trial(policy, encoder, env, num_episodes=4,
                     max_steps=500, latent_dim=8):
    """
    PEARL: эпизод 1 с z из prior, далее z обновляется из накопленного контекста.
    """
    from algorithms.pearl.encoder import ContextEncoder

    episodes = []
    context_obs = []
    context_act = []
    context_rew = []
    context_next_obs = []

    for ep in range(num_episodes):
        # Семплируем z
        if ep == 0 or len(context_obs) == 0:
            z = encoder.sample_prior(batch_size=1).squeeze(0)
        else:
            with torch.no_grad():
                obs_c = torch.from_numpy(np.stack(context_obs))
                act_c = torch.tensor(context_act, dtype=torch.long)
                rew_c = torch.tensor(context_rew, dtype=torch.float32)
                next_c = torch.from_numpy(np.stack(context_next_obs))
                mu, var = encoder(obs_c, act_c, rew_c, next_c)
                z = encoder.sample(mu, var)

        obs = env.reset()
        frames, ret, steps = [obs.copy()], 0.0, 0
        done = False
        while not done and steps < max_steps:
            with torch.no_grad():
                obs_t = torch.from_numpy(obs)
                action, _, _ = policy.act(obs_t, z)
            a = int(action.item())
            next_obs, r, done, _ = env.step(a)
            # Накапливаем контекст
            context_obs.append(obs)
            context_act.append(a)
            context_rew.append(r)
            context_next_obs.append(next_obs)
            # Чтобы контекст не разбухал
            if len(context_obs) > 200:
                context_obs.pop(0); context_act.pop(0)
                context_rew.pop(0); context_next_obs.pop(0)
            frames.append(next_obs.copy())
            obs = next_obs
            ret += r
            steps += 1
        episodes.append({"frames": frames, "return": ret, "steps": steps})
    return episodes


def load_training_steps(history_path):
    if not Path(history_path).exists():
        return None
    with open(history_path) as f:
        h = json.load(f)
    return int(h["step"][-1]) if h.get("step") else None


def eval_ppo(results_dir, game, train_level, test_level, videos_dir):
    """PPO baseline: train-level (зубрит) + test-level (провал)."""
    base = Path(results_dir) / f"ppo_baseline_level_{train_level}"
    ckpt = base / "final.pt"
    if not ckpt.exists():
        print(f"[SKIP] PPO: {ckpt} не найден"); return None

    policy = PPOPolicy(num_actions=15)
    policy.load_state_dict(torch.load(ckpt, map_location="cpu"))
    policy.eval()

    # На том уровне, где обучали
    env = ProcgenTaskWrapper(env_name=game, level_seed=train_level)
    frames, ret_train, _ = play_ppo_episode(policy, env)
    env.close()
    save_video(frames, f"{videos_dir}/ppo_on_train_level.mp4")

    # На новом уровне — несколько прогонов для среднего
    test_returns = []
    for i in range(5):
        env = ProcgenTaskWrapper(env_name=game, level_seed=test_level + i)
        frames, ret, _ = play_ppo_episode(policy, env)
        env.close()
        if i == 0:
            save_video(frames, f"{videos_dir}/ppo_on_new_level.mp4")
        test_returns.append(ret)

    return {
        "train_level_return": float(ret_train),
        "test_return_mean": float(np.mean(test_returns)),
        "test_return_std": float(np.std(test_returns)),
        "training_steps": load_training_steps(base / "history.json"),
    }


def eval_rl2(results_dir, game, test_level, num_train_levels, videos_dir,
              num_episodes_in_trial=4):
    base = Path(results_dir) / f"rl2_train_{num_train_levels}_levels"
    ckpt = base / "final.pt"
    if not ckpt.exists():
        print(f"[SKIP] RL²: {ckpt} не найден"); return None

    policy = RL2Policy(num_actions=15)
    policy.load_state_dict(torch.load(ckpt, map_location="cpu"))
    policy.eval()

    env = ProcgenTaskWrapper(env_name=game, level_seed=test_level)
    episodes = play_rl2_trial(policy, env, num_episodes=num_episodes_in_trial)
    env.close()

    for i, ep in enumerate(episodes, 1):
        save_video(ep["frames"], f"{videos_dir}/rl2_episode_{i}.mp4")

    returns_per_ep = [ep["return"] for ep in episodes]
    return {
        "returns_per_episode": returns_per_ep,
        "first_episode_return": float(returns_per_ep[0]),
        "last_episode_return": float(returns_per_ep[-1]),
        "adaptation_gain": float(returns_per_ep[-1] - returns_per_ep[0]),
        "training_steps": load_training_steps(base / "history.json"),
    }


def eval_pearl(results_dir, game, test_level, num_train_levels, videos_dir,
                latent_dim=8, num_episodes_in_trial=4):
    from algorithms.pearl.procgen_encoder import ContextEncoder
    from algorithms.pearl.procgen_networks import PEARLPolicy

    base = Path(results_dir) / f"pearl_train_{num_train_levels}_levels"
    ckpt = base / "final.pt"
    if not ckpt.exists():
        print(f"[SKIP] PEARL: {ckpt} не найден"); return None

    state = torch.load(ckpt, map_location="cpu")
    policy = PEARLPolicy(num_actions=15, latent_dim=latent_dim)
    encoder = ContextEncoder(num_actions=15, latent_dim=latent_dim)
    policy.load_state_dict(state["policy"])
    encoder.load_state_dict(state["encoder"])
    policy.eval(); encoder.eval()

    env = ProcgenTaskWrapper(env_name=game, level_seed=test_level)
    episodes = play_pearl_trial(policy, encoder, env,
                                  num_episodes=num_episodes_in_trial,
                                  latent_dim=latent_dim)
    env.close()

    for i, ep in enumerate(episodes, 1):
        save_video(ep["frames"], f"{videos_dir}/pearl_episode_{i}.mp4")

    returns_per_ep = [ep["return"] for ep in episodes]
    return {
        "returns_per_episode": returns_per_ep,
        "first_episode_return": float(returns_per_ep[0]),
        "last_episode_return": float(returns_per_ep[-1]),
        "adaptation_gain": float(returns_per_ep[-1] - returns_per_ep[0]),
        "training_steps": load_training_steps(base / "history.json"),
    }


def eval_maml(results_dir, game, test_level, num_train_levels, videos_dir):
    """MAML: показываем один эпизод pre-adapt и один post-adapt."""
    base = Path(results_dir) / f"maml_train_{num_train_levels}_levels"
    ckpt = base / "final.pt"
    if not ckpt.exists():
        print(f"[SKIP] MAML: {ckpt} не найден"); return None

    policy = PPOPolicy(num_actions=15)
    policy.load_state_dict(torch.load(ckpt, map_location="cpu"))

    # Pre-adapt episode
    env = ProcgenTaskWrapper(env_name=game, level_seed=test_level)
    frames_pre, ret_pre, _ = play_ppo_episode(policy, env)
    env.close()
    save_video(frames_pre, f"{videos_dir}/maml_pre_adapt.mp4")

    # Inner-loop adaptation (один шаг градиента на rollout)
    # Используем ту же функцию что в trainer
    from algorithms.maml.procgen_trainer import collect_short_rollout
    import torch.nn.functional as F
    env = ProcgenTaskWrapper(env_name=game, level_seed=test_level)
    adapt_batch = collect_short_rollout(env, policy, steps=128)
    env.close()

    obs = torch.from_numpy(adapt_batch["obs"])
    acts = torch.from_numpy(adapt_batch["actions"])
    rets = torch.from_numpy(adapt_batch["returns"])
    new_lp, value, _ = policy.evaluate(obs, acts)
    advs = rets - value.detach()
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    loss = -(new_lp * advs).mean() + 0.5 * F.mse_loss(value, rets)
    grads = torch.autograd.grad(loss, policy.parameters())
    with torch.no_grad():
        for p, g in zip(policy.parameters(), grads):
            p.sub_(0.1 * g)

    # Post-adapt episode
    env = ProcgenTaskWrapper(env_name=game, level_seed=test_level)
    frames_post, ret_post, _ = play_ppo_episode(policy, env)
    env.close()
    save_video(frames_post, f"{videos_dir}/maml_post_adapt.mp4")

    return {
        "pre_adapt_return": float(ret_pre),
        "post_adapt_return": float(ret_post),
        "adaptation_gain": float(ret_post - ret_pre),
        "training_steps": load_training_steps(base / "history.json"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="coinrun")
    p.add_argument("--train-level", type=int, default=42)
    p.add_argument("--test-level", type=int, default=9000)
    p.add_argument("--num-train-levels", type=int, default=50)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--methods", nargs="+",
                   default=["ppo", "rl2", "pearl", "maml"])
    p.add_argument("--num-episodes-in-trial", type=int, default=4)
    args = p.parse_args()

    videos_dir = Path(args.results_dir) / "comparison" / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "game": args.game,
        "train_level_for_ppo": args.train_level,
        "test_level": args.test_level,
        "num_train_levels_for_meta": args.num_train_levels,
    }

    if "ppo" in args.methods:
        print("\n=== PPO ===")
        summary["ppo"] = eval_ppo(args.results_dir, args.game,
                                    args.train_level, args.test_level,
                                    str(videos_dir))
    if "rl2" in args.methods:
        print("\n=== RL² ===")
        summary["rl2"] = eval_rl2(args.results_dir, args.game,
                                    args.test_level, args.num_train_levels,
                                    str(videos_dir), args.num_episodes_in_trial)
    if "pearl" in args.methods:
        print("\n=== PEARL ===")
        summary["pearl"] = eval_pearl(args.results_dir, args.game,
                                        args.test_level, args.num_train_levels,
                                        str(videos_dir),
                                        num_episodes_in_trial=args.num_episodes_in_trial)
    if "maml" in args.methods:
        print("\n=== MAML ===")
        summary["maml"] = eval_maml(args.results_dir, args.game,
                                      args.test_level, args.num_train_levels,
                                      str(videos_dir))

    out_path = Path(args.results_dir) / "comparison" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {out_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
