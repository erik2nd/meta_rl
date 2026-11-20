"""
Procgen environment wrapper for Meta-RL.

В Procgen 'задача' = конкретный seed уровня (start_level).
Train: семплируем seeds из [0, num_train_levels).
Test:  фиксированные seeds >= num_train_levels (guaranteed OOD).

Procgen ALWAYS возвращает картинки 64x64x3 uint8.
"""
import numpy as np
import gym

try:
    from procgen import ProcgenGym3Env
    _HAS_PROCGEN = True
except ImportError:
    _HAS_PROCGEN = False


class ProcgenTaskWrapper(gym.Env):
    """
    Один уровень Procgen как gym.Env с фиксированным seed.
    rgb наблюдение (64, 64, 3) uint8 и Discrete(15) actions.
    """
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, env_name="coinrun", level_seed=0,
                 distribution_mode="easy", num_levels=1):
        if not _HAS_PROCGEN:
            raise ImportError(
                "procgen не установлен. pip install procgen "
                "(или см. README; Procgen иногда требует ручной сборки)."
            )
        self.env_name = env_name
        self.level_seed = level_seed
        self.distribution_mode = distribution_mode
        self.num_levels = num_levels

        # gym3 env — обертка над "сырым" Procgen
        self._env = ProcgenGym3Env(
            num=1,
            env_name=env_name,
            num_levels=num_levels,
            start_level=level_seed,
            distribution_mode=distribution_mode,
            use_backgrounds=True,
            rand_seed=level_seed,
        )

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(64, 64, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.Discrete(15)
        self._last_obs = None
        self._episode_steps = 0
        self._max_steps = 500  # safety cap

    def reset(self):
        # gym3 не имеет explicit reset — делаем no-op step и читаем obs
        # На самом деле создаём env заново для гарантии чистого старта
        self._env = ProcgenGym3Env(
            num=1, env_name=self.env_name,
            num_levels=self.num_levels,
            start_level=self.level_seed,
            distribution_mode=self.distribution_mode,
            use_backgrounds=True,
            rand_seed=self.level_seed,
        )
        _, obs, _ = self._env.observe()
        self._last_obs = obs["rgb"][0]
        self._episode_steps = 0
        return self._last_obs.copy()

    def step(self, action):
        self._env.act(np.array([action], dtype=np.int32))
        reward, obs, first = self._env.observe()
        self._last_obs = obs["rgb"][0]
        self._episode_steps += 1

        # В Procgen-gym3 'first' = True на первом шаге НОВОГО эпизода
        # т.е. если first[0]==True, значит ПРЕДЫДУЩИЙ эпизод закончился
        # Используем это как сигнал done
        done = bool(first[0]) and self._episode_steps > 1
        truncated = self._episode_steps >= self._max_steps
        return (self._last_obs.copy(), float(reward[0]),
                done or truncated, {"truncated": truncated})

    def render(self, mode="rgb_array"):
        # Возвращаем последнее наблюдение в виде картинки
        if self._last_obs is None:
            return np.zeros((64, 64, 3), dtype=np.uint8)
        return self._last_obs.copy()

    def close(self):
        if hasattr(self._env, "close"):
            self._env.close()


class ProcgenTaskSampler:
    """
    Управляет 'распределением задач'.

    sample_train_task() -> int (seed из train-распределения)
    make_env(seed)      -> ProcgenTaskWrapper для одного уровня
    make_test_env(idx)  -> уровень из test-распределения (гарантированно OOD)
    """
    def __init__(self, env_name="coinrun", num_train_levels=50,
                 distribution_mode="easy", seed=0):
        self.env_name = env_name
        self.num_train_levels = num_train_levels
        self.distribution_mode = distribution_mode
        self.rng = np.random.RandomState(seed)

    def sample_train_task(self):
        return int(self.rng.randint(0, self.num_train_levels))

    def make_env(self, level_seed):
        return ProcgenTaskWrapper(
            env_name=self.env_name,
            level_seed=level_seed,
            distribution_mode=self.distribution_mode,
            num_levels=1,
        )

    def make_test_env(self, test_idx=0):
        """Test seeds начинаются с num_train_levels + 1000 (зазор для надёжности)."""
        seed = self.num_train_levels + 1000 + test_idx
        return self.make_env(seed)
