"""
Environment utilities for Meta-World benchmarks.

Provides a uniform interface for sampling tasks and creating environments
across ML1/ML10/ML45 benchmarks.
"""
import random
from typing import List, Tuple


class MetaWorldTaskSampler:
    """
    Wrapper around Meta-World benchmarks for Meta-RL.

    Usage:
        sampler = MetaWorldTaskSampler(benchmark="ML1", task_name="reach-v2")
        env, task_info = sampler.sample_train_task()
        # ... interact with env ...
    """

    def __init__(self, benchmark: str = "ML1", task_name: str = "reach-v2"):
        self.benchmark_name = benchmark
        self.task_name = task_name
        self._load_benchmark()

    def _load_benchmark(self):
        """Lazy-load metaworld to avoid import errors when not installed."""
        try:
            import metaworld
        except ImportError as e:
            raise ImportError(
                "Meta-World is not installed. Run:\n"
                "  pip install git+https://github.com/Farama-Foundation/Metaworld.git@master"
            ) from e

        if self.benchmark_name == "ML1":
            self.benchmark = metaworld.ML1(self.task_name)
            self.train_classes = self.benchmark.train_classes
            self.test_classes = self.benchmark.test_classes
            self.train_tasks = self.benchmark.train_tasks
            self.test_tasks = self.benchmark.test_tasks
        elif self.benchmark_name == "ML10":
            self.benchmark = metaworld.ML10()
            self.train_classes = self.benchmark.train_classes
            self.test_classes = self.benchmark.test_classes
            self.train_tasks = self.benchmark.train_tasks
            self.test_tasks = self.benchmark.test_tasks
        elif self.benchmark_name == "ML45":
            self.benchmark = metaworld.ML45()
            self.train_classes = self.benchmark.train_classes
            self.test_classes = self.benchmark.test_classes
            self.train_tasks = self.benchmark.train_tasks
            self.test_tasks = self.benchmark.test_tasks
        else:
            raise ValueError(f"Unknown benchmark: {self.benchmark_name}")

    def _make_env_from_task(self, task, classes):
        """Create an env instance from a Meta-World task object."""
        cls_name = task.env_name
        env = classes[cls_name]()
        env.set_task(task)
        return env

    def sample_train_task(self, rng: random.Random = None):
        """Sample a random training task; return (env, task_info)."""
        rng = rng or random
        task = rng.choice(self.train_tasks)
        env = self._make_env_from_task(task, self.train_classes)
        return env, {"env_name": task.env_name, "split": "train"}

    def sample_test_task(self, rng: random.Random = None):
        """Sample a random test task; return (env, task_info)."""
        rng = rng or random
        task = rng.choice(self.test_tasks)
        env = self._make_env_from_task(task, self.test_classes)
        return env, {"env_name": task.env_name, "split": "test"}

    def get_obs_action_dims(self) -> Tuple[int, int]:
        """Return (obs_dim, action_dim) for the benchmark."""
        env, _ = self.sample_train_task()
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        env.close() if hasattr(env, "close") else None
        return obs_dim, action_dim
