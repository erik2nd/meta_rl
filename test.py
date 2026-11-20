from envs.procgen_env import ProcgenTaskWrapper
env = ProcgenTaskWrapper('coinrun', level_seed=42)
obs = env.reset()
print('OK, obs shape:', obs.shape)
env.close()