import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_baselines3.common.env_checker import check_env
from environment.slicing_env import SlicingEnv

env = SlicingEnv(split="train")
check_env(env, warn=True)
print("check_env passed")

obs, _ = env.reset(seed=0)
total = 0.0
for _ in range(1000):
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    total += r
    if trunc:
        break
print("random policy: reward =", round(total, 1), "| throughput =", round(info["episode"]["thr_mbps_total"], 1), "Mbps")