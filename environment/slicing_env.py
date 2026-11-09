"""
Gymnasium wrappers around SlicingSim, used by Stable-Baselines3.
Requires:  pip install gymnasium

SlicingEnv         continuous action in [-1, 1]^3 (mapped to [0, 1], then normalised to a bandwidth split) -> PPO
DiscreteSlicingEnv discrete action = index of one of the 8 preset splits                                -> DQN
Observation: 24 numbers (8 per slice).
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from algorithms.presets import PRESETS, N_ACTIONS
from .slicing_sim import SlicingSim, OBS_DIM


class SlicingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, **sim_kwargs):
        super().__init__()
        self.sim = SlicingSim(**sim_kwargs)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if seed is None:                       # training: a new traffic seed every episode
            seed = int(self.np_random.integers(0, 2**31 - 1))
        return self.sim.reset(seed=seed, split=options.get("split"), outage=options.get("outage"))

    def step(self, action):
        a = (np.asarray(action, dtype=float) + 1.0) / 2.0     # [-1,1] -> [0,1]; the simulator normalises to sum 1
        return self.sim.step(a)


class DiscreteSlicingEnv(SlicingEnv):
    def __init__(self, **sim_kwargs):
        super().__init__(**sim_kwargs)
        self.action_space = spaces.Discrete(N_ACTIONS)

    def step(self, action):
        return self.sim.step(PRESETS[int(action)])


class LagrangianWrapper(gym.Wrapper):
    """
    Constrained RL (novelty A). Adds a penalty  lambda_j * cost_j  to the reward and raises lambda_j after every
    episode in which the average URLLC cost exceeded its budget (dual ascent); lambda_j shrinks (down to 0) when it did not.
        cost 0: URLLC head-of-line delay above the 5 ms limit (0..1)
        cost 1: URLLC packets dropped (0..1)
    budgets are the allowed AVERAGE cost per step (assumptions, state them in the report).
    """
    def __init__(self, env, budgets=(0.005, 0.005), lr=1.0, lam_max=10.0):
        super().__init__(env)
        self.budgets = np.asarray(budgets, dtype=float)
        self.lr, self.lam_max = lr, lam_max
        self.lam = np.zeros(2)
        self.lam_history = []                 # (lambda_0, lambda_1, mean_cost_0, mean_cost_1) per finished episode
        self._cost, self._n = np.zeros(2), 0

    def reset(self, **kwargs):
        self._cost, self._n = np.zeros(2), 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        c = np.array([info["cost_latency"], info["cost_loss"]], dtype=float) * info["link_up"]   # no cost while the link is down (uncontrollable)
        self._cost += c
        self._n += 1
        r_pen = float(r - self.lam @ c)
        if term or trunc:
            mean_c = self._cost / max(self._n, 1)
            self.lam = np.clip(self.lam + self.lr * (mean_c - self.budgets), 0.0, self.lam_max)
            self.lam_history.append((*self.lam, *mean_c))
        return obs, r_pen, term, trunc, info