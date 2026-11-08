"""
Tabular Q-learning with a small hand-made state discretisation (120 states) and the 8 preset splits.

Discretised state (read from the 24-number observation; per slice the 8 features are
 [queue, arrival_rate_ema, head_of_line_delay, drop_ema, prev_alloc, eta, link_up, users_frac]):
   eMBB arrival-rate level  (5 levels)   obs[1]   edges 0.15, 0.30, 0.50, 0.70
   eMBB queue level         (4 levels)   obs[0]   edges 0.05, 0.30, 1.00
   URLLC head-of-line delay (3 levels)   obs[10]  0 (empty) / up to 30% of deadline / above
   link up                  (2 levels)   obs[6]
"""
import numpy as np

from .presets import PRESETS, N_ACTIONS, DEFAULT_ACTION

ARR_EDGES = np.array([0.15, 0.30, 0.50, 0.70])
Q_EDGES = np.array([0.05, 0.30, 1.00])
N_STATES = 5 * 4 * 3 * 2


def discretise(obs):
    a = int(np.searchsorted(ARR_EDGES, obs[1]))
    q = int(np.searchsorted(Q_EDGES, obs[0]))
    h = 0 if obs[10] <= 1e-9 else (1 if obs[10] <= 0.3 else 2)
    l = int(obs[6] > 0.5)
    return ((a * 4 + q) * 3 + h) * 2 + l


class QLearningAgent:
    name = "q_learning"

    def __init__(self, alpha=0.1, gamma=0.95, seed=0):
        self.alpha, self.gamma = alpha, gamma
        self.Q = np.zeros((N_STATES, N_ACTIONS))
        self.visits = np.zeros(N_STATES)
        self.rng = np.random.default_rng(seed)

    # --- training
    def select(self, s, eps):
        if self.rng.random() < eps:
            return int(self.rng.integers(N_ACTIONS))
        return self.greedy(s)

    def greedy(self, s):
        return DEFAULT_ACTION if not self.Q[s].any() else int(np.argmax(self.Q[s]))

    def update(self, s, a, r, s2):
        self.visits[s] += 1
        target = r + self.gamma * self.Q[s2].max()          # episodes are time-limited, so always bootstrap
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    # --- policy interface used by evaluation.runner
    def reset(self):
        pass

    def act(self, state):
        return PRESETS[self.greedy(discretise(state["obs"]))]