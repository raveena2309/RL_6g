"""Preset bandwidth splits (eMBB, URLLC, mMTC) for the discrete-action agents (Q-learning, DQN)."""
import numpy as np

PRESETS = np.array([
    [1 / 3, 1 / 3, 1 / 3],   # 0 equal
    [0.50, 0.30, 0.20],      # 1
    [0.60, 0.30, 0.10],      # 2
    [0.70, 0.20, 0.10],      # 3 same as the static baseline
    [0.80, 0.15, 0.05],      # 4
    [0.85, 0.10, 0.05],      # 5
    [0.90, 0.08, 0.02],      # 6 eMBB-heavy
    [0.40, 0.50, 0.10],      # 7 URLLC-heavy (e.g. to flush URLLC backlog)
])
N_ACTIONS = len(PRESETS)
DEFAULT_ACTION = 3           # used by Q-learning in states it has never visited