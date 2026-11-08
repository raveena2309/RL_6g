"""
Classical (non-learning) schedulers. Each one maps the raw simulator state to a bandwidth split (3 fractions).
    policy.reset()          called at the start of every episode
    policy.act(state) ->    array of 3 non-negative numbers (normalised to sum 1 by the simulator)
"""
import numpy as np


class Baseline:
    name = "base"

    def reset(self):
        pass

    def act(self, state):
        raise NotImplementedError


class EqualShare(Baseline):
    name = "equal_share"

    def act(self, state):
        return np.ones(3) / 3


class StaticSplit(Baseline):
    """One fixed split chosen by an operator (default 70/20/10 ~ roughly follows average demand)."""
    name = "static"

    def __init__(self, weights=(0.7, 0.2, 0.1)):
        self.w = np.asarray(weights, dtype=float)

    def act(self, state):
        return self.w


class RoundRobin(Baseline):
    """Time-domain round-robin: the whole band goes to one slice per step, in turn."""
    name = "round_robin"

    def act(self, state):
        a = np.zeros(3)
        a[state["t"] % 3] = 1.0
        return a


class ProportionalFair(Baseline):
    """Metric_k = achievable rate / average served rate; only slices with backlog compete."""
    name = "prop_fair"

    def __init__(self, tc_steps=100):
        self.tc = tc_steps
        self.R = None

    def reset(self):
        self.R = None

    def act(self, state):
        served_rate = state["last_served_bits"]
        self.R = served_rate + 1.0 if self.R is None else (1 - 1 / self.tc) * self.R + (1 / self.tc) * served_rate
        m = state["eta"] / np.maximum(self.R, 1.0)
        m = np.where(state["backlog_bits"] > 0, m, 0.0)
        return m if m.sum() > 0 else np.ones(3) / 3


class MaxWeight(Baseline):
    """Weight_k = backlog_k * achievable rate (shares proportional to the weights)."""
    name = "max_weight"

    def act(self, state):
        w = state["backlog_bits"] * state["eta"]
        return w if w.sum() > 0 else np.ones(3) / 3


ALL_BASELINES = [EqualShare, StaticSplit, RoundRobin, ProportionalFair, MaxWeight]  