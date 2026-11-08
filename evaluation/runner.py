"""Run policies on the simulator with identical seeds, and aggregate mean +/- 95% CI."""
import numpy as np
from scipy import stats


def run_episode(sim, policy, seed, **reset_kwargs):
    """Run one 1000-step episode of `policy` on `sim`; return the episode summary dict."""
    obs, _ = sim.reset(seed=seed, **reset_kwargs)
    policy.reset()
    done = False
    info = {}
    while not done:
        state = sim.get_state()
        state["obs"] = obs
        action = policy.act(state)
        obs, r, term, trunc, info = sim.step(action)
        done = term or trunc
    return info["episode"]


def mean_ci(values, conf=0.95):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return float(v.mean()) if len(v) else float("nan"), float("nan")
    half = stats.t.ppf(0.5 + conf / 2, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean()), float(half)


def evaluate(sim, policy, seeds, **reset_kwargs):
    """Run `policy` on every seed; return {metric: (mean, ci_halfwidth)}."""
    runs = [run_episode(sim, policy, s, **reset_kwargs) for s in seeds]
    return {k: mean_ci([r[k] for r in runs]) for k in runs[0]}