"""
Final evaluation on the TEST split (never used for training or tuning).
Evaluates the baselines and every saved agent group with identical traffic, mean +/- 95% CI over episodes.

    python scripts/final_eval.py --quick       # 5 episodes, 3 scenarios: smoke test (a few minutes)
    python scripts/final_eval.py               # 20 episodes, 5 scenarios (about 30-60 minutes)

How the numbers are built: every method sees the same test episodes (same traffic and channel seeds).
For an RL group (5 trained agents) the value of one episode is the MEAN over the agents, so the CI shows variation
across test episodes, with agent-seed variation averaged inside each episode.
Outputs: results/tables/final_test_results.csv, final_paired_vs_static.csv and figures in results/figures/final_*.png
"""
import argparse
import glob
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd

from algorithms.baselines import EqualShare, MaxWeight, ProportionalFair, RoundRobin, StaticSplit
from algorithms.presets import PRESETS
from algorithms.q_learning import QLearningAgent
from environment.config import Config
from environment.slicing_sim import SlicingSim
from evaluation.runner import mean_ci, run_episode
from dataclasses import replace

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=20)
ap.add_argument("--quick", action="store_true")
ap.add_argument("--groups", nargs="*", default=None, help="agent groups to include (default: the core set)")
args = ap.parse_args()

cfg = Config()
N = 5 if args.quick else args.seeds
SEEDS = list(range(1000, 1000 + N))
SCEN = {"light (0.5x)": (0.5, False), "normal (1.0x)": (1.0, False), "heavy (1.6x)": (1.6, False),
        "scarce (2.2x)": (2.2, False), "outage (1.0x)": (1.0, (400, 200))}
if args.quick:
    SCEN = {k: SCEN[k] for k in ("normal (1.0x)", "heavy (1.6x)", "outage (1.0x)")}

# name -> (kind, evaluated with capacity bound?)
CORE = {"q_learning": ("q", False), "dqn_s1.6": ("dqn", False), "ppo_s1.6": ("ppo", False),
        "dqn_s1.6_weakw": ("dqn", False), "dqn_lag_s1.6_weakw": ("dqn", False), "dqn_lag_s1.6_tight": ("dqn", False),
        "ppo_s1.6_cb": ("ppo", True), "ppo_s1.6_out": ("ppo", False)}
EXTRA = {"dqn": ("dqn", False), "ppo": ("ppo", False), "dqn_lag_s1.6": ("dqn", False), "ppo_lag_s1.6": ("ppo", False),
         "ppo_lag_s1.6_cb": ("ppo", True), "ppo_lag_s1.6_cb_out": ("ppo", True)}
ALLGROUPS = {**CORE, **EXTRA}
wanted = args.groups if args.groups else list(CORE)

METRICS = ["thr_mbps_total", "lat_p95_ms_embb", "lat_p95_ms_urllc", "loss_embb", "loss_urllc",
           "sla_violation_rate", "jain_service", "overalloc_frac", "reward"]


class SB3Wrap:
    def __init__(self, model, discrete, name):
        self.model, self.discrete, self.name = model, discrete, name

    def reset(self):
        pass

    def act(self, state):
        a, _ = self.model.predict(state["obs"], deterministic=True)
        return PRESETS[int(a)] if self.discrete else (np.asarray(a, dtype=float) + 1.0) / 2.0


def load_group(tag, kind):
    pols = []
    if kind == "q":
        for i in range(10):
            f = Path(f"results/models/qlearning_Q_seed{i}.npy")
            if f.exists():
                a = QLearningAgent(); a.Q = np.load(f); pols.append(a)
    else:
        from stable_baselines3 import DQN, PPO
        cls = DQN if kind == "dqn" else PPO
        for i in range(10):
            # SB3 only adds ".zip" when the name has no dot, so tags like "dqn_s1.6" were saved WITHOUT an extension
            for f in (Path(f"results/models/{tag}_seed{i}.zip"), Path(f"results/models/{tag}_seed{i}")):
                if f.is_file():
                    pols.append(SB3Wrap(cls.load(str(f), device="cpu"), kind == "dqn", tag))
                    break
    return pols


# ---- build the list of things to evaluate: (name, capacity_bound, [policies])
entries = [(c.name, False, [c()]) for c in (EqualShare, StaticSplit, RoundRobin, ProportionalFair, MaxWeight)]
entries += [(c.name + "+cb", True, [c()]) for c in (EqualShare, StaticSplit, MaxWeight)]
for tag in wanted:
    if tag not in ALLGROUPS:
        print("unknown group", tag); continue
    kind, bound = ALLGROUPS[tag]
    pols = load_group(tag, kind)
    if pols:
        entries.append((tag, bound, pols)); print(f"loaded {tag}: {len(pols)} agents")
    else:
        print(f"skipping {tag}: no saved models found")

# ---- evaluation
raw = {}                                   # (scenario, name) -> {metric: array over seeds}
for scen, (scale, outage) in SCEN.items():
    sims = {b: SlicingSim(replace(cfg, load_scale=scale), split="test", capacity_bound=b) for b in (False, True)}
    for name, bound, pols in entries:
        per_seed = []
        for s in SEEDS:
            runs = [run_episode(sims[bound], p, s, outage=outage) for p in pols]
            per_seed.append({m: float(np.nanmean([r[m] for r in runs])) for m in METRICS})
        raw[(scen, name)] = {m: np.array([d[m] for d in per_seed]) for m in METRICS}
        print(f"  done: {scen:14s} {name}", flush=True)

rows = []
for (scen, name), d in raw.items():
    row = {"scenario": scen, "policy": name, "episodes": N}
    for m, v in d.items():
        row[m], row[m + "_ci"] = mean_ci(v)
    rows.append(row)
df = pd.DataFrame(rows)
Path("results/tables").mkdir(parents=True, exist_ok=True)
df.to_csv("results/tables/final_test_results.csv", index=False)

# ---- paired differences vs static (same episodes, so differences are meaningful)
prow = []
for (scen, name), d in raw.items():
    if name == "static":
        continue
    base = raw[(scen, "static")]
    r = {"scenario": scen, "policy": name}
    for m in ("lat_p95_ms_embb", "sla_violation_rate", "thr_mbps_total"):
        mu, ci = mean_ci(d[m] - base[m])
        r[m + "_diff"], r[m + "_ci"] = mu, ci
        r[m + "_significant"] = bool(not np.isnan(ci) and abs(mu) > ci)
    prow.append(r)
pd.DataFrame(prow).to_csv("results/tables/final_paired_vs_static.csv", index=False)

# ---- print
def fmt(m, k, scale=1.0, nd=1):
    return f"{m*scale:.{nd}f} +/- {k*scale:.{nd}f}"

for scen in SCEN:
    g = df[df.scenario == scen]
    print(f"\n=== {scen}  (test split, {N} episodes, mean +/- 95% CI) ===")
    out = pd.DataFrame({"policy": g.policy,
                        "thr_Mbps": [fmt(a, b) for a, b in zip(g.thr_mbps_total, g.thr_mbps_total_ci)],
                        "embb_p95_ms": [fmt(a, b) for a, b in zip(g.lat_p95_ms_embb, g.lat_p95_ms_embb_ci)],
                        "urllc_loss_%": [fmt(a, b, 100, 2) for a, b in zip(g.loss_urllc, g.loss_urllc_ci)],
                        "SLA_viol_%": [fmt(a, b, 100, 2) for a, b in zip(g.sla_violation_rate, g.sla_violation_rate_ci)]})
    print(out.to_string(index=False))
print("\nPaired difference vs static (negative eMBB latency = better); 'significant' means the 95% CI excludes 0:")
pv = pd.DataFrame(prow)
pv = pv[pv.scenario.isin([s for s in SCEN if "heavy" in s or "scarce" in s])]
print(pv[["scenario", "policy", "lat_p95_ms_embb_diff", "lat_p95_ms_embb_significant", "sla_violation_rate_diff", "sla_violation_rate_significant"]].round(3).to_string(index=False))
print("\nSaved results/tables/final_test_results.csv and final_paired_vs_static.csv")

# ---- figures
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    names = [n for n in ("static", "max_weight", "static+cb", "q_learning", "dqn_s1.6", "ppo_s1.6", "ppo_s1.6_cb") if any(e[0] == n for e in entries)]
    scales = {s: v[0] for s, v in SCEN.items() if not v[1]}

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for n in names:
        sub = df[(df.policy == n) & df.scenario.isin(scales)].copy()
        sub["x"] = sub.scenario.map(scales); sub = sub.sort_values("x")
        ax[0].errorbar(sub.x, sub.lat_p95_ms_embb, sub.lat_p95_ms_embb_ci, marker="o", capsize=3, label=n)
        ax[1].errorbar(sub.x, 100 * sub.sla_violation_rate, 100 * sub.sla_violation_rate_ci, marker="o", capsize=3, label=n)
    ax[0].set_yscale("log"); ax[0].set_ylabel("eMBB p95 latency (ms, log)"); ax[1].set_ylabel("URLLC SLA violations (% of steps)")
    for a in ax: a.set_xlabel("load multiplier"); a.grid(alpha=0.3)
    ax[0].legend(fontsize=8); fig.suptitle("Load scaling on the test split (E2/E4)"); fig.tight_layout()
    fig.savefig("results/figures/final_load_scaling.png", dpi=130)

    out_sc = [s for s, v in SCEN.items() if v[1]]
    if out_sc:
        fig, a = plt.subplots(figsize=(10, 4))
        for n, bound, pols in entries:
            if n in names:
                sim = SlicingSim(cfg, split="test", capacity_bound=bound)
                run_episode(sim, pols[0], SEEDS[0], outage=SCEN[out_sc[0]][1])
                thr = sim.history["served_bits"].sum(axis=1) / cfg.step_s / 1e6
                a.plot(np.convolve(thr, np.ones(25) / 25, mode="same"), label=n, lw=1)
        s0, d0 = SCEN[out_sc[0]][1]
        a.axvspan(s0, s0 + d0, color="red", alpha=0.1, label="outage")
        a.set_xlabel("step (1 ms)"); a.set_ylabel("served throughput (Mbps, 25 ms avg)"); a.legend(fontsize=8); a.grid(alpha=0.3)
        a.set_title("Throughput around an outage (E5)"); fig.tight_layout(); fig.savefig("results/figures/final_outage.png", dpi=130)

    lam = {}
    for f in sorted(glob.glob("results/tables/*_lambda_seed*.csv")):
        lam.setdefault(Path(f).name.split("_lambda_seed")[0], []).append(pd.read_csv(f)["lambda_latency"].to_numpy())
    if lam:
        fig, a = plt.subplots(figsize=(9, 4))
        for k, v in lam.items():
            L = min(map(len, v)); a.plot(np.mean([x[:L] for x in v], axis=0), label=k)
        a.set_xlabel("training episode"); a.set_ylabel("mean lambda (latency)"); a.legend(fontsize=8); a.grid(alpha=0.3)
        a.set_title("Lagrange multiplier during training (mean of seeds)"); fig.tight_layout(); fig.savefig("results/figures/final_lambda.png", dpi=130)

    fig, a = plt.subplots(figsize=(10, 4))
    for tag in [t for t in wanted if t in ALLGROUPS and ALLGROUPS[t][0] != "q"]:
        f = Path(f"results/tables/{tag}_training.csv")
        if f.exists():
            t = pd.read_csv(f, index_col=0)
            a.plot(t.index, t.mean(axis=1), label=tag)
    a.set_xlabel("training steps"); a.set_ylabel("reward on fixed validation episodes"); a.legend(fontsize=7); a.grid(alpha=0.3)
    a.set_title("Learning curves (E1)"); fig.tight_layout(); fig.savefig("results/figures/final_training_curves.png", dpi=130)
    print("Saved figures: results/figures/final_*.png")
except Exception as e:
    print("Plotting failed:", repr(e))