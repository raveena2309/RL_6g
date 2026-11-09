"""
Train tabular Q-learning (several agent seeds) and compare it with the classical baselines.
Run from the project root:
    python scripts/train_qlearning.py --episodes 300 --seeds 1      # quick test
    python scripts/train_qlearning.py --episodes 1500 --seeds 5     # real run
Training uses the TRAIN split; evaluation uses the VALIDATION split (test split stays untouched).
"""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from algorithms.baselines import ALL_BASELINES
from algorithms.presets import PRESETS
from algorithms.q_learning import QLearningAgent, discretise
from environment.config import Config
from environment.slicing_sim import SlicingSim
from evaluation.runner import evaluate, mean_ci, run_episode

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=1500)
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--alpha", type=float, default=0.1)
ap.add_argument("--gamma", type=float, default=0.95)
ap.add_argument("--eps_end", type=float, default=0.05)
ap.add_argument("--outage_prob", type=float, default=0.0, help="share of training episodes that contain an outage")
args = ap.parse_args()
print("Hyperparameters:", vars(args))

cfg = Config()
EVAL_SEEDS = list(range(10))


def train(agent_seed):
    agent = QLearningAgent(args.alpha, args.gamma, seed=agent_seed)
    sim = SlicingSim(cfg, split="train", outage_prob=args.outage_prob)
    curve = []
    for ep in range(args.episodes):
        eps = max(args.eps_end, 1.0 - (1.0 - args.eps_end) * ep / (0.6 * args.episodes))
        obs, _ = sim.reset(seed=100_000 * (agent_seed + 1) + ep)
        s = discretise(obs)
        done = False
        while not done:
            a = agent.select(s, eps)
            obs, r, term, trunc, info = sim.step(PRESETS[a])
            s2 = discretise(obs)
            agent.update(s, a, r, s2)
            s, done = s2, term or trunc
        curve.append(info["episode"]["reward"])
        if (ep + 1) % max(1, args.episodes // 5) == 0:
            print(f"  seed {agent_seed}: episode {ep+1}/{args.episodes}  eps={eps:.2f}  reward={np.mean(curve[-20:]):.1f}")
    return agent, curve


agents, curves = [], []
for sd in range(args.seeds):
    print(f"Training agent seed {sd} ...")
    ag, cv = train(sd)
    agents.append(ag); curves.append(cv)

Path("results/models").mkdir(parents=True, exist_ok=True)
for i, ag in enumerate(agents):
    np.save(f"results/models/qlearning_Q_seed{i}.npy", ag.Q)
curves = np.array(curves)
pd.DataFrame(curves.T, columns=[f"seed{i}" for i in range(args.seeds)]).to_csv("results/tables/qlearning_training.csv", index_label="episode")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w = max(1, args.episodes // 30)
    sm = np.array([np.convolve(c, np.ones(w) / w, mode="valid") for c in curves])
    m = sm.mean(0)
    ci = np.array([mean_ci(sm[:, i])[1] for i in range(sm.shape[1])]) if args.seeds > 1 else np.zeros_like(m)
    plt.figure(figsize=(9, 4))
    plt.plot(m, label=f"Q-learning (mean of {args.seeds} seeds)")
    plt.fill_between(range(len(m)), m - ci, m + ci, alpha=0.25, label="95% CI")
    plt.xlabel("training episode"); plt.ylabel("episode reward (smoothed)"); plt.grid(alpha=0.3); plt.legend()
    plt.title("Q-learning training curve (E1)")
    plt.tight_layout(); plt.savefig("results/figures/qlearning_training.png", dpi=130)
except ImportError:
    pass

# ---------------- evaluation on the validation split
SCEN = {"normal (1.0x)": (1.0, False), "heavy (1.6x)": (1.6, False), "outage (1.0x)": (1.0, (400, 200))}
rows = []
for scen, (scale, outage) in SCEN.items():
    sim = SlicingSim(replace(cfg, load_scale=scale), split="val")
    for cls in ALL_BASELINES:
        res = evaluate(sim, cls(), EVAL_SEEDS, outage=outage)
        rows.append({"scenario": scen, "policy": cls.name, **{k: v[0] for k, v in res.items()}})
    per_agent = [evaluate(sim, ag, EVAL_SEEDS, outage=outage) for ag in agents]
    rows.append({"scenario": scen, "policy": "q_learning",
                 **{k: float(np.mean([r[k][0] for r in per_agent])) for k in per_agent[0]}})
df = pd.DataFrame(rows)
df.to_csv("results/tables/qlearning_vs_baselines.csv", index=False)
show = pd.DataFrame({"scenario": df.scenario, "policy": df.policy, "thr_Mbps": df.thr_mbps_total.round(1),
                     "embb_p95_ms": df.lat_p95_ms_embb.round(1), "urllc_p95_ms": df.lat_p95_ms_urllc.round(1),
                     "urllc_loss_%": (100 * df.loss_urllc).round(2), "embb_loss_%": (100 * df.loss_embb).round(1),
                     "SLA_viol_%": (100 * df.sla_violation_rate).round(1), "jain": df.jain_service.round(3)})
pd.set_option("display.width", 200)
for scen, g in show.groupby("scenario", sort=False):
    print(f"\n=== {scen} ===")
    print(g.drop(columns="scenario").to_string(index=False))
print("\nSaved results/tables/qlearning_vs_baselines.csv and results/figures/qlearning_training.png")
print("States visited:", int((agents[0].visits > 0).sum()), "of", len(agents[0].visits))