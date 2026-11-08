"""
Train DQN (preset splits) or PPO (continuous split) with Stable-Baselines3 and compare with the baselines.
Run from the project root:
    python scripts/train_sb3.py --algo ppo --timesteps 20000  --seeds 1     # smoke test (about a minute)
    python scripts/train_sb3.py --algo ppo --timesteps 500000 --seeds 5     # real run
    python scripts/train_sb3.py --algo dqn --timesteps 300000 --seeds 5
Training uses the TRAIN split; evaluation uses the VALIDATION split (the test split stays untouched).
"""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback

from algorithms.baselines import ALL_BASELINES
from algorithms.presets import PRESETS
from environment.config import Config
from environment.slicing_env import DiscreteSlicingEnv, LagrangianWrapper, SlicingEnv
from environment.slicing_sim import SlicingSim
from evaluation.runner import evaluate, mean_ci, run_episode

ap = argparse.ArgumentParser()
ap.add_argument("--algo", choices=["ppo", "dqn"], required=True)
ap.add_argument("--timesteps", type=int, default=500_000)
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--outage_prob", type=float, default=0.0, help="share of training episodes that contain an outage")
ap.add_argument("--capacity_bound", action="store_true", help="switch the capacity bound on (training and evaluation)")
ap.add_argument("--eval_every", type=int, default=25_000)
ap.add_argument("--train_load_scale", type=float, default=1.0, help="demand multiplier during TRAINING (use e.g. 1.6 so constraints are actually tested)")
ap.add_argument("--lagrangian", action="store_true", help="constrained RL: Lagrangian penalty on URLLC latency/loss (novelty A)")
ap.add_argument("--budget_lat", type=float, default=0.005, help="allowed mean URLLC latency-violation cost per step")
ap.add_argument("--budget_loss", type=float, default=0.005, help="allowed mean URLLC drop cost per step")
ap.add_argument("--lag_lr", type=float, default=1.0, help="dual ascent step size for lambda")
args = ap.parse_args()
print("Settings:", vars(args))

try:
    import torch
    torch.set_num_threads(1)       # small networks run faster on one thread
except ImportError:
    pass

cfg = Config()
DISCRETE = args.algo == "dqn"
EVAL_SEEDS = list(range(10))
CURVE_SEEDS = list(range(100, 105))        # fixed validation episodes used for the training curve
tag = args.algo + ("_lag" if args.lagrangian else "") + (f"_s{args.train_load_scale:g}" if args.train_load_scale != 1.0 else "") + ("_cb" if args.capacity_bound else "") + ("_out" if args.outage_prob > 0 else "")


class SB3Policy:
    """Lets the evaluation runner call a trained SB3 model like a baseline."""
    def __init__(self, model):
        self.model, self.name = model, tag

    def reset(self):
        pass

    def act(self, state):
        a, _ = self.model.predict(state["obs"], deterministic=True)
        return PRESETS[int(a)] if DISCRETE else (np.asarray(a, dtype=float) + 1.0) / 2.0


class CurveCallback(BaseCallback):
    """Every `eval_every` steps: run the current policy on fixed validation episodes and log the mean reward."""
    def __init__(self, every):
        super().__init__()
        self.every, self.next_t, self.log = every, every, []
        self.sim = SlicingSim(cfg, split="val", capacity_bound=args.capacity_bound)

    def _on_step(self):
        if self.num_timesteps >= self.next_t:
            res = [run_episode(self.sim, SB3Policy(self.model), s, outage=False) for s in CURVE_SEEDS]
            self.log.append((self.num_timesteps, float(np.mean([r["reward"] for r in res]))))
            self.next_t += self.every
        return True


def make_model(env, seed):
    if DISCRETE:
        return DQN("MlpPolicy", env, learning_rate=1e-3, buffer_size=100_000, learning_starts=5_000, batch_size=64,
                   gamma=0.95, train_freq=4, target_update_interval=1_000, exploration_fraction=0.3,
                   exploration_final_eps=0.05, policy_kwargs=dict(net_arch=[64, 64]), seed=seed, verbose=0)
    return PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2_000, batch_size=250, n_epochs=10, gamma=0.95,
               gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, policy_kwargs=dict(net_arch=[64, 64]), seed=seed, verbose=0)


models, curves = [], []
EnvCls = DiscreteSlicingEnv if DISCRETE else SlicingEnv
for sd in range(args.seeds):
    print(f"Training {args.algo.upper()} agent seed {sd} ({args.timesteps} steps) ...", flush=True)
    env = EnvCls(cfg=replace(cfg, load_scale=args.train_load_scale), split="train", outage_prob=args.outage_prob, capacity_bound=args.capacity_bound)
    if args.lagrangian:
        env = LagrangianWrapper(env, budgets=(args.budget_lat, args.budget_loss), lr=args.lag_lr)
    model = make_model(env, sd)
    cb = CurveCallback(args.eval_every)
    model.learn(total_timesteps=args.timesteps, callback=cb)
    Path("results/models").mkdir(parents=True, exist_ok=True)
    model.save(f"results/models/{tag}_seed{sd}")
    models.append(model); curves.append(cb.log)
    if args.lagrangian:
        h = np.array(env.lam_history)
        pd.DataFrame(h, columns=["lambda_latency", "lambda_loss", "cost_latency", "cost_loss"]).to_csv(
            f"results/tables/{tag}_lambda_seed{sd}.csv", index_label="episode")
        print(f"   final lambda (latency, loss): {h[-1,0]:.2f}, {h[-1,1]:.2f} | mean cost last 50 episodes: {h[-50:,2].mean():.4f}, {h[-50:,3].mean():.4f}")
    print("   curve (steps: reward):", ", ".join(f"{t//1000}k:{r:.0f}" for t, r in cb.log), flush=True)

# ---- training curve (E1)
steps = [t for t, _ in curves[0]]
cv = np.array([[r for _, r in c] for c in curves])
pd.DataFrame(cv.T, index=steps, columns=[f"seed{i}" for i in range(args.seeds)]).to_csv(f"results/tables/{tag}_training.csv", index_label="timesteps")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = cv.mean(0)
    ci = np.array([mean_ci(cv[:, i])[1] for i in range(cv.shape[1])]) if args.seeds > 1 else np.zeros_like(m)
    plt.figure(figsize=(9, 4))
    plt.plot(steps, m, label=f"{args.algo.upper()} (mean of {args.seeds} seeds)")
    plt.fill_between(steps, m - ci, m + ci, alpha=0.25, label="95% CI")
    plt.xlabel("training steps"); plt.ylabel("reward on fixed validation episodes"); plt.grid(alpha=0.3); plt.legend()
    plt.title(f"{args.algo.upper()} training curve (E1)")
    plt.tight_layout(); plt.savefig(f"results/figures/{tag}_training.png", dpi=130)
except ImportError:
    pass

# ---- evaluation on the validation split
SCEN = {"normal (1.0x)": (1.0, False), "heavy (1.6x)": (1.6, False), "outage (1.0x)": (1.0, (400, 200))}
rows = []
for scen, (scale, outage) in SCEN.items():
    sim = SlicingSim(replace(cfg, load_scale=scale), split="val", capacity_bound=args.capacity_bound)
    for cls in ALL_BASELINES:
        res = evaluate(sim, cls(), EVAL_SEEDS, outage=outage)
        rows.append({"scenario": scen, "policy": cls.name, **{k: v[0] for k, v in res.items()}})
    per_agent = [evaluate(sim, SB3Policy(m), EVAL_SEEDS, outage=outage) for m in models]
    rows.append({"scenario": scen, "policy": tag, **{k: float(np.mean([r[k][0] for r in per_agent])) for k in per_agent[0]}})
df = pd.DataFrame(rows)

q_csv = Path("results/tables/qlearning_vs_baselines.csv")           # add the Q-learning rows if they exist
if q_csv.exists() and not args.capacity_bound and args.outage_prob == 0:
    dq = pd.read_csv(q_csv)
    df = pd.concat([df, dq[dq.policy == "q_learning"]], ignore_index=True)
df.to_csv(f"results/tables/{tag}_vs_baselines.csv", index=False)

show = pd.DataFrame({"scenario": df.scenario, "policy": df.policy, "thr_Mbps": df.thr_mbps_total.round(1),
                     "embb_p95_ms": df.lat_p95_ms_embb.round(1), "urllc_p95_ms": df.lat_p95_ms_urllc.round(1),
                     "urllc_loss_%": (100 * df.loss_urllc).round(2), "embb_loss_%": (100 * df.loss_embb).round(1),
                     "SLA_viol_%": (100 * df.sla_violation_rate).round(1), "jain": df.jain_service.round(3)})
pd.set_option("display.width", 200)
for scen, g in show.groupby("scenario", sort=False):
    print(f"\n=== {scen} ===")
    print(g.drop(columns="scenario").to_string(index=False))
print(f"\nSaved results/tables/{tag}_vs_baselines.csv, results/tables/{tag}_training.csv, results/figures/{tag}_training.png")