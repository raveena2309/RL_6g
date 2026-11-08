"""
Run the classical baselines in the simulator (no RL yet) to check that the environment behaves sensibly.
Run from the project root:   python scripts/run_baselines.py
Uses the VALIDATION split (the test split is kept untouched for final results).
"""
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from algorithms.baselines import ALL_BASELINES
from environment.config import Config
from environment.slicing_sim import SlicingSim
from evaluation.runner import evaluate, run_episode

SEEDS = list(range(10))
SCENARIOS = {                       # name: (load_scale, outage)
    "light (0.3x)":  (0.3, False),
    "normal (1.0x)": (1.0, False),
    "heavy (1.6x)":  (1.6, False),
    "outage (1.0x)": (1.0, (400, 200)),
}
base_cfg = Config()

# determinism check
sim = SlicingSim(base_cfg, split="val")
a = run_episode(sim, ALL_BASELINES[0](), seed=5, outage=False)
b = run_episode(sim, ALL_BASELINES[0](), seed=5, outage=False)
print("Determinism (same seed, same result):", "PASS" if a == b else "FAIL", "\n")

rows = []
for scen, (scale, outage) in SCENARIOS.items():
    sim = SlicingSim(replace(base_cfg, load_scale=scale), split="val")
    for cls in ALL_BASELINES:
        res = evaluate(sim, cls(), SEEDS, outage=outage)
        row = {"scenario": scen, "policy": cls.name}
        for k, (m, ci) in res.items():
            row[k] = m
            row[k + "_ci"] = ci
        rows.append(row)

df = pd.DataFrame(rows)
Path("results/tables").mkdir(parents=True, exist_ok=True)
df.to_csv("results/tables/baselines_sanity.csv", index=False)

show = pd.DataFrame({
    "scenario": df.scenario, "policy": df.policy,
    "thr_Mbps": df.thr_mbps_total.round(1),
    "embb_Mbps": df.thr_mbps_embb.round(1),
    "urllc_p95_ms": df.lat_p95_ms_urllc.round(1),
    "urllc_loss_%": (100 * df.loss_urllc).round(2),
    "embb_loss_%": (100 * df.loss_embb).round(1),
    "SLA_viol_%": (100 * df.sla_violation_rate).round(1),
    "jain": df.jain_service.round(3),
    "overalloc_%": (100 * df.overalloc_frac).round(1),
})
pd.set_option("display.width", 200)
for scen, g in show.groupby("scenario", sort=False):
    print(f"=== {scen} ===")
    print(g.drop(columns="scenario").to_string(index=False), "\n")
print("Saved results/tables/baselines_sanity.csv (means and 95% CI half-widths over", len(SEEDS), "seeds)")