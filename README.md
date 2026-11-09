# RL-Based Bandwidth Allocation in a Simulated 6G Network

A reproducible Python project that simulates a multi-slice 6G base station and trains reinforcement learning (RL) agents to allocate bandwidth among eMBB, URLLC and mMTC slices. The agents are benchmarked against classical schedulers with proper statistics. The project adds a capacity-aware action bound, hard URLLC SLA constraints (Lagrangian PPO) and outage-robust training.

This document is the full project specification. Build the repository to match it.

---

## 1. Problem

One base station has limited bandwidth `B_total`. Three network slices compete for it:

| Slice | Requirement | Typical traffic |
|---|---|---|
| **eMBB** | High throughput | Large bursty flows (video, downloads) |
| **URLLC** | Very low latency, near-zero loss, strict per-packet deadline | Small packets with a deadline |
| **mMTC** | Many devices, tiny sparse data | Many independent sparse sources |

Traffic volume and radio quality change every few milliseconds. Fixed rules (equal share, round-robin, proportional-fair) cannot adapt well.

**Goal:** at every time step, an RL agent observes the network state and chooses a bandwidth split across the three slices so that overall QoS beats classical schedulers, while URLLC SLAs are respected.

**Loop:** observe network, choose split, receive QoS-based reward, repeat.

---

## 2. Core idea: real traffic shape, simulated packets

RL needs an environment that reacts to its actions, so a fixed dataset cannot be used directly for training. Instead:

1. A **real dataset** (Telecom Italia, Milan) supplies the *load pattern*: how busy each slice is at each time of day.
2. A **traffic generator** converts each load value into millisecond-level packet arrivals (bursty or sparse, depending on the slice).
3. A **channel model** computes per-user SNR and capacity from distance, path loss, shadowing and blockage.
4. A **Gymnasium environment** serves the queues according to the agent's split and returns state, reward and done flag.

The real data is therefore used for realism and for the train/validation/test split, and the simulator provides the interactive environment.

---

## 3. Dataset

**Telecommunications - SMS, Call, Internet - MI** (Telecom Italia Big Data Challenge), Harvard Dataverse, DOI `10.7910/DVN/EGZHFV`.

- Milan is divided into a 100 x 100 grid (10,000 squares). Each record is one square in one 10-minute interval.
- Fields (verify against the actual files before coding): square id, time interval, country code, SMS in, SMS out, call in, call out, internet activity.
- Values are proportional to real activity (scaled by a constant for privacy), not actual counts.
- Raw files are placed by the user in `data/raw/`. **Never modify files in `data/raw/`.**

### Slice mapping (documented assumption)

| Dataset signal | Slice |
|---|---|
| Internet | eMBB |
| Calls (in + out) | URLLC |
| SMS (in + out) | mMTC |

### Preprocessing pipeline (`data_pipeline/`)

1. Load all daily files; parse timestamps to local datetime.
2. Combine in + out for SMS and for calls.
3. Sum over country codes.
4. Sum over a configurable cluster of grid squares (default: a group of busy central squares) to represent one cell. Output: one time series per slice at 10-minute resolution.
5. Split **by date, never randomly**: train (about the first 40 days), validation (next about 10 days), test (last about 12 days, which include the holiday period). Make the cut dates configurable.
6. Normalise each series to [0, 1] using the **training-set peak only**; apply the same peak to validation and test (test values may exceed 1.0).
7. Save to `data/processed/` as CSV or parquet with a small metadata file recording the peak values, cluster squares and split dates.
8. Provide an inspection script that prints columns, row counts, grid-square count, interval count, missing values and plots daily profiles per slice.

---

## 4. Simulation specification

All values below are configurable defaults, stored in `configs/default.yaml`. Each default must be documented with a comment citing its justification (3GPP TR 38.901 for channel values, literature for traffic). Do not tune parameters to favour RL.

### 4.1 Network
- 1 base station (design so that extension to 3 cells is possible).
- `B_total`: default 100 MHz (sub-6) with an option for a 1 GHz THz-style setting.
- K = 3 slices; N users distributed over slices (default 40 eMBB, 20 URLLC, 40 mMTC; scalable from about 20 to 300 for load experiments).
- Time step: configurable 1 to 10 ms (default 10 ms).
- Episode length: 1000 steps.

### 4.2 Channel (`channel_model.py`)
- User distance: uniform in a configurable range (default 50 to 500 m).
- Path loss from distance (3GPP-style; optional free-space plus molecular absorption for THz).
- Log-normal shadowing.
- Random blockage events (sharp SNR drops, with configurable probability and duration).
- Output: SNR_i(t) per user, aggregated to mean SNR per slice.
- Capacity: `C = b * log2(1 + SNR)` where `b` is the bandwidth allocated.

### 4.3 Traffic (`traffic_generator.py`)
- Mean rate for slice k at time t = normalised dataset load(t) x peak rate for slice k (peak rates configurable).
- Fine-grained arrivals around that mean:
  - eMBB: Gamma or Pareto bursts (heavy-tailed).
  - URLLC: small packets, each with a hard deadline (default 5 ms).
  - mMTC: sparse independent Poisson arrivals.
- Each episode samples a start time from the chosen split (train, validation or test) and a random seed.
- All randomness uses explicit seeds via `numpy.random.Generator`.

### 4.4 Queues and serving
- One queue per slice (packet-level or aggregated bytes with per-packet timestamps for URLLC).
- Per step, each slice transmits `min(queue, capacity x step duration)`.
- Packets exceeding their deadline are dropped and counted as loss. Bounded queue sizes cause overflow drops.
- Track per-packet waiting time to compute latency (mean and 95th percentile).

### 4.5 Outages
- Optional outage window per episode (random start and duration, link or node failure).
- During an outage the link-up flag is 0 and capacity is 0 for the affected slice(s) or cell.
- A training mode randomises outages (domain randomisation); evaluation mode can force specific outage windows.

---

## 5. MDP formulation

**State (per slice, concatenated and normalised):** queue length, arrival-rate estimate, average queue latency, packets dropped in the last window, current allocation, mean SNR, link-up flag, number of active users.

**Action:** fractions `a = (a_eMBB, a_URLLC, a_mMTC)` with `a_k >= 0` and `sum(a) <= 1`.
- Continuous agents (PPO, Lagrangian PPO): softmax output (or Dirichlet policy).
- Discrete agents (Q-learning, DQN): a small set of presets, for example 50/30/20, 30/50/20, 20/20/60, 33/33/34, 20/60/20.

**Reward (base):**
```
r_t = w1 * norm_throughput - w2 * norm_latency - w3 * packet_loss - w4 * (1 - Jain_fairness)
```
Each term must be normalised to a comparable scale. Latency and loss penalties must be smooth (they must grow with the violation), never sparse (for example "+1 only if latency is exactly zero").

**Constrained version (novelty A, Lagrangian PPO):**
```
maximise E[throughput utility]
subject to E[latency_URLLC] <= L_max   and   loss_URLLC <= epsilon
r'_t = r_t - sum_j( lambda_j * c_j(t) )
```
Each `lambda_j` is updated by dual ascent: it increases while its constraint is violated and decreases (floored at 0) when satisfied.

**Capacity bound (novelty B):** the useful allocation for slice k is clipped to what its channel can carry (derived from the Shannon limit given current SNR and demand). Implement as action masking or clipping, and count any excess bandwidth as `bandwidth_wasted`. Make this switchable by a config flag so it can be ablated.

---

## 6. Algorithms

### Baselines (non-learning, `algorithms/baselines/`)
| Name | Rule |
|---|---|
| Equal share | Split bandwidth equally |
| Static split | One fixed split, set in config |
| Round-robin | Slices take turns receiving the bandwidth |
| Proportional-fair | Priority = achievable rate / exponentially averaged past rate |
| Max-weight | Priority = queue length x channel quality |

All baselines expose the same interface as the agents: `act(observation) -> action`.

### RL agents (`algorithms/agents/`)
1. **Tabular Q-learning:** discretised state (document and justify the bucket edges), preset discrete actions, epsilon-greedy with a decaying schedule.
2. **DQN:** neural Q-function, replay buffer, target network, preset discrete actions (Stable-Baselines3 acceptable).
3. **PPO:** continuous split; main agent (Stable-Baselines3 acceptable). Normalise observations and rewards.
4. **PPO + capacity bound** (novelty B).
5. **Lagrangian PPO** (novelty A), implemented as a custom callback or wrapper that maintains dual variables and penalises constraint cost.
6. **Full model:** capacity bound + Lagrangian constraints + outage-randomised training.
7. *(Stretch)* SAC, or multi-agent PPO with centralised training and decentralised execution.

All hyperparameters (learning rate, gamma, epsilon schedule, network sizes, batch size, number of episodes, reward weights, seeds) live in config files and are logged with every run.

---

## 7. Metrics

Computed per episode by `evaluation/metrics.py`:

- Average throughput (total and per slice)
- Latency: mean and 95th percentile (per slice)
- Packet loss rate (deadline misses plus overflow)
- URLLC SLA-violation rate (fraction of steps or packets violating latency or loss limits)
- Jain's fairness index: `(sum x)^2 / (n * sum x^2)`
- Bandwidth wasted (over-allocation beyond channel capacity)
- Cumulative reward
- Episodes to convergence (training)
- Outage recovery time (time to return to a stated fraction of pre-outage throughput)

Aggregate over seeds as **mean +/- 95% confidence interval**.

---

## 8. Experiments

Every experiment uses at least **5 seeds** (10 preferred) and **identical traffic seeds across all methods**. Evaluation is on the **test split only**; the validation split is used for tuning.

| ID | Experiment | Description |
|---|---|---|
| E1 | Training curves | Reward vs episodes for Q-learning, DQN, PPO (mean +/- CI) |
| E2 | Load scaling | Performance vs number of users (about 20 to 300) |
| E3 | Traffic dynamics | Slowly vs rapidly changing demand |
| E4 | Resource scarcity | Demand exceeds available bandwidth |
| E5 | Outage | Throughput before, during and after link or node failure; recovery time |
| E6 | Generalisation | Train on early days, test on unseen days (including holidays) |
| E7 | Ablation | PPO vs +capacity bound vs +constraints vs full model |
| E8 | Sensitivity | Learning rate, gamma, reward weights |

### Sanity checks before any RL training (must pass)
- Light load: all baselines show low delay and near-zero loss.
- Heavy load: loss and delay rise for all methods.
- Equal share under heavy load hurts URLLC.
- Max-weight clears large eMBB queues but can starve small slices.
- Outage window: throughput drops to zero during the outage, then recovers.
- Same seed twice gives identical results; different seeds give different but statistically similar results.
- No single baseline dominates every metric (there must be trade-offs for RL to improve on).

---

## 9. Outputs

- Results tables (CSV) per experiment: method x metric, with mean and CI.
- Plots (PNG and PDF, saved in `results/figures/`):
  - Training reward curves with confidence bands
  - Bar charts of each method on each metric with error bars
  - Performance vs number of users
  - Throughput over time around an outage
  - Ablation table and chart
  - Bandwidth split over time for the best agent (stacked area), alongside queue lengths
- Saved trained models in `results/models/` with the config and seed used.
- Logs of every run (config snapshot, seed, git commit if available).

### Optional demo dashboard (Streamlit, `dashboard/app.py`)
Controls: number of users, total bandwidth, traffic intensity, outage toggle, algorithm choice. Displays: live bandwidth split (chart), per-slice queues, throughput, latency, loss, fairness, and a comparison of algorithms on the same traffic seed.

---

## 10. Repository structure

```text
RL_6G_Bandwidth/
├── README.md
├── requirements.txt
├── configs/
│   ├── default.yaml           network, channel, traffic, reward, SLA parameters
│   └── experiments/           per-experiment overrides (E1..E8)
├── data/
│   ├── raw/                   original dataset files (read-only)
│   └── processed/             per-slice load curves, split metadata
├── data_pipeline/
│   ├── inspect_data.py
│   └── build_load_curves.py
├── environment/
│   ├── channel_model.py
│   ├── traffic_generator.py
│   ├── queues.py
│   └── slicing_env.py         Gymnasium environment (reset, step)
├── algorithms/
│   ├── baselines/             equal_share, static, round_robin, proportional_fair, max_weight
│   └── agents/                q_learning, dqn, ppo, lagrangian_ppo, capacity_mask
├── training/
│   └── train.py               CLI: --agent, --seed, --config
├── evaluation/
│   ├── metrics.py
│   ├── run_experiments.py     runs E1..E8 across seeds
│   ├── statistics.py          mean, 95% CI
│   └── plots.py
├── dashboard/
│   └── app.py                 optional
├── results/
│   ├── figures/
│   ├── tables/
│   └── models/
└── tests/
    ├── test_env_determinism.py
    ├── test_baselines_sanity.py
    └── test_metrics.py
```

---

## 11. Tech stack

Python 3.10 to 3.12 (must run on Windows, macOS and Linux; use `pathlib` for paths). Libraries: `numpy`, `pandas`, `matplotlib`, `pyyaml`, `scipy`, `gymnasium`, `stable-baselines3`, `torch` (CPU is sufficient), optional `streamlit`, `pytest`. Pin versions in `requirements.txt`.

Develop and validate everything in the Gymnasium simulator. An ns-3 / ns3-gym / TeraSim port for packet-level validation of the final policy is an optional extension.

---

## 12. Contributions

1. An open, reproducible Gymnasium environment for multi-slice 6G bandwidth allocation, driven by real traffic patterns and THz-style channel effects.
2. A capacity-aware, constraint-based deep RL scheduler (Lagrangian PPO with a Shannon-limit action bound) that explicitly enforces URLLC SLAs.
3. An outage-robustness study using randomised failures during training.
4. A generalisation test on unseen real-traffic periods.
5. A fair benchmark against strong classical schedulers (proportional-fair, max-weight) with multiple seeds and confidence intervals.

**Stretch:** multi-agent RL (one agent per cell or slice), and multi-objective optimisation with fairness and energy per bit (Pareto front).

---

## 13. Engineering rules

- Seed everything; every run must be reproducible from its config and seed.
- All methods are evaluated on identical traffic seeds.
- Never modify `data/raw/`; compute normalisation peaks from training data only; split by time only.
- No hard-coded paths or magic numbers; parameters belong in YAML configs.
- Do not tune simulator parameters to make RL win. Report negative results honestly.
- Plots, tables and any written text must agree with each other.
- Document every assumption and cite its source in code comments and in the report.
- Include unit tests for determinism, baseline sanity and metric correctness.
- Build in this order and verify each stage before the next: data pipeline, channel and traffic, environment, baselines and sanity checks, Q-learning and DQN, PPO, novelties, experiments, plots, optional dashboard.

---

## 14. Limitations (state in the final report)

- Simulation only; a 6G-inspired model, not a deployed network.
- Dataset-to-slice mapping is an assumption; dataset values are proportional rather than actual counts and cover one city over about two months in 2013.
- Simplified channel and scheduling model.
- Results depend on reward weights and simulator parameters (tested in E8).

---

## 15. References

1. Koçoğlu et al., RL in a knowledge-defined network (KDN), arXiv 2509.26075.
2. Radosavljević et al., actor-critic queue-level bandwidth scheduling via SDN, MDPI Future Internet, 2025.
3. Ibrahim et al., Q-learning with Shannon capacity for 5G/6G, Scientific Reports, 2026.
4. Telecom Italia, "Telecommunications - SMS, Call, Internet - MI", Harvard Dataverse, doi:10.7910/DVN/EGZHFV.
5. Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd ed.
6. Schulman et al., Proximal Policy Optimization Algorithms, 2017.
7. Achiam et al., Constrained Policy Optimization, 2017.
8. Raffin et al., Stable-Baselines3, JMLR 2021.
9. 3GPP TR 38.901, channel models.

Verify each reference's exact year and venue against the original source before citing.
