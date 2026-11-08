"""
Traffic generator: turns the (real or synthetic) 10-minute load curve into 1 ms packet arrivals.

Load curve file: data/processed/load_curves.csv  (columns: embb, urllc, mmtc normalised to 0-1, plus 'split').
Each episode picks a random window of `window_hours` inside the chosen split and time-compresses it
onto the episode, so demand rises and falls inside one episode (the agent must adapt).

Arrival models per slice:
  eMBB  - Gamma-distributed bits per step, modulated by a slowly varying burst state (bursty, heavy-ish tail)
  URLLC - Poisson number of small packets per step
  mMTC  - Poisson number of tiny packets per step (sparse, many independent sources)
"""
import numpy as np
import pandas as pd

from .config import Config, SLICES


class TrafficGenerator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        df = pd.read_csv(cfg.load_csv, parse_dates=["timestamp"])
        self.splits = {s: g.reset_index(drop=True) for s, g in df.groupby("split")}

    def generate_episode(self, split: str, rng: np.random.Generator, start_idx: int = None):
        c = self.cfg
        T = c.episode_steps
        df = self.splits[split]
        n_win = int(c.window_hours * 60 / c.interval_min)
        if start_idx is None:
            start_idx = int(rng.integers(0, len(df) - n_win + 1))
        win = df.iloc[start_idx:start_idx + n_win][list(SLICES)].to_numpy()

        x_src = np.arange(n_win)
        x_dst = np.linspace(0, n_win - 1, T)
        load = np.stack([np.interp(x_dst, x_src, win[:, k]) for k in range(3)], axis=1)   # [T,3] in 0..~1

        mean_bits = load * np.asarray(c.peak_rate_bps) * c.load_scale * c.step_s            # [T,3] bits per step
        pkt_bits = np.asarray(c.packet_bits, dtype=float)

        bits = np.zeros((T, 3))
        pkts = np.zeros((T, 3))

        # eMBB: slow burst state (log-normal AR(1), mean 1) times Gamma jitter
        phi = np.exp(-c.step_s / c.embb_burst_tau_s)
        sig = c.embb_burst_sigma
        x = np.zeros(T)
        e = rng.normal(0.0, sig * np.sqrt(1 - phi ** 2), T)
        for t in range(1, T):
            x[t] = phi * x[t - 1] + e[t]
        burst = np.exp(x - sig ** 2 / 2)
        k = c.embb_gamma_shape
        bits[:, 0] = rng.gamma(k, mean_bits[:, 0] * burst / k)
        pkts[:, 0] = bits[:, 0] / pkt_bits[0]

        # URLLC and mMTC: Poisson packet counts
        for j in (1, 2):
            n = rng.poisson(mean_bits[:, j] / pkt_bits[j])
            pkts[:, j] = n
            bits[:, j] = n * pkt_bits[j]

        return {"bits": bits, "pkts": pkts, "load": load, "start_idx": start_idx}