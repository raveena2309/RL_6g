"""
Core simulator (pure numpy, no Gymnasium needed). One episode = 1000 steps of 1 ms.

Every step:
  1. new packets arrive in the three slice queues
  2. packets older than their deadline are dropped (counted as loss)
  3. the agent's bandwidth split gives each slice a capacity  b_k * eta_k * dt   (zero during an outage)
  4. (optional) capacity bound: bandwidth a slice cannot use is moved to slices that still have backlog
  5. queues are served first-in-first-out; delay of every served packet is recorded
  6. reward and next observation are computed
"""
from collections import deque
from dataclasses import replace

import numpy as np

from .channel_model import ChannelModel
from .config import Config, SLICES
from .traffic_generator import TrafficGenerator

OBS_PER_SLICE = 8
OBS_DIM = 3 * OBS_PER_SLICE


class SlicingSim:
    def __init__(self, cfg: Config = None, split: str = "train",
                 reward_weights=(1.0, 0.5, 2.0, 0.3),      # w1 throughput, w2 latency, w3 loss, w4 unfairness
                 urllc_latency_limit_s: float = 0.005,      # SLA: URLLC waiting time <= 5 ms
                 capacity_bound: bool = False,              # novelty B switch
                 outage_prob: float = 0.0):                 # chance that a random episode contains an outage
        self.cfg = cfg or Config()
        self.split = split
        self.w = reward_weights
        self.L_max = urllc_latency_limit_s
        self.capacity_bound = capacity_bound
        self.outage_prob = outage_prob
        self.traffic = TrafficGenerator(self.cfg)
        self.channel = ChannelModel(self.cfg)
        c = self.cfg
        self.peak_bits_step = np.asarray(c.peak_rate_bps) * c.step_s                  # bits per step at load 1.0
        self.peak_pkts_step = self.peak_bits_step / np.asarray(c.packet_bits)
        self.deadline = np.asarray(c.deadline_s)
        self.users_frac = np.asarray(c.users_per_slice) / float(sum(c.users_per_slice))

    # ------------------------------------------------------------------ reset
    def reset(self, seed: int = 0, split: str = None, outage=None):
        """outage: None -> random (probability outage_prob); False -> never; (start, duration) -> forced."""
        c = self.cfg
        split = split or self.split
        self.seed = seed
        self.ep = self.traffic.generate_episode(split, np.random.default_rng([seed, 1]))
        self.ch = self.channel.generate_episode(np.random.default_rng([seed, 2]))
        rng_o = np.random.default_rng([seed, 3])
        if outage is None:
            outage = (int(rng_o.integers(200, 600)), int(rng_o.integers(100, 300))) if rng_o.random() < self.outage_prob else False
        self.outage = outage if outage else (-1, 0)
        self.t = 0
        self.queues = [deque() for _ in range(3)]
        self.backlog = np.zeros(3)
        self.arr_ema = np.zeros(3)
        self.drop_ema = np.zeros(3)
        self.prev_alloc = np.ones(3) / 3
        self.last_served_bits = np.zeros(3)
        self.delay_samples = [[] for _ in range(3)]
        self.tot = {k: np.zeros(3) for k in ("served_bits", "arr_bits", "arr_pkts", "drop_pkts")}
        self.sla_steps = 0
        self.overalloc_sum = 0.0
        self.linkup_steps = 0
        self.reward_total = 0.0
        self.history = {"alloc": np.zeros((c.episode_steps, 3)), "served_bits": np.zeros((c.episode_steps, 3)),
                        "link_up": np.zeros(c.episode_steps)}
        self.hol = np.zeros(3)
        return self._obs(), {}

    # ------------------------------------------------------------------ helpers
    def link_up(self, t):
        s, d = self.outage
        return 0.0 if s <= t < s + d else 1.0

    def _normalise(self, action):
        a = np.clip(np.asarray(action, dtype=float), 0.0, None)
        s = a.sum()
        return a / s if s > 1e-9 else np.ones(3) / 3

    def get_state(self):
        """Raw (unnormalised) state, used by the classical baselines."""
        t = min(self.t, self.cfg.episode_steps - 1)
        return {"t": self.t, "backlog_bits": self.backlog.copy(), "eta": self.ch["eta"][t].copy(),
                "hol_s": self.hol.copy(), "link_up": self.link_up(t), "last_served_bits": self.last_served_bits.copy()}

    def _obs(self):
        c = self.cfg
        t = min(self.t, c.episode_steps - 1)
        q_norm = np.clip(self.backlog / (self.peak_bits_step / c.step_s * 0.1 + 1e-9), 0, 5)  # backlog in units of 100 ms of peak traffic (load 1.0)
        hol = np.clip(self.hol / self.deadline, 0, 1)
        eta = self.ch["eta"][t] / c.eta_cap
        link = np.full(3, self.link_up(t))
        feats = np.stack([q_norm, self.arr_ema, hol, self.drop_ema, self.prev_alloc, eta, link, self.users_frac], axis=1)
        return feats.reshape(-1).astype(np.float32)

    # ------------------------------------------------------------------ step
    def step(self, action):
        c, t, dt = self.cfg, self.t, self.cfg.step_s
        a = self._normalise(action)
        B = c.total_bandwidth_hz

        # 1. arrivals
        for k in range(3):
            bits, pk = self.ep["bits"][t, k], self.ep["pkts"][t, k]
            if bits > 0:
                self.queues[k].append([t, bits, pk])
                self.backlog[k] += bits
            self.tot["arr_bits"][k] += bits
            self.tot["arr_pkts"][k] += pk

        # 2. drop expired packets
        dropped = np.zeros(3)
        for k in range(3):
            q = self.queues[k]
            while q and (t - q[0][0]) * dt >= self.deadline[k]:
                ch = q.popleft()
                dropped[k] += ch[2]
                self.backlog[k] -= ch[1]
        self.backlog = np.maximum(self.backlog, 0.0)
        offered = self.backlog.copy()

        # 3./4. capacity (and optional capacity bound)
        link = self.link_up(t)
        eta = self.ch["eta"][t]
        bw = a * B
        need_hz = offered / (eta * dt)                          # bandwidth that would drain the queue completely
        overalloc = np.maximum(bw - need_hz, 0.0).sum() / B if link else 0.0
        if self.capacity_bound and link:
            excess = np.maximum(bw - need_hz, 0.0).sum()
            unmet = np.maximum(need_hz - bw, 0.0)
            if excess > 0 and unmet.sum() > 0:
                give = unmet * min(1.0, excess / unmet.sum())
                bw = np.minimum(bw, need_hz) + give
            elif excess > 0:
                bw = np.minimum(bw, need_hz)
        cap_bits = bw * eta * dt * link

        # 5. serve FIFO
        served = np.zeros(3)
        for k in range(3):
            q, cap = self.queues[k], cap_bits[k]
            while q and cap > 1e-9:
                ch = q[0]
                delay = (t - ch[0] + 1) * dt
                if ch[1] <= cap + 1e-9:
                    cap -= ch[1]; served[k] += ch[1]; self.backlog[k] -= ch[1]
                    self.delay_samples[k].append((delay, ch[2]))
                    q.popleft()
                else:
                    frac = cap / ch[1]
                    self.delay_samples[k].append((delay, ch[2] * frac))
                    ch[1] -= cap; ch[2] -= ch[2] * frac
                    served[k] += cap; self.backlog[k] -= cap; cap = 0.0
        self.backlog = np.maximum(self.backlog, 0.0)

        # 6. head-of-line waiting time, reward, bookkeeping
        self.hol = np.array([(t + 1 - self.queues[k][0][0]) * dt if self.queues[k] else 0.0 for k in range(3)])
        thr = np.mean(np.minimum(1.0, served / self.peak_bits_step))
        lat = np.mean(np.minimum(1.0, self.hol / self.deadline))
        loss = np.mean(np.minimum(1.0, dropped / self.peak_pkts_step))
        ratio = np.where(offered > 0, served / np.maximum(offered, 1e-9), 1.0)
        jain = float(ratio.sum() ** 2 / (3 * (ratio ** 2).sum() + 1e-12))
        w1, w2, w3, w4 = self.w
        reward = w1 * thr - w2 * lat - w3 * loss - w4 * (1 - jain)

        sla_viol = bool(self.hol[1] > self.L_max or dropped[1] > 0)
        cost_lat = float(np.clip((self.hol[1] - self.L_max) / (self.deadline[1] - self.L_max), 0, 1))
        cost_loss = float(min(1.0, dropped[1] / self.peak_pkts_step[1]))

        self.tot["served_bits"] += served
        self.tot["drop_pkts"] += dropped
        self.sla_steps += sla_viol
        self.overalloc_sum += overalloc
        self.linkup_steps += link
        self.reward_total += reward
        self.history["alloc"][t] = a
        self.history["served_bits"][t] = served
        self.history["link_up"][t] = link
        self.arr_ema += 0.05 * (self.ep["bits"][t] / self.peak_bits_step - self.arr_ema)
        self.drop_ema += 0.05 * (np.minimum(1.0, dropped / self.peak_pkts_step) - self.drop_ema)
        self.prev_alloc = a
        self.last_served_bits = served
        self.t += 1

        truncated = self.t >= c.episode_steps
        info = {"throughput_norm": float(thr), "latency_norm": float(lat), "loss_norm": float(loss), "jain": jain,
                "sla_violation": sla_viol, "cost_latency": cost_lat, "cost_loss": cost_loss, "link_up": link}
        if truncated:
            info["episode"] = self.summary()
        return self._obs(), float(reward), False, truncated, info

    # ------------------------------------------------------------------ episode summary
    @staticmethod
    def _wquantile(samples, q):
        if not samples:
            return float("nan")
        d = np.array([s[0] for s in samples]); w = np.array([s[1] for s in samples])
        o = np.argsort(d); d, w = d[o], w[o]
        cw = np.cumsum(w)
        return float(d[np.searchsorted(cw, q * cw[-1])])

    def summary(self):
        c = self.cfg
        T = c.episode_steps
        out = {"reward": self.reward_total}
        frac_served = self.tot["served_bits"] / np.maximum(self.tot["arr_bits"], 1e-9)
        for k, name in enumerate(SLICES):
            s = self.delay_samples[k]
            out[f"thr_mbps_{name}"] = self.tot["served_bits"][k] / (T * c.step_s) / 1e6
            out[f"lat_mean_ms_{name}"] = (1e3 * sum(d * w for d, w in s) / max(sum(w for _, w in s), 1e-9)) if s else float("nan")
            out[f"lat_p95_ms_{name}"] = 1e3 * self._wquantile(s, 0.95)
            out[f"loss_{name}"] = self.tot["drop_pkts"][k] / max(self.tot["arr_pkts"][k], 1e-9)
        out["thr_mbps_total"] = sum(out[f"thr_mbps_{n}"] for n in SLICES)
        out["sla_violation_rate"] = self.sla_steps / T
        out["jain_service"] = float(frac_served.sum() ** 2 / (3 * (frac_served ** 2).sum() + 1e-12))
        out["overalloc_frac"] = self.overalloc_sum / max(self.linkup_steps, 1)
        return out