"""
Sanity checks for the traffic generator and channel model.
Run from the project root:   python scripts/check_traffic_channel.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from environment.config import Config, SLICES
from environment.channel_model import ChannelModel, noise_dbm
from environment.traffic_generator import TrafficGenerator

cfg = Config()
tg, ch = TrafficGenerator(cfg), ChannelModel(cfg)
ok = True


def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(("PASS  " if cond else "FAIL  ") + name + (f"   {extra}" if extra else ""))


# 1. determinism: same seed -> identical episode
a = tg.generate_episode("train", np.random.default_rng(7))
b = tg.generate_episode("train", np.random.default_rng(7))
check("traffic is deterministic for the same seed", np.array_equal(a["bits"], b["bits"]))
ca, cb = ch.generate_episode(np.random.default_rng(7)), ch.generate_episode(np.random.default_rng(7))
check("channel is deterministic for the same seed", np.array_equal(ca["eta"], cb["eta"]))
d = tg.generate_episode("train", np.random.default_rng(8))
check("different seeds give different traffic", not np.array_equal(a["bits"], d["bits"]))

# 2. long-run mean arrival rate matches the target (load x peak rate)
rel = []
for k in range(3):
    got = target = 0.0
    for s in range(60):
        ep = tg.generate_episode("train", np.random.default_rng(1000 + s))
        got += ep["bits"][:, k].sum()
        target += (ep["load"][:, k] * cfg.peak_rate_bps[k] * cfg.load_scale * cfg.step_s).sum()
    rel.append(got / target - 1)
check("mean arrival rate within 3% of target (all slices)", max(abs(r) for r in rel) < 0.03,
      "errors: " + ", ".join(f"{SLICES[k]} {rel[k]*100:+.2f}%" for k in range(3)))

# 3. URLLC / mMTC arrivals are whole packets
check("URLLC bits are whole packets", np.allclose(a["bits"][:, 1] % cfg.packet_bits[1], 0))

# 4. channel sanity
etas, snrs, blk = [], [], []
for s in range(40):
    c = ch.generate_episode(np.random.default_rng(2000 + s))
    etas.append(c["eta"].mean(axis=0)); snrs.append(c["snr_db"].mean(axis=0)); blk.append(c["blocked_fraction"])
etas, snrs = np.array(etas), np.array(snrs)
print(f"\nNoise power: {noise_dbm(cfg):.1f} dBm over {cfg.total_bandwidth_hz/1e6:.0f} MHz")
print("Mean spectral efficiency per slice (bit/s/Hz):", dict(zip(SLICES, etas.mean(axis=0).round(2))))
print("Mean SNR per slice (dB):                      ", dict(zip(SLICES, snrs.mean(axis=0).round(1))))
print(f"Mean blocked fraction: {np.mean(blk)*100:.1f}%")
check("spectral efficiency between 0 and cap", etas.min() > 0 and etas.max() <= cfg.eta_cap)
check("blocked fraction is a few percent", 0.01 < np.mean(blk) < 0.15)

# 5. is there real contention? compare peak demand with the capacity of the whole band
eta_mean = etas.mean(axis=0)
need_mhz = np.array(cfg.peak_rate_bps) / eta_mean / 1e6
print("\nBandwidth needed at peak load (MHz):", dict(zip(SLICES, need_mhz.round(1))), f"| total needed {need_mhz.sum():.0f} of {cfg.total_bandwidth_hz/1e6:.0f} MHz")
mean_load = np.array([tg.splits['train'][s].mean() for s in SLICES])
need_avg = mean_load * np.array(cfg.peak_rate_bps) / eta_mean / 1e6
print("Bandwidth needed at average load (MHz):", dict(zip(SLICES, need_avg.round(1))), f"| total needed {need_avg.sum():.0f}")

# 6. plot one episode
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = tg.generate_episode("test", np.random.default_rng(3))
    chn = ch.generate_episode(np.random.default_rng(3))
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    w = 50
    for k, lab in enumerate(["eMBB (Mbps)", "URLLC (Mbps)", "mMTC (Mbps)"]):
        rate = np.convolve(ep["bits"][:, k], np.ones(w) / w, mode="same") / cfg.step_s / 1e6
        ax[k].plot(rate, lw=0.9, label="arrival rate (50 ms avg)")
        ax[k].plot(ep["load"][:, k] * cfg.peak_rate_bps[k] / 1e6, "r--", lw=1, label="target = load x peak")
        ax[k].set_ylabel(lab); ax[k].grid(alpha=0.3)
    ax[0].legend(); ax[0].set_title("One test episode: generated arrivals vs target"); ax[2].set_xlabel("step (1 ms)")
    fig.tight_layout()
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig("results/figures/traffic_check.png", dpi=130)
    fig2, a2 = plt.subplots(figsize=(11, 3.5))
    for k in range(3):
        a2.plot(chn["eta"][:, k], label=SLICES[k])
    a2.set_title("Spectral efficiency per slice (bit/s/Hz), one episode"); a2.legend(); a2.grid(alpha=0.3)
    fig2.tight_layout(); fig2.savefig("results/figures/channel_check.png", dpi=130)
    print("\nSaved plots to results/figures/")
except ImportError:
    pass

print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")