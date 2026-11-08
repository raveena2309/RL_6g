"""
Generate SYNTHETIC per-slice load curves that stand in for the Telecom Italia data.

Output schema is the same one the real-data pipeline will produce, so swapping
synthetic -> real later means replacing ONE file (data/processed/load_curves.csv).

Run from the project root:
    python scripts/generate_synthetic_load.py

NOTE: this is synthetic data. Do not present it as real measurements.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
START = "2013-11-01 00:00"      # same period as the real dataset (Milan local time)
N_DAYS = 62                      # 1 Nov 2013 .. 1 Jan 2014
STEP_MIN = 10                    # 10-minute intervals, like the real data
TRAIN_DAYS, VAL_DAYS = 40, 10    # remaining 12 days = test (includes Christmas/New Year)

HOLIDAYS = pd.to_datetime(["2013-11-01", "2013-12-08", "2013-12-25", "2013-12-26", "2014-01-01"]).date
XMAS_LULL = pd.to_datetime(["2013-12-23", "2013-12-24", "2013-12-27", "2013-12-30", "2013-12-31"]).date

OUT_DIR = Path("data/processed")


def bump(h, mu, sigma):
    """Gaussian bump on a 24-hour circle, so the profile is continuous across midnight."""
    d = np.abs(h - mu)
    d = np.minimum(d, 24 - d)
    return np.exp(-0.5 * (d / sigma) ** 2)


def daily_profiles(h):
    """Shape of a normal weekday for each slice, h = hour of day (0-24)."""
    embb = 0.12 + 0.45 * bump(h, 13.0, 3.5) + 0.55 * bump(h, 21.0, 2.5)                    # video/data: evening peak
    urllc = 0.05 + 0.75 * bump(h, 11.0, 2.2) + 0.65 * bump(h, 16.5, 2.2) + 0.20 * bump(h, 20.0, 2.0)  # calls: work hours
    mmtc = 0.15 + 0.40 * bump(h, 13.0, 4.0) + 0.45 * bump(h, 19.5, 3.0)                     # SMS: flatter, sparse
    return embb, urllc, mmtc


def ar1_noise(n, rng, phi=0.9, sigma=0.08):
    """Slowly varying multiplicative noise (log-space AR(1))."""
    x = np.zeros(n)
    e = rng.normal(0, sigma, n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return np.exp(x)


def main():
    rng = np.random.default_rng(SEED)
    idx = pd.date_range(START, periods=N_DAYS * 24 * 60 // STEP_MIN, freq=f"{STEP_MIN}min")
    n = len(idx)
    h = idx.hour.values + idx.minute.values / 60.0
    dates = idx.date
    weekend = idx.dayofweek.values >= 5
    holiday = np.isin(dates, HOLIDAYS)
    lull = np.isin(dates, XMAS_LULL)

    embb, urllc, mmtc = daily_profiles(h)

    # day-type effects
    off = weekend | holiday
    embb = embb * np.where(off, 1.05, 1.0)
    urllc = urllc * np.where(off, 0.5, 1.0) * np.where(lull & ~off, 0.8, 1.0)
    mmtc = mmtc * np.where(off, 0.95, 1.0) * np.where(np.isin(dates, pd.to_datetime(["2013-12-25"]).date), 1.2, 1.0)

    # New Year greeting spike for SMS around midnight 31 Dec -> 1 Jan
    ny = pd.Timestamp("2014-01-01 00:00")
    mins = (idx - ny).total_seconds().values / 60
    mmtc = mmtc + 1.2 * np.exp(-0.5 * (mins / 25.0) ** 2)

    # random surges in eMBB (events, viral content): ~30% of days get one, 30-60 min long
    surge = np.ones(n)
    for d in range(N_DAYS):
        if rng.random() < 0.30:
            start = d * 144 + int(rng.integers(60, 130))
            length = int(rng.integers(3, 7))
            surge[start:start + length] = rng.uniform(1.3, 1.8)
    embb = embb * surge

    # slowly varying noise per slice
    embb *= ar1_noise(n, rng, sigma=0.08)
    urllc *= ar1_noise(n, rng, sigma=0.06)
    mmtc *= ar1_noise(n, rng, sigma=0.10)

    df = pd.DataFrame({"timestamp": idx, "embb_raw": embb, "urllc_raw": urllc, "mmtc_raw": mmtc})
    df["day_index"] = (np.arange(n) // 144) + 1
    df["split"] = np.where(df.day_index <= TRAIN_DAYS, "train",
                   np.where(df.day_index <= TRAIN_DAYS + VAL_DAYS, "val", "test"))

    # normalise with the TRAINING peak only (test may exceed 1.0 - that is intended)
    peaks = {k: float(df.loc[df.split == "train", f"{k}_raw"].max()) for k in ["embb", "urllc", "mmtc"]}
    for k, p in peaks.items():
        df[k] = df[f"{k}_raw"] / p

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "load_curves.csv", index=False)
    meta = {
        "source": "SYNTHETIC",
        "seed": SEED,
        "interval_minutes": STEP_MIN,
        "n_days": N_DAYS,
        "split_days": {"train": TRAIN_DAYS, "val": VAL_DAYS, "test": N_DAYS - TRAIN_DAYS - VAL_DAYS},
        "split_dates": {s: [str(g.timestamp.min()), str(g.timestamp.max())] for s, g in df.groupby("split")},
        "train_peaks": peaks,
        "holidays": [str(d) for d in HOLIDAYS],
    }
    (OUT_DIR / "load_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Wrote {len(df)} rows to {OUT_DIR / 'load_curves.csv'}")
    print(df.groupby("split")[["embb", "urllc", "mmtc"]].agg(["mean", "max"]).round(2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        wk = df[(df.timestamp >= "2013-11-11") & (df.timestamp < "2013-11-18")]
        for a, (k, lab) in zip(ax, [("embb", "eMBB (internet)"), ("urllc", "URLLC (calls)"), ("mmtc", "mMTC (SMS)")]):
            a.plot(wk.timestamp, wk[k], lw=0.9)
            a.set_ylabel(lab)
            a.grid(alpha=0.3)
        ax[0].set_title("Synthetic normalised load, one week (Mon 11 Nov - Sun 17 Nov 2013)")
        fig.tight_layout()
        (Path("results") / "figures").mkdir(parents=True, exist_ok=True)
        fig.savefig(Path("results/figures/synthetic_load_week.png"), dpi=130)
        print("Saved plot to results/figures/synthetic_load_week.png")
    except ImportError:
        print("matplotlib not installed - skipping plot")


if __name__ == "__main__":
    main()