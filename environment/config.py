"""
Central configuration for the 6G slicing simulator.
Every number here is an ASSUMPTION. Document the justification in the report.
Do not tune these to make the RL agent win.
"""
from dataclasses import dataclass

SLICES = ("embb", "urllc", "mmtc")      # fixed order used everywhere: index 0, 1, 2


@dataclass
class Config:
    # ---- time ----
    step_s: float = 1e-3                 # 1 ms scheduling step
    episode_steps: int = 1000            # 1000 steps = 1 s of simulated time
    window_hours: float = 6.0            # the 1 s episode is time-compressed to cover a 6 h window of the load curve
    interval_min: int = 10               # resolution of the load curve (10 minutes)

    # ---- radio ----
    total_bandwidth_hz: float = 100e6    # 100 MHz sub-6 GHz carrier
    carrier_ghz: float = 3.5             # 3.5 GHz
    tx_power_dbm: float = 40.0           # total transmit power, spread evenly over the bandwidth
    noise_figure_db: float = 7.0
    bs_height_m: float = 10.0            # urban micro cell
    ue_height_m: float = 1.5
    dist_range_m: tuple = (50.0, 500.0)  # user distance from the base station
    users_per_slice: tuple = (40, 20, 40)  # eMBB, URLLC, mMTC
    fading_sigma_db: float = 1.0         # small slow fading around the mean
    fading_phi: float = 0.98
    blockage_start_prob: float = 0.0005  # per step, per user
    blockage_end_prob: float = 0.01      # mean blockage ~ 100 steps
    blockage_loss_db: float = 20.0
    shannon_gap: float = 0.75            # implementation loss vs ideal Shannon
    eta_cap: float = 7.0                 # max spectral efficiency (bit/s/Hz), ~256-QAM limit

    # ---- traffic (slice-level) ----
    peak_rate_bps: tuple = (300e6, 20e6, 5e6)     # slice demand when normalised load = 1.0
    packet_bits: tuple = (12000, 1600, 800)       # 1500 B, 200 B, 100 B
    deadline_s: tuple = (0.5, 0.010, 0.5)         # packets older than this are dropped (URLLC: 10 ms)
    embb_gamma_shape: float = 4.0
    embb_burst_sigma: float = 0.5        # strength of slow bursts for eMBB
    embb_burst_tau_s: float = 0.05       # burst memory (50 ms)
    load_scale: float = 1.0              # multiply all demand (used in load-scaling / scarcity experiments)

    # ---- paths ----
    load_csv: str = "data/processed/load_curves.csv"