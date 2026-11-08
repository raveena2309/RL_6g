"""
Channel model: distance -> path loss -> shadowing/fading/blockage -> SNR -> spectral efficiency.

Path loss follows the 3GPP TR 38.901 Urban-Micro street-canyon formulas (LOS / NLOS with LOS probability).
Constants below are written from memory of the standard: VERIFY them against TR 38.901 Table 7.4.1-1
before putting them in the report.

Design choice: transmit power is spread evenly over the whole bandwidth, so SNR does not depend on how much
bandwidth a slice receives. A slice's capacity is then  C_k = b_k * eta_k  with
eta_k = mean over its users of  min(cap, gap * log2(1 + SNR_i)).
"""
import numpy as np

from .config import Config


def los_probability(d2d):
    """UMi street-canyon LOS probability."""
    d = np.maximum(d2d, 1e-3)
    p = 18.0 / d + np.exp(-d / 36.0) * (1 - 18.0 / d)
    return np.where(d <= 18.0, 1.0, p)


def path_loss_db(d2d, los, cfg: Config):
    d3d = np.sqrt(d2d ** 2 + (cfg.bs_height_m - cfg.ue_height_m) ** 2)
    fc = cfg.carrier_ghz
    pl_los = 32.4 + 21.0 * np.log10(d3d) + 20.0 * np.log10(fc)
    pl_nlos = 35.3 * np.log10(d3d) + 22.4 + 21.3 * np.log10(fc) - 0.3 * (cfg.ue_height_m - 1.5)
    return np.where(los, pl_los, np.maximum(pl_los, pl_nlos))


def noise_dbm(cfg: Config):
    return -174.0 + 10.0 * np.log10(cfg.total_bandwidth_hz) + cfg.noise_figure_db


class ChannelModel:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.slice_of_user = np.concatenate([np.full(n, k) for k, n in enumerate(cfg.users_per_slice)])

    def generate_episode(self, rng: np.random.Generator):
        """Return per-step, per-slice spectral efficiency and mean SNR (arrays of shape [T, 3])."""
        c = self.cfg
        T, N = c.episode_steps, len(self.slice_of_user)

        d2d = rng.uniform(*c.dist_range_m, size=N)
        los = rng.random(N) < los_probability(d2d)
        shadow = rng.normal(0.0, np.where(los, 4.0, 7.82))           # dB, fixed per user per episode
        base_snr = c.tx_power_dbm - path_loss_db(d2d, los, c) - shadow - noise_dbm(c)

        fading = np.zeros((T, N))
        blocked = np.zeros((T, N), dtype=bool)
        state = np.zeros(N, dtype=bool)
        f = np.zeros(N)
        innov = c.fading_sigma_db * np.sqrt(1 - c.fading_phi ** 2)
        for t in range(T):
            f = c.fading_phi * f + rng.normal(0.0, innov, N)
            fading[t] = f
            u = rng.random(N)
            state = np.where(state, u > c.blockage_end_prob, u < c.blockage_start_prob)
            blocked[t] = state

        snr_db = base_snr[None, :] - fading - blocked * c.blockage_loss_db
        eta_user = np.minimum(c.eta_cap, c.shannon_gap * np.log2(1.0 + 10.0 ** (snr_db / 10.0)))

        eta = np.zeros((T, 3))
        snr_slice = np.zeros((T, 3))
        for k in range(3):
            m = self.slice_of_user == k
            eta[:, k] = eta_user[:, m].mean(axis=1)
            snr_slice[:, k] = snr_db[:, m].mean(axis=1)
        return {"eta": eta, "snr_db": snr_slice, "blocked_fraction": float(blocked.mean())}