"""Noise-shape SNR wall for scale-invariant (second-order) detectors, for an arbitrary signal PSD.

Uncertainty set U_R: noise PSD shapes c(f) with max c / min c <= R (unit average power).
Signal PSD p(f) >= 0 with unit average power; q = p / p_max.

Wall (see THEORY.md):  gamma_wall = (R - 1/R) / ( p_max * I ),  I = mean_f max(1/R, 1 - (R - 1/R) q(f)).
Extremal pair at gamma = gamma_wall (both shapes have ratio <= R):
    c1*(f) = v * max(1/R, 1 - (R - 1/R) q(f)),      v = 1 / I          (noise under H1)
    c0*(f) = (gamma_wall * p(f) + c1*(f)) / mean(...)                (noise under H0)
so that gamma_wall*p + c1* is proportional to c0*: the two hypotheses have identical covariances up
to scale, and no scale-invariant test can separate them (Gaussian signal) for any N.
"""
import numpy as np
import sim


def signal_psd(N, sps=8, beta=0.35, n=4000, seed=123):
    """Average periodogram of `sim.modulated` on the N-point DFT grid (fft order), unit mean."""
    rng = np.random.default_rng(seed)
    x = sim.modulated(n, N, rng, sps=sps, beta=beta)
    w = np.hanning(N)
    P = np.mean(np.abs(np.fft.fft(x * w[None, :], axis=1)) ** 2, axis=0)
    return P / P.mean()


def wall(p, R_db):
    R = 10 ** (R_db / 10)
    pm = p.max()
    q = p / pm
    I = np.mean(np.maximum(1 / R, 1 - (R - 1 / R) * q))
    return (R - 1 / R) / (pm * I)


def extremal_pair(p, R_db):
    R = 10 ** (R_db / 10)
    pm = p.max()
    q = p / pm
    base = np.maximum(1 / R, 1 - (R - 1 / R) * q)
    I = base.mean()
    c1 = base / I
    g = (R - 1 / R) / (pm * I)
    c0 = g * p + c1
    c0 = c0 / c0.mean()
    return g, c0, c1


def check_ratio(c):
    return c.max() / c.min()


if __name__ == "__main__":
    for N in [128, 512, 2048]:
        p = signal_psd(N)
        for R_db in [0.5, 1.0, 2.0, 3.0]:
            g, c0, c1 = extremal_pair(p, R_db)
            print(f"N={N} R={R_db}dB wall={10*np.log10(g):6.2f} dB  ratio c0={10*np.log10(check_ratio(c0)):.2f}dB c1={10*np.log10(check_ratio(c1)):.2f}dB"
                  f"  flat-band formula={10*np.log10(sim.shape_wall(sim.occupancy(8), R_db)):6.2f} dB")
