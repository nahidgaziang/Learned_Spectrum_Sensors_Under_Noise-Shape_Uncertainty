"""Numerical check of Theorem 2 (achievability of the flat-band wall) — CPU only.
Flat-band Gaussian signal p = 1/rho on |f| <= rho/2 (rho = 0.169), uncertainty u = 1 dB (R = 10^0.1).
Test: band-power ratio T_N > R + eta with a FIXED threshold (no calibration), eta chosen for SNR = wall + 1 dB.
H0: the two-level extremal shapes c0, c1 and 40 random members of U_R (random two-level and smooth shapes with
peak-to-peak ratio exactly R). H1: signal at SNR = wall + delta in noise c1 (worst case) and in white noise.
Predictions: max Pf over shapes -> 0 and Pd(delta = +1 dB) -> 1 as N grows; Pd(delta <= 0) stays small.
Writes RESULTS/theorem_check.json."""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rho, u = 0.169, 1.0
R = 10 ** (u / 10)
gw = rho * (R ** 2 - 1) / (rho + R * (1 - rho))
rng = np.random.default_rng(2026)


def grid_shapes(N):
    f = np.fft.fftfreq(N)
    inB = np.abs(f) <= rho / 2
    p = np.where(inB, 1.0, 0.0); p /= p.mean()
    Z1 = R / (rho + R * (1 - rho)); c1 = np.where(inB, Z1 / R, Z1)
    Z0 = 1 / (rho * R + 1 - rho); c0 = np.where(inB, R * Z0, Z0)
    rand = []
    r2 = np.random.default_rng(7)
    for i in range(40):
        if i % 2 == 0:      # random two-level shape on a random band
            lo, w = r2.uniform(-0.5, 0.5), r2.uniform(0.05, 0.6)
            m = np.abs(((f - lo + 0.5) % 1.0) - 0.5) <= w / 2
            s = np.where(m, R, 1.0) if r2.random() < 0.5 else np.where(m, 1.0, R)
        else:               # random smooth shape rescaled to peak-to-peak ratio R
            a = r2.normal(0, 1, 6) / np.arange(1, 7)
            g = sum(a[k] * np.cos(2 * np.pi * (k + 1) * f + r2.uniform(0, 2 * np.pi)) for k in range(6))
            g = (g - g.min()) / max(np.ptp(g), 1e-12)
            s = 10 ** (u * g / 10)
        rand.append(s / s.mean())
    return f, inB, p, c0, c1, rand


def ratio(x, inB):
    P = np.abs(np.fft.fft(x, axis=1)) ** 2
    return P[:, inB].mean(1) / P[:, ~inB].mean(1)


out = {"rho": rho, "u_db": u, "wall_db": float(10 * np.log10(gw))}
for N in [512, 2048, 8192, 32768]:
    f, inB, p, c0, c1, rand = grid_shapes(N)
    g1 = gw * 10 ** 0.1                                  # wall + 1 dB
    lim_h1 = g1 * (rho + R * (1 - rho)) / (rho * R) + 1 / R
    eta = 0.5 * (lim_h1 - R)
    thr = R + eta
    n = max(200, int(4_000_000 // N))
    pf = [float(np.mean(ratio(sim.from_psd(n, N, rng, s), inB) > thr)) for s in [c0, c1] + rand]
    pd = {}
    for d in [-1.0, 0.0, 0.5, 1.0]:
        a = np.sqrt(gw * 10 ** (d / 10))
        r = []
        for noise in [c1, np.ones(N)]:
            x = a * sim.from_psd(n, N, rng, p) + sim.from_psd(n, N, rng, noise)
            r.append(float(np.mean(ratio(x, inB) > thr)))
        pd[str(d)] = min(r)
    out[f"N{N}"] = {"threshold": thr, "n": n, "max_Pf_over_shapes": max(pf), "worst_Pd": pd}
    print(f"N={N:6d} thr={thr:.3f} n={n} max Pf over {len(pf)} shapes {max(pf):.4f}  worst Pd " +
          " ".join(f"{d}:{v:.3f}" for d, v in pd.items()), flush=True)
json.dump(out, open(os.path.join(ROOT, "RESULTS", "theorem_check.json"), "w"), indent=1)
