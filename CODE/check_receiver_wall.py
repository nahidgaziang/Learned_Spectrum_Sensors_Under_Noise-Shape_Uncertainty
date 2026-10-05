"""Does the receiver uncertainty set admit the wall condition k*c0 = gamma*p + c1 (Proposition) for gamma > 0?
For every ordered pair (c0, c1) of distinct receiver shapes, fit k and gamma >= 0 by least squares on the N-point
grid and report the relative residual ||k c0 - gamma p - c1|| / ||c1||. A clearly non-zero residual for every pair
means no exact solution, i.e. the proposition places no wall on this set (gamma_w = 0).
Shapes: theoretical PSDs (Butterworth |H|^2 for the channel filters, rtl_like_psd for the RTL-like roll-offs).
Writes RESULTS/receiver_wall_check.json."""
import json, os, sys
import numpy as np
import scipy.signal as ss
from scipy.optimize import lsq_linear

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim, wall as W

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 2048
f = np.fft.fftfreq(N)


def butter_psd(cut, order=6):
    b, a = ss.butter(order, cut)
    _, h = ss.freqz(b, a, worN=2 * np.pi * f)
    return np.abs(h) ** 2


shapes = {"white": np.ones(N), "lpf0.9": butter_psd(0.9), "lpf0.8": butter_psd(0.8),
          "rtl1dB": sim.rtl_like_psd(f, 1.0), "rtl3dB": sim.rtl_like_psd(f, 3.0)}
shapes = {k: v / v.mean() for k, v in shapes.items()}
p = W.signal_psd(N, n=4000)
out = {}
for n0, c0 in shapes.items():
    for n1, c1 in shapes.items():
        if n0 == n1:
            continue
        # minimise || k c0 - g p - c1 || over k >= 0, g >= 0
        r = lsq_linear(np.stack([c0, -p], 1), c1, bounds=([0, 0], [np.inf, np.inf]))
        k, g = r.x
        rel = float(np.linalg.norm(k * c0 - g * p - c1) / np.linalg.norm(c1))
        out[f"{n0}->{n1}"] = {"k": float(k), "gamma": float(g), "rel_residual": rel}
mn = min(v["rel_residual"] for v in out.values())
json.dump({"N": N, "pairs": out, "min_rel_residual": mn}, open(os.path.join(ROOT, "RESULTS", "receiver_wall_check.json"), "w"), indent=1)
for k_, v in sorted(out.items(), key=lambda kv: kv[1]["rel_residual"])[:6]:
    print(f"{k_:16s} k={v['k']:.3f} gamma={v['gamma']:.4f} rel.residual={v['rel_residual']:.4f}")
print("min relative residual over all ordered pairs:", round(mn, 4))
