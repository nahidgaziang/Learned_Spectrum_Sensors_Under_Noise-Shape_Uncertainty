"""Why does long-reference whitening leave LP .8 false alarms at ~12 % (queue C) while smooth shapes return to ~5 %?
Hypothesis: per-bin (DFT) whitening of a 128-sample window is exact only for circulant noise. The Butterworth
channel filter is a real (non-circulant) filter whose response falls to zero at Nyquist, so the window's periodogram
near the edge is dominated by leakage that whitening amplifies (residual inter-bin correlation).
Test: whiten with a near-exact PSD (100k reference windows; estimation error ~0.3 % per bin) and compare
  - lpf0.8 (linear filter, non-circulant)   vs
  - circulant noise with the SAME expected periodogram (sim.from_psd), which per-bin whitening makes exactly white.
Detector: GapCNN-WHITE seed 0 (CPU), nominal white-noise threshold (DC not blocked, as in queue C).
Writes RESULTS/whitening_probe_lpf.json.
"""
import json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim, detectors as D
from evaluate import whitener
from run import learned_score_fn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 128
m = D.ARCHS["GapCNN"]()
m.load_state_dict(torch.load(os.path.join(ROOT, "EXPERIMENTS", "ckpt", "GapCNN_WHITE_N128_sps8_s0.pt"), map_location="cpu"))
m.eval()
f = learned_score_fn(m, "GapCNN", dev="cpu")
rng = np.random.default_rng(11)
thr = float(np.quantile(f(sim.white(20000, N, rng)), 0.95))
out = {"thr": thr}
for cutoff in (0.8, 0.9):
    ref = np.concatenate([sim.butter_rolloff(20_000, N, rng, cutoff) for _ in range(5)])   # 100k windows, in chunks
    P = np.mean(np.abs(np.fft.fft(ref, axis=1)) ** 2, axis=0)          # expected periodogram (near-exact)
    w_raw = whitener(ref, method="movavg", smooth=1)                      # per-bin, no smoothing
    test_lin = sim.butter_rolloff(4000, N, rng, cutoff)                   # linear filter (non-circulant)
    test_circ = sim.from_psd(4000, N, rng, P)                             # circulant, same expected periodogram
    r = {"none_linear": float(np.mean(f(test_lin) > thr)),
         "whitened_linear": float(np.mean(f(w_raw(test_lin)) > thr)),
         "none_circulant": float(np.mean(f(test_circ) > thr)),
         "whitened_circulant": float(np.mean(f(w_raw(test_circ)) > thr)),
         "edge_power_db": float(10 * np.log10(P[N // 2] / P.mean()))}
    out[f"lpf{cutoff}"] = r
    print(f"lpf{cutoff}: " + ", ".join(f"{k}={v:.3f}" for k, v in r.items()), flush=True)
json.dump(out, open(os.path.join(ROOT, "RESULTS", "whitening_probe_lpf.json"), "w"), indent=1)
