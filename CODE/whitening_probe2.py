"""Why does real RTL-SDR noise not whiten fully? Separate (a) smooth-fit bias (fine spectral ripple),
(b) reference/test non-stationarity, (c) non-Gaussian or non-stationary structure beyond second order.
Variants per recording (test = windows 2000..2999):
  oracle-mov3 / oracle-cos16 / oracle-raw : PSD estimated from the TEST windows themselves (no drift, no K limit)
  ref1024-early : reference windows 0..1023 (precede test), cos16 and mov3
  ref1024-spread: 1024 windows spread over the whole recording excluding the test block, cos16
Detectors: GapCNN-WHITE seed 0 (CPU) and BAND. If oracle-raw whitening restores ~5 %, the excess Pf is a
second-order (PSD) effect; if not, something beyond second order (or within-window non-stationarity) remains."""
import glob, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim, detectors as D, real_noise_eval as R
from evaluate import whitener
from run import learned_score_fn, classical_score_fn, DEFAULTS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 128
m = D.ARCHS["GapCNN"](); m.load_state_dict(torch.load(os.path.join(ROOT, "EXPERIMENTS", "ckpt", "GapCNN_WHITE_N128_sps8_s0.pt"), map_location="cpu")); m.eval()
fns = {"GapCNN": learned_score_fn(m, "GapCNN", dev="cpu"), "BAND": classical_score_fn(dict(DEFAULTS, kind="classical", detector="BAND", N=N, sps=8))}
rng = np.random.default_rng(5)
wcal = sim.white(20000, N, rng); wcal = wcal - wcal.mean(1, keepdims=True)
thr = {k: np.quantile(f(wcal), 0.95) for k, f in fns.items()}


def raw_whitener(ref):
    P = np.mean(np.abs(np.fft.fft(ref, axis=1)) ** 2, axis=0)
    P = P / P.mean()
    inv = 1 / np.sqrt(np.maximum(P, 1e-6))
    return lambda x: np.fft.ifft(np.fft.fft(x, axis=1) * inv[None, :], axis=1)


import json
RESULT = {}
files = sorted(glob.glob(os.path.join(ROOT, "DATA", "real_noise_hf", "*.npy")))
pick = [f for f in files if os.path.basename(f) in ("noise_1200_0_001.npy", "noise_1200_0_002.npy", "noise_300_0_001.npy",
                                                     "noise_50_0_001.npy", "noise_800_0_001.npy", "NOISE_003.npy")]
for fp in pick:
    iq, meta = R.load(fp)
    w = R.windows(iq, N, max_win=4000)
    a = len(w) // 2
    b = a + min(1000, len(w) // 4)
    test = w[a:b]
    others = np.concatenate([w[:a], w[b:]])
    spread = others[np.linspace(0, len(others) - 1, min(1024, len(others))).astype(int)]
    V = {"none": lambda x: x,
         "oracle-raw": raw_whitener(test), "oracle-mov3": whitener(test, method="movavg"), "oracle-cos16": whitener(test, method="cos", n_cos=16),
         "ref1024-early-mov3": whitener(w[:1024], method="movavg"), "ref1024-early-cos16": whitener(w[:1024], method="cos", n_cos=16),
         "ref1024-spread-cos16": whitener(spread, method="cos", n_cos=16), "ref1024-spread-raw": raw_whitener(spread)}
    res = []
    RESULT[os.path.basename(fp)] = {}
    for name, wf in V.items():
        vals = {k: float(np.mean(f(wf(test)) > thr[k])) for k, f in fns.items()}
        RESULT[os.path.basename(fp)][name] = vals
        res.append(f"{name}: " + "/".join(f"{v:.3f}" for v in vals.values()))
    print(os.path.basename(fp)[:20], " | ".join(res), flush=True)
json.dump({"detectors": list(fns), "note": "GapCNN = GapCNN-WHITE seed 0 (CPU); test = middle block of windows",
           "results": RESULT}, open(os.path.join(ROOT, "RESULTS", "whitening_probe_oracle.json"), "w"), indent=1)
