"""Probe: Pf after reference whitening on real RTL-SDR noise for whitening variants (K, method).
CPU only (classical BAND + learned GapCNN-WHITE seed 0 on CPU), a few recordings per group."""
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
fns = {"GapCNN-WHITE": learned_score_fn(m, "GapCNN", dev="cpu"), "BAND": classical_score_fn(dict(DEFAULTS, kind="classical", detector="BAND", N=N, sps=8))}
rng = np.random.default_rng(5)
wcal = sim.white(20000, N, rng); wcal = wcal - wcal.mean(1, keepdims=True)
thr = {k: np.quantile(f(wcal), 0.95) for k, f in fns.items()}
files = sorted(glob.glob(os.path.join(ROOT, "DATA", "real_noise_hf", "*.npy")))
pick = [f for f in files if os.path.basename(f).startswith(("NOISE_00", "noise_1200_0_00", "noise_300_0_00", "noise_50_0_00"))][:12]
variants = [(16, "movavg"), (16, "cos"), (64, "movavg"), (64, "cos"), (256, "movavg"), (256, "cos")]
for fp in pick:
    iq, meta = R.load(fp)
    w = R.windows(iq, N)
    test = w[300:1300]
    out = []
    for K, meth in variants:
        wh = whitener(w[:K], method=meth)
        out.append(f"K{K}-{meth}: " + " ".join(f"{k}={np.mean(f(wh(test)) > thr[k]):.3f}" for k, f in fns.items()))
    raw = " ".join(f"{k}={np.mean(f(test) > thr[k]):.3f}" for k, f in fns.items())
    print(os.path.basename(fp)[:22], "raw:", raw, "|", " | ".join(out), flush=True)
