"""Verification 2026-10-05 (internal notes): is CycNet's advantage over classical CYC due to 'nonlinear fusion'
(as the text says) or simply to using 16 lags instead of 5? Classical normalised cyclic-autocorrelation detector
(detectors.cyc: sum over lags of |R_alpha(tau)|^2 / P0^2, alpha = 1/sps) with lags 0-4 (as in the paper) and 0-15 (as
CycNet's input), under the escape-test protocol: N = 32768, u = 1 and 2 dB, extremal pair (wall_test_long.psd_on_grid),
thresholds = max (1-alpha)-quantile over {white, c0, c1}, worst-case Pd over H1 noise {white, c1} at wall + delta.
CPU only. Output RESULTS/cyc_lag_ablation.json.   usage: py -3.12 cyc_lag_ablation.py"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim, detectors as D, wall as W
from wall_test_long import psd_on_grid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALPHA, N = 0.05, 32768
FNS = {"CYC5": lambda x: D.cyc(x, sps=8, lags=tuple(range(5))), "CYC16": lambda x: D.cyc(x, sps=8, lags=tuple(range(16)))}


def sc(gen, n, chunk=64):
    out = {k: [] for k in FNS}
    for i in range(0, n, chunk):
        x = gen(min(chunk, n - i))
        for k, f in FNS.items():
            out[k].append(f(x))
    return {k: np.concatenate(v) for k, v in out.items()}


res = {}
for seed in (0, 1, 2):
    for R_db in (1.0, 2.0):
        g, c0, c1 = W.extremal_pair(psd_on_grid(N), R_db)
        _, rcal, rte = sim.streams(60_000 + seed)
        cal = [sc(lambda m: sim.white(m, N, rcal), 2000), sc(lambda m: sim.from_psd(m, N, rcal, c0), 2000),
               sc(lambda m: sim.from_psd(m, N, rcal, c1), 2000)]
        thr = {k: max(np.quantile(c[k], 1 - ALPHA) for c in cal) for k in FNS}
        r = {}
        for d in (-2, 0):
            a = np.sqrt(g * 10 ** (d / 10))
            w = {}
            for nz, ng in (("white", lambda m: sim.white(m, N, rte)), ("c1", lambda m: sim.from_psd(m, N, rte, c1))):
                s = sc(lambda m: sim.modulated(m, N, rte) * a + ng(m), 1000)
                w[nz] = {k: float(np.mean(s[k] > thr[k])) for k in FNS}
            r[str(d)] = {k: min(w["white"][k], w["c1"][k]) for k in FNS}
        res[f"s{seed}_R{R_db}"] = r
        print(seed, R_db, r, flush=True)
summ = {f"R{R}": {d: {k: float(np.mean([res[f's{s}_R{R}'][d][k] for s in (0, 1, 2)])) for k in FNS} for d in ("-2", "0")}
        for R in (1.0, 2.0)}
print("mean over seeds:", summ)
json.dump({"per_seed": res, "mean": summ, "N": N, "alpha": ALPHA}, open(os.path.join(ROOT, "RESULTS", "cyc_lag_ablation.json"), "w"), indent=1)
