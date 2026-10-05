"""Long observation wall test for Plan C escape models:
Evaluates whether learned detectors trained at N = 8192 or with spectral pairing escape below the wall.

Detectors evaluated:
- BAND (oracle second-order baseline)
- CYC (classical cyclic autocorrelation baseline)
- GapCNN-WHITE-N2048 (unconstrained CNN trained at N=2048, inference at N)
- GapCNN-WHITE-N8192 (unconstrained CNN trained at N=8192, inference at N)
- GapCNN-SURRXM-N8192 (matched surrogate mixture trained at N=8192)
- GapCNN-PAIRED-N8192 (exact spectral surrogate twins trained at N=8192)
- CycNet-WHITE-N8192 (cyclic-feature network trained at N=8192)

Protocol:
- Observation lengths N = 8192 and 32768
- Uncertainty sets u = 1.0 dB and 2.0 dB
- SNR = wall + delta, delta in [-2, 0, 2, 4] dB
- Calibration: n_cal = 3000 draws
- Evaluation: n_te = 2000 draws
- Streams: sim.streams(40_000 + seed), exactly matching wall_test_long.py and wall_test_long_nets.py
Output: RESULTS/wall_test_escape_s{seed}.json (atomic)
"""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim, detectors as D, wall as W
from run import learned_score_fn
from wall_test_long import psd_on_grid, scores

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALPHA = 0.05


def run_seed(seed=0, Ns=(8192, 32768), Rs=(1.0, 2.0), deltas=(-2, 0, 2, 4), n_cal=3000, n_te=2000):
    path = os.path.join(ROOT, "RESULTS", f"wall_test_escape_s{seed}.json")
    out = json.load(open(path)) if os.path.exists(path) else {}

    hw = sim.occupancy(8, 0.35) / 2
    fns = {
        "BAND": lambda x: D.band_ratio(x, half_width=hw),
        "CYC": lambda x: D.cyc(x, sps=8),
    }

    models = [
        ("GapCNN-WHITE-N2048", "GapCNN", f"GapCNN_WHITE_N2048_sps8_s{seed}.pt", 64),
        ("GapCNN-WHITE-N8192", "GapCNN", f"GapCNN_WHITE_N8192_sps8_s{seed}.pt", 64),
        ("GapCNN-SURRXM-N8192", "GapCNN", f"GapCNN_SURRXM_N8192_sps8_s{seed}.pt", 64),
        ("GapCNN-PAIRED-N8192", "GapCNN", f"GapCNN_PAIRED_N8192_sps8_s{seed}.pt", 64),
        ("CycNet-WHITE-N8192", "CycNet", f"CycNet_WHITE_N8192_sps8_s{seed}.pt", 512),
    ]

    for tag, arch, ck_name, bs in models:
        ck = os.path.join(ROOT, "EXPERIMENTS", "ckpt", ck_name)
        m = D.ARCHS[arch]().cuda()
        m.load_state_dict(torch.load(ck, map_location="cuda"))
        m.eval()
        fns[tag] = learned_score_fn(m, arch, bs=bs)

    t0 = time.time()
    for R_db in Rs:
        for N in Ns:
            key = f"N{N}_R{R_db}"
            if key in out:
                continue
            chunk = max(50, int(2_000_000 // N))
            p = psd_on_grid(N)
            g, c0, c1 = W.extremal_pair(p, R_db)
            wall_db = 10 * np.log10(g)
            _, rcal, rte = sim.streams(40_000 + seed)
            cal = [scores(fns, gen, n_cal, chunk) for gen in
                   (lambda m: sim.white(m, N, rcal), lambda m: sim.from_psd(m, N, rcal, c0), lambda m: sim.from_psd(m, N, rcal, c1))]
            thr = {k: float(max(np.quantile(c[k], 1 - ALPHA) for c in cal)) for k in fns}
            pf = {}
            for name, gen in {"white": lambda m: sim.white(m, N, rte), "c0": lambda m: sim.from_psd(m, N, rte, c0),
                              "c1": lambda m: sim.from_psd(m, N, rte, c1)}.items():
                s = scores(fns, gen, n_te, chunk)
                pf[name] = {k: float(np.mean(s[k] > thr[k])) for k in fns}
            pd = {}
            for d in deltas:
                a = np.sqrt(10 ** ((wall_db + d) / 10))
                r = {}
                for hn, ng in {"white": lambda m: sim.white(m, N, rte), "c1": lambda m: sim.from_psd(m, N, rte, c1)}.items():
                    s = scores(fns, lambda m: sim.modulated(m, N, rte) * a + ng(m), n_te, chunk)
                    r[hn] = {k: float(np.mean(s[k] > thr[k])) for k in fns}
                pd[str(d)] = {k: {"white": r["white"][k], "c1": r["c1"][k], "worst": min(r["white"][k], r["c1"][k])} for k in fns}
            out[key] = {"N": N, "R_db": R_db, "wall_db": float(wall_db), "thr": thr, "Pf": pf, "Pd": pd, "n_cal": n_cal, "n_te": n_te}
            with open(path + ".tmp", "w") as fh:
                json.dump(out, fh, indent=1)
            os.replace(path + ".tmp", path)
            print(f"seed={seed} u={R_db} N={N} wall={wall_db:.2f} dB  " + "  ".join(
                f"{k}: Pfmax={max(pf[s_][k] for s_ in pf):.3f} worst " + " ".join(f"{d}:{pd[str(d)][k]['worst']:.3f}" for d in deltas)
                for k in fns) + f"  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    seeds = [int(sys.argv[1])] if len(sys.argv) > 1 else [0, 1, 2]
    for s in seeds:
        run_seed(s)
