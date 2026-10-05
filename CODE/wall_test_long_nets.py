"""J5: do learned detectors escape the wall at long observation lengths?

The size-agnostic GAP CNN (global pooling) trained at N = 2048 is applied to N = 2048, 8192 and 32768 windows
(inference only) in the same protocol as wall_test_long.py: extremal pair for u = 1 and 2 dB, robust threshold over
{white, c0, c1}, worst-case Pd over H1 noise {white, c1} at SNR = wall + delta. Recipes: WHITE (second order;
prediction: stays at alpha at the wall for every N) and SURRXM (matched exact-surrogate negatives; uses structure at
moderate SNR). Seeds 0-2 use the matching N = 2048 checkpoints. Same random streams and call order as
wall_test_long.py, so the draws equal those seen by BAND/CYC for u = 1 dB.
Output: RESULTS/wall_test_long_nets_s{seed}.json (incremental, atomic).   Usage: py -3.12 wall_test_long_nets.py <seed>
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


def main(seed=0, Ns=(2048, 8192, 32768), Rs=(1.0, 2.0), deltas=(-2, 0, 2, 4), n_cal=3000, n_te=2000):
    path = os.path.join(ROOT, "RESULTS", f"wall_test_long_nets_s{seed}.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    fns = {}
    for rec in ["WHITE", "SURRXM"]:
        ck = os.path.join(ROOT, "EXPERIMENTS", "ckpt", f"GapCNN_{rec}_N2048_sps8_s{seed}.pt")
        m = D.ARCHS["GapCNN"]().cuda(); m.load_state_dict(torch.load(ck, map_location="cuda")); m.eval()
        fns[f"GapCNN-{rec}"] = learned_score_fn(m, "GapCNN", bs=64)
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
            out[key] = {"N": N, "R_db": R_db, "wall_db": float(wall_db), "thr": thr, "Pf": pf, "Pd": pd, "n_cal": n_cal, "n_te": n_te,
                        "note": "networks trained at N=2048, applied at N (global pooling)"}
            with open(path + ".tmp", "w") as fh:
                json.dump(out, fh, indent=1)
            os.replace(path + ".tmp", path)
            print(f"u={R_db} N={N} wall={wall_db:.2f} dB  " + "  ".join(
                f"{k}: Pfmax={max(pf[s_][k] for s_ in pf):.3f} worst " + " ".join(f"{d}:{pd[str(d)][k]['worst']:.3f}" for d in deltas)
                for k in fns) + f"  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
