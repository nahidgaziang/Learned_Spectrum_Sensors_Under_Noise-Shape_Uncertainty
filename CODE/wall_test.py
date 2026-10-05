"""X4/X8 wall test with the theory's extremal noise pair.

For R in {1, 2} dB and N in {128, 512, 2048}: robust threshold over U* = {white, c0*, c1*}
(calibration stream); report worst-case Pd over H1 noise in {white, c1*} at SNR = wall + delta,
delta in {-4, -2, 0, +2, +4, +6} dB, together with realised max Pf on the test stream.
Prediction: second-order scale-invariant detectors (BAND, MME, SFM, learned WHITE) give
worst-case Pd <= alpha at delta <= 0 for every N; structure-based detectors (CYC, learned SURR) can
exceed alpha and improve with N.  Output: RESULTS/wall_test.json
"""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim, detectors as D, wall as W
from run import learned_score_fn, classical_score_fn, DEFAULTS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALPHA = 0.05


def scores(fn, gen, n, chunk=2000):
    return np.concatenate([fn(gen(min(chunk, n - i))) for i in range(0, n, chunk)])


def save(out, path):
    with open(path + ".tmp", "w") as fh:          # atomic write (power-loss safe)
        json.dump(out, fh, indent=1)
    os.replace(path + ".tmp", path)


def main(seed=0, n_cal=6000, n_te=3000):
    """Incremental: keys already present in RESULTS/wall_test_s{seed}.json are kept and not recomputed.
    Every detector sees the same calibration and test draws (streams re-created per detector)."""
    os.makedirs(os.path.join(ROOT, "RESULTS"), exist_ok=True)
    path = os.path.join(ROOT, "RESULTS", f"wall_test_s{seed}.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    t0 = time.time()
    dets = ["ED", "BAND", "MME", "SFM", "CYC"]
    for N in [128, 512, 2048]:
        p = W.signal_psd(N)
        learned = {}
        for rec in ["WHITE", "SURR", "SURRX", "SURRXM"]:
            ck = os.path.join(ROOT, "EXPERIMENTS", "ckpt", f"GapCNN_{rec}_N{N}_sps8_s{seed}.pt")
            if os.path.exists(ck):
                m = D.ARCHS["GapCNN"]().cuda(); m.load_state_dict(torch.load(ck, map_location="cuda")); m.eval()
                learned[f"GapCNN-{rec}"] = learned_score_fn(m, "GapCNN")
        if N == 128:
            ck = os.path.join(ROOT, "EXPERIMENTS", "ckpt", f"CLDNN_WHITE_N128_sps8_s{seed}.pt")
            if os.path.exists(ck):
                m = D.ARCHS["CLDNN"]().cuda(); m.load_state_dict(torch.load(ck, map_location="cuda")); m.eval()
                learned["CLDNN-WHITE"] = learned_score_fn(m, "CLDNN")
        fns = {d: classical_score_fn(dict(DEFAULTS, kind="classical", detector=d, N=N, sps=8)) for d in dets}
        fns.update(learned)
        for R_db in [1.0, 2.0]:
            g, c0, c1 = W.extremal_pair(p, R_db)
            wall_db = 10 * np.log10(g)
            for name, fn in fns.items():
                if f"N{N}_R{R_db}_{name}" in out:
                    continue
                _, rcal, rte = sim.streams(30_000 + seed)        # common random numbers across detectors
                gens_h0 = {"white": lambda m: sim.white(m, N, rcal), "c0": lambda m: sim.from_psd(m, N, rcal, c0),
                           "c1": lambda m: sim.from_psd(m, N, rcal, c1)}
                thr = max(np.quantile(scores(fn, gen, n_cal), 1 - ALPHA) for gen in gens_h0.values())
                pf = {k: float(np.mean(scores(fn, gen, n_te) > thr)) for k, gen in
                      {"white": lambda m: sim.white(m, N, rte), "c0": lambda m: sim.from_psd(m, N, rte, c0),
                       "c1": lambda m: sim.from_psd(m, N, rte, c1)}.items()}
                pd = {}
                for delta in [-4, -2, 0, 2, 4, 6]:
                    snr = wall_db + delta
                    a = np.sqrt(10 ** (snr / 10))
                    r = {}
                    for hn, ng in {"white": lambda m: sim.white(m, N, rte), "c1": lambda m: sim.from_psd(m, N, rte, c1)}.items():
                        s = scores(fn, lambda m: sim.modulated(m, N, rte) * a + ng(m), n_te)
                        u = scores(fn, lambda m: sim.surrogate(m, N, rte) * a + ng(m), n_te)
                        r[hn] = float(np.mean(s > thr)); r[hn + "_sur"] = float(np.mean(u > thr))
                    r["worst"] = min(r["white"], r["c1"])
                    r["worst_sur"] = min(r["white_sur"], r["c1_sur"])
                    pd[str(delta)] = r
                out[f"N{N}_R{R_db}_{name}"] = {"wall_db": wall_db, "Pf": pf, "Pd": pd}
                print(f"N={N} R={R_db} {name:12s} wall={wall_db:.2f}dB Pfmax={max(pf.values()):.3f} worstPd="
                      + " ".join(f"{d}:{pd[str(d)]['worst']:.3f}" for d in [-4, -2, 0, 2, 4, 6]) + f"  ({time.time()-t0:.0f}s)", flush=True)
        save(out, path)                               # checkpoint after each N


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)

