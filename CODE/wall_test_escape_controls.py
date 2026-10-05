"""Control experiment for the escape claims (verification 2026-10-05, internal notes).

Concern: sim.from_psd synthesises noise by circulant FFT shaping, so every simulated coloured-noise window is periodic
(circular), while sim.modulated signals and sim.surrogate are linear segments of longer streams. The PAIRED recipe uses
per-window FFT surrogates (also circular) as negatives, so a network could learn 'non-circular narrowband component =
signal', a simulation artefact that would also let it pass the wall test (whose c0/c1 noise is circular).
Controls (N = 8192, u = 1 and 2 dB, delta = -2, 0, +2 dB below/at/above the wall, seeds 0-2):
  noise mode 'circ' = original synthesis (from_psd at N, extremal pair on the N grid);
  noise mode 'lin'  = linear windows: from_psd at 4N with the extremal pair on the 4N grid, middle N samples kept
                      (covariance = Toeplitz section, as for a real receiver);
  H1 'sig'  = modulated signal (as in wall_test_escape.py);
  H1 'gsur' = linear stationary Gaussian surrogate with the signal's PSD (sim.surrogate): no modulation or
              cyclostationary structure, so for 'lin' noise at the wall Proposition 1 gives Pd = Pf exactly in law.
Thresholds per noise mode: max over {white, c0, c1} of the (1-alpha) quantile (as in the original).
Reading: a detector that escapes through genuine structure keeps Pd(sig, lin) high and has Pd(gsur) <= ~alpha in both
modes; an artefact shows as Pd(sig, lin) << Pd(sig, circ) and/or Pd(gsur, circ) >> alpha.
Output: RESULTS/wall_test_escape_controls_s{seed}.json (atomic, incremental).   usage: py -3.12 wall_test_escape_controls.py [seed]"""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim, detectors as D, wall as W
from run import learned_score_fn
from wall_test_long import psd_on_grid, scores

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALPHA, M = 0.05, 4


def lin_from_psd(m, N, rng, psd_fine):
    x = sim.from_psd(m, M * N, rng, psd_fine)
    a = (M * N - N) // 2
    return np.ascontiguousarray(x[:, a:a + N])


def run_seed(seed=0, N=8192, Rs=(1.0, 2.0), deltas=(-2, 0, 2), n_cal=3000, n_te=1500):
    path = os.path.join(ROOT, "RESULTS", f"wall_test_escape_controls_s{seed}.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    hw = sim.occupancy(8, 0.35) / 2
    fns = {"BAND": lambda x: D.band_ratio(x, half_width=hw), "CYC": lambda x: D.cyc(x, sps=8)}
    for tag, arch, ck_name, bs in [("GapCNN-WHITE-N8192", "GapCNN", f"GapCNN_WHITE_N8192_sps8_s{seed}.pt", 64),
                                   ("GapCNN-SURRXM-N8192", "GapCNN", f"GapCNN_SURRXM_N8192_sps8_s{seed}.pt", 64),
                                   ("GapCNN-PAIRED-N8192", "GapCNN", f"GapCNN_PAIRED_N8192_sps8_s{seed}.pt", 64),
                                   ("CycNet-WHITE-N8192", "CycNet", f"CycNet_WHITE_N8192_sps8_s{seed}.pt", 512)]:
        m = D.ARCHS[arch]().cuda()
        m.load_state_dict(torch.load(os.path.join(ROOT, "EXPERIMENTS", "ckpt", ck_name), map_location="cuda"))
        m.eval()
        fns[tag] = learned_score_fn(m, arch, bs=bs)
    t0 = time.time()
    chunk = max(50, int(1_000_000 // N))
    for R_db in Rs:
        key = f"N{N}_R{R_db}"
        if key in out:
            continue
        g, c0, c1 = W.extremal_pair(psd_on_grid(N), R_db)
        gf, c0f, c1f = W.extremal_pair(psd_on_grid(M * N), R_db)
        _, rcal, rte = sim.streams(50_000 + seed)
        modes = {
            "circ": {"wall": g, "white": lambda m_, r: sim.white(m_, N, r), "c0": lambda m_, r: sim.from_psd(m_, N, r, c0),
                     "c1": lambda m_, r: sim.from_psd(m_, N, r, c1)},
            "lin": {"wall": gf, "white": lambda m_, r: sim.white(m_, N, r), "c0": lambda m_, r: lin_from_psd(m_, N, r, c0f),
                    "c1": lambda m_, r: lin_from_psd(m_, N, r, c1f)},
        }
        res = {"N": N, "R_db": R_db, "wall_db_circ": float(10 * np.log10(g)), "wall_db_lin": float(10 * np.log10(gf)),
               "n_cal": n_cal, "n_te": n_te}
        for mode, G in modes.items():
            cal = [scores(fns, lambda m_: G[s](m_, rcal), n_cal, chunk) for s in ("white", "c0", "c1")]
            thr = {k: float(max(np.quantile(c[k], 1 - ALPHA) for c in cal)) for k in fns}
            pf = {}
            for s in ("white", "c0", "c1"):
                sc = scores(fns, lambda m_: G[s](m_, rte), n_te, chunk)
                pf[s] = {k: float(np.mean(sc[k] > thr[k])) for k in fns}
            pd = {}
            for d in deltas:
                a = np.sqrt(G["wall"] * 10 ** (d / 10))
                for h1 in ("sig", "gsur"):
                    src = (lambda m_: sim.modulated(m_, N, rte)) if h1 == "sig" else (lambda m_: sim.surrogate(m_, N, rte))
                    r = {}
                    for nz in ("white", "c1"):
                        sc = scores(fns, lambda m_: src(m_) * a + G[nz](m_, rte), n_te, chunk)
                        r[nz] = {k: float(np.mean(sc[k] > thr[k])) for k in fns}
                    pd[f"{h1}_{d}"] = {k: {"white": r["white"][k], "c1": r["c1"][k], "worst": min(r["white"][k], r["c1"][k])} for k in fns}
            res[mode] = {"thr": thr, "Pf": pf, "Pd": pd}
            print(f"seed={seed} u={R_db} mode={mode} ({time.time()-t0:.0f}s)", flush=True)
            for k in fns:
                print(f"   {k:20s} Pfmax {max(pf[s][k] for s in pf):.3f} | " + " ".join(
                    f"{h}{d}:{pd[f'{h}_{d}'][k]['worst']:.3f}" for h in ("sig", "gsur") for d in deltas), flush=True)
        out[key] = res
        with open(path + ".tmp", "w") as fh:
            json.dump(out, fh, indent=1)
        os.replace(path + ".tmp", path)


if __name__ == "__main__":
    for s in ([int(sys.argv[1])] if len(sys.argv) > 1 else [0, 1, 2]):
        run_seed(s)
