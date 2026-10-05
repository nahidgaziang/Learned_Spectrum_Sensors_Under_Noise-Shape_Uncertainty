"""Gaussianity of recorded RTL-SDR noise (found while building the real-signal surrogate test, the project plan T6).
Per recording (first 256 windows skipped, N = 128 windows, per-window DC removed, at most 3000 windows):
  kappa = mean over windows of E|x|^4 / (E|x|^2)^2 (complex circular Gaussian: 2; a finite-N simulation gives the
  reference value), fraction of windows with kappa > the 99th percentile of simulated Gaussian windows, peak-to-rms
  ratio of the samples, number of distinct I levels (ADC quantisation).
Also: Spearman correlation of per-recording kappa with the real-noise P_f of the white-trained networks (v2 file; v3 if
present), to see whether non-Gaussianity rather than spectral shape could drive false alarms.
Writes RESULTS/noise_gaussianity.json.   usage: py -3.12 noise_gaussianity.py"""
import glob, json, os, sys
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_noise_eval as RN, sim

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N, SKIP = 128, 256


def kap(w):
    p = np.abs(w) ** 2
    return np.mean(p ** 2, 1) / np.mean(p, 1) ** 2


rng = np.random.default_rng(11)
g = sim.white(50000, N, rng)
g = g - g.mean(1, keepdims=True)
kg = kap(g)
q99 = float(np.quantile(kg, 0.99))
out = {"gaussian_ref": {"mean": float(kg.mean()), "sd": float(kg.std()), "q99": q99}, "receivers": {}}
for rx, d in [("hf_composite", os.path.join(ROOT, "DATA", "real_noise_hf")), ("trevtron_R828D", os.path.join(ROOT, "DATA", "trevtron", "noise"))]:
    rec = {}
    for fp in sorted(glob.glob(os.path.join(d, "*.npy"))):
        iq, meta = RN.load(fp)
        if not meta.get("sample_rate"):
            continue
        w = RN.windows(iq, N)[SKIP:SKIP + 3000]
        k = kap(w)
        x = iq[SKIP * N:].astype(np.complex128)
        rec[os.path.basename(fp)] = {"kappa_mean": float(k.mean()), "kappa_sd": float(k.std()), "frac_above_q99": float(np.mean(k > q99)),
                                     "peak_to_rms": float(np.abs(x.real).max() / x.real.std()),
                                     "n_levels_I": int(len(np.unique(np.round(x.real, 6))))}
    if not rec:
        continue
    km = np.array([v["kappa_mean"] for v in rec.values()])
    fr = np.array([v["frac_above_q99"] for v in rec.values()])
    summ = {"n": len(rec), "kappa_mean_median": float(np.median(km)), "kappa_mean_range": [float(km.min()), float(km.max())],
            "frac_above_q99_median": float(np.median(fr))}
    for ver in ["v2", "v3"]:
        f = os.path.join(ROOT, "RESULTS", f"real_noise_{ver}_N128.json") if rx == "hf_composite" else \
            os.path.join(ROOT, "RESULTS", "real_noise_trevtron_N128.json")
        if not os.path.exists(f) or (rx != "hf_composite" and ver == "v2"):
            continue
        RR = json.load(open(f))["recordings"]
        names = [n for n in rec if n in RR and not RR[n]["shape"]["narrowband"]]       # clean subset, as in the paper
        pf = np.array([np.mean([RR[n]["Pf"][k] for k in RR[n]["Pf"] if k.startswith(("CLDNN-WHITE", "GapCNN-WHITE"))]) for n in names])
        pfo = np.array([np.mean([RR[n]["Pf_wbig"][k] for k in RR[n]["Pf_wbig"] if k.startswith(("CLDNN-WHITE", "GapCNN-WHITE"))]) for n in names])
        roll = np.abs(np.array([RR[n]["shape"]["edge_rolloff_db"] for n in names]))
        kk = np.array([rec[n]["kappa_mean"] for n in names])

        def partial(a, b, c):        # Spearman partial correlation of a and b given c (rank-based)
            from scipy.stats import rankdata
            ra, rb, rc = rankdata(a), rankdata(b), rankdata(c)
            res = lambda y: y - np.polyval(np.polyfit(rc, y, 1), rc)
            return float(np.corrcoef(res(ra), res(rb))[0, 1])
        summ[f"n_clean_{ver}"] = len(names)
        summ[f"spearman_kappa_vs_Pf_{ver}"] = float(spearmanr(kk, pf)[0])
        summ[f"spearman_rolloff_vs_Pf_{ver}"] = float(spearmanr(roll, pf)[0])
        summ[f"spearman_kappa_vs_rolloff_{ver}"] = float(spearmanr(kk, roll)[0])
        summ[f"partial_rolloff_vs_Pf_given_kappa_{ver}"] = partial(roll, pf, kk)
        summ[f"partial_kappa_vs_Pf_given_rolloff_{ver}"] = partial(kk, pf, roll)
        summ[f"spearman_kappa_vs_Pf_whitened_{ver}"] = float(spearmanr(kk, pfo)[0])
    out["receivers"][rx] = {"summary": summ, "recordings": rec}
    print(rx, json.dumps(summ), flush=True)
print("gaussian reference", out["gaussian_ref"])
with open(os.path.join(ROOT, "RESULTS", "noise_gaussianity.json"), "w") as fh:
    json.dump(out, fh, indent=1)
