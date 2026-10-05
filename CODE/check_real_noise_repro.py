"""Reproducibility check of RESULTS/real_noise_v2_N128.json (does not modify it).

Replays real_noise_eval.main() for the first M recordings (same sorted order, same random stream: calibration
draws first, then one signal draw per recording) and compares every stored value (Pf, Pf_w16, Pf_wbig,
Pd_m10_nominal, Pd_m10_wbig for every detector) with the recomputed one.
Purpose: the original evaluation overlapped a burst of GPU-driver 'FECS logging' events (02:01-02:07, 2026-10-02).
Writes RESULTS/real_noise_v2_repro_check.json.   Usage: py -3.12 check_real_noise_repro.py [M]
"""
import glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim, real_noise_eval as R
from evaluate import whitener

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIELDS = ["Pf", "Pf_w16", "Pf_wbig", "Pd_m10_nominal", "Pd_m10_wbig"]


def main(M=40, N=128):
    ref = json.load(open(os.path.join(ROOT, "RESULTS", f"real_noise_v2_N{N}.json")))
    files = sorted(glob.glob(os.path.join(R.DATA, "*.npy")))[:M]
    rng = np.random.default_rng(77)
    fns = R.detectors_for(N)
    wcal = sim.white(20000, N, rng)
    wcal = wcal - wcal.mean(1, keepdims=True)
    thr = {k: float(np.quantile(f(wcal), 1 - R.ALPHA)) for k, f in fns.items()}
    thr_diff = max(abs(thr[k] - ref["thresholds"][k]) for k in ref["thresholds"])
    report = {"M": len(files), "max_threshold_abs_diff": thr_diff, "recordings": {}}
    worst = 0.0
    for fp in files:
        iq, meta = R.load(fp)
        w = R.windows(iq, N)
        kbig = 1024 if len(w) >= 2048 else len(w) // 2
        test = w[kbig: kbig + 2000]
        wh16 = whitener(w[:16], method="cos", n_cos=16)
        whbig = whitener(w[:kbig], method="cos", n_cos=16)
        pnoise = float(np.mean(np.abs(test) ** 2))
        sig = sim.modulated(len(test), N, rng) * np.sqrt(pnoise * 10 ** (-10 / 10))
        sig = sig - sig.mean(1, keepdims=True)
        new = {"Pf": {}, "Pf_w16": {}, "Pf_wbig": {}, "Pd_m10_nominal": {}, "Pd_m10_wbig": {}}
        for k, f in fns.items():
            new["Pf"][k] = float(np.mean(f(test) > thr[k]))
            new["Pf_w16"][k] = float(np.mean(f(wh16(test)) > thr[k]))
            new["Pf_wbig"][k] = float(np.mean(f(whbig(test)) > thr[k]))
            new["Pd_m10_nominal"][k] = float(np.mean(f(test + sig) > thr[k]))
            new["Pd_m10_wbig"][k] = float(np.mean(f(whbig(test + sig)) > thr[k]))
        old = ref["recordings"][os.path.basename(fp)]
        # compare only detectors present in the stored file (the detector list grew after v2)
        diffs = {fld: max(abs(new[fld][k] - old[fld][k]) for k in old[fld]) for fld in FIELDS}
        n_diff = sum(int(new[fld][k] != old[fld][k]) for fld in FIELDS for k in old[fld])
        worst = max(worst, max(diffs.values()))
        report["recordings"][os.path.basename(fp)] = {"max_abs_diff": diffs, "n_values_differing": n_diff,
                                                       "n_values": sum(len(new[fld]) for fld in FIELDS)}
        print(f"{os.path.basename(fp):28s} max|diff| " + " ".join(f"{fld}={d:.4f}" for fld, d in diffs.items())
              + f"  differing {n_diff}/{report['recordings'][os.path.basename(fp)]['n_values']}", flush=True)
    report["max_abs_diff_all"] = worst
    json.dump(report, open(os.path.join(ROOT, "RESULTS", "real_noise_v2_repro_check.json"), "w"), indent=1)
    print(f"threshold max |diff| {thr_diff:.2e}; values max |diff| over {len(files)} recordings: {worst:.4f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
