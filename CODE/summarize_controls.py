"""Summary of the verification controls (2026-10-05): wall_test_escape_controls_s*.json (noise synthesis circular vs
linear; signal vs linear Gaussian surrogate) and cyc_lag_ablation.json. Means over the seeds present.
Writes RESULTS/verification_controls_summary.json and prints a compact table."""
import glob, json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(ROOT, "RESULTS", "wall_test_escape_controls_s*.json")))]
out = {"n_seeds": len(F), "escape": {}, "cyc_lags": None}
for key in ["N8192_R1.0", "N8192_R2.0"]:
    D = [f[key] for f in F if key in f]
    if not D:
        continue
    dets = list(D[0]["circ"]["Pf"]["white"].keys())
    out["escape"][key] = {}
    print(f"{key}: seeds {len(D)}; wall circ {D[0]['wall_db_circ']:.2f} dB, lin {D[0]['wall_db_lin']:.2f} dB")
    for det in dets:
        row = {}
        for mode in ("circ", "lin"):
            row[f"{mode}_Pfmax"] = float(np.mean([max(d[mode]["Pf"][s][det] for s in d[mode]["Pf"]) for d in D]))
            for h in ("sig", "gsur"):
                for dl in (-2, 0, 2):
                    row[f"{mode}_{h}{dl}"] = float(np.mean([d[mode]["Pd"][f"{h}_{dl}"][det]["worst"] for d in D]))
        out["escape"][key][det] = row
        print(f"  {det:20s} " + " | ".join(
            f"{m}: Pf {row[m+'_Pfmax']:.3f} sig-2 {row[m+'_sig-2']:.3f} sig0 {row[m+'_sig0']:.3f} gsur-2 {row[m+'_gsur-2']:.3f} gsur0 {row[m+'_gsur0']:.3f} gsur+2 {row[m+'_gsur2']:.3f}"
            for m in ("circ", "lin")))
fc = os.path.join(ROOT, "RESULTS", "cyc_lag_ablation.json")
if os.path.exists(fc):
    out["cyc_lags"] = json.load(open(fc))["mean"]
    print("CYC lag ablation (N = 32768, mean over seeds):", out["cyc_lags"])
json.dump(out, open(os.path.join(ROOT, "RESULTS", "verification_controls_summary.json"), "w"), indent=1)
