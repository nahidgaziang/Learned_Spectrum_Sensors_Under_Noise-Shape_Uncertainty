"""Sensitivity of the main findings to the false-alarm target alpha (1 %, 5 %, 10 %), from saved raw scores
(RESULTS/scores/<id>.npz; no model is re-run).

Protocol: the threshold is the (1-alpha)-quantile of the first 2000 white-noise test scores; P_f on white noise is
measured on the remaining white draws, P_f on every other shape and P_d on all of their draws (independent of the
threshold draws). The surrogate gap is the mean of Pd_signal - Pd_surrogate over SNRs with 0.1 < Pd_signal < 0.9.
Writes RESULTS/alpha_sensitivity.json and TABLES/alpha_sensitivity.csv."""
import glob, json, os, re
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALPHAS = (0.01, 0.05, 0.10)
SHAPES = ["white+1dB", "lpf0.8", "rtl3dB", "ar+0.1", "ar-0.1", "pink"]


def one(path, alpha):
    z = np.load(path)
    w = z["h0_white"]
    thr = np.quantile(w[:2000], 1 - alpha)
    r = {"Pf_white": float(np.mean(w[2000:] > thr))}
    for s in SHAPES:
        if f"h0_{s}" in z.files:
            r[f"Pf_{s}"] = float(np.mean(z[f"h0_{s}"] > thr))
    snrs = sorted({int(m.group(1)) for k in z.files for m in [re.match(r"h1_white_(-?\d+)$", k)] if m})
    sig = {d: float(np.mean(z[f"h1_white_{d}"] > thr)) for d in snrs}
    sur = {d: float(np.mean(z[f"sur_white_{d}"] > thr)) for d in snrs if f"sur_white_{d}" in z.files}
    mid = [d for d in snrs if 0.1 < sig[d] < 0.9 and d in sur]
    r["surgap"] = float(np.mean([sig[d] - sur[d] for d in mid])) if mid else float("nan")
    r["Pd_-10"] = sig.get(-10, float("nan"))
    for s in ["pink", "lpf0.8", "rtl3dB"]:
        for d in (-10, 0):
            k = f"h1_{s}_{d}"
            if k in z.files:
                r[f"PdIn_{s}_{d}"] = float(np.mean(z[k] > thr))
                # colour-calibrated detection: threshold from that colour's own noise scores (P_f = alpha on it).
                # Low values here mean the network cannot separate signal+colour from colour (information lost),
                # not merely a mis-set threshold.
                tc = np.quantile(z[f"h0_{s}"], 1 - alpha)
                r[f"PdCal_{s}_{d}"] = float(np.mean(z[k] > tc))
    return r


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "RESULTS", "scores", "*.npz"))):
        rid = os.path.basename(f)[:-4]
        rec = os.path.join(ROOT, "RESULTS", "runs", rid + ".json")
        if not os.path.exists(rec):
            continue
        c = json.load(open(rec))["config"]
        if c.get("tag") or c["N"] != 128 or c["sps"] != 8:
            continue
        for a in ALPHAS:
            r = one(f, a)
            r.update({"id": rid, "kind": c["kind"], "model": c["detector"] if c["kind"] == "classical" else c["arch"],
                      "recipe": c.get("recipe", "-") if c["kind"] == "learned" else "-", "seed": c["seed"], "alpha": a})
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "TABLES", "alpha_sensitivity.csv"), index=False)
    g = df.groupby(["kind", "model", "recipe", "alpha"]).mean(numeric_only=True).reset_index()
    json.dump(g.to_dict(orient="records"), open(os.path.join(ROOT, "RESULTS", "alpha_sensitivity.json"), "w"), indent=1)
    pd.set_option("display.width", 220)
    cols = ["model", "recipe", "alpha", "Pf_white", "Pf_white+1dB", "Pf_lpf0.8", "Pf_rtl3dB", "Pf_pink", "surgap", "Pd_-10",
            "PdIn_pink_-10", "PdCal_pink_-10", "PdIn_lpf0.8_-10", "PdCal_lpf0.8_-10", "PdCal_pink_0"]
    print(g[[c for c in cols if c in g]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
