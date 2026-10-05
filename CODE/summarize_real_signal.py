"""Summary of real_signal_surrogate.py output: pooled Pd curves (signal vs surrogate), surrogate gap per receiver x class x
detector (mean over SNRs with 0.1 < pooled Pd_sig < 0.9, the paper's definition), and a 95 % bootstrap CI that resamples
CAPTURES (B = 2000). Learned detectors are also grouped by family (arch-recipe, mean over seeds).
usage: py -3.12 summarize_real_signal.py [nonets]   -> RESULTS/real_signal_summary[_nonets].json, TABLES/real_signal_gaps[_nonets].csv"""
import json, os, re, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = 2000


def gap_of(sig, sur, n):
    pd_s, pd_r = sig / n, sur / n
    m = (pd_s > 0.1) & (pd_s < 0.9)
    return (float(np.mean(pd_s[m] - pd_r[m])) if m.any() else float("nan")), int(m.sum())


def main(tag=""):
    R = json.load(open(os.path.join(ROOT, "RESULTS", f"real_signal_surrogate{tag}.json")))
    snrs = R["snrs"]
    rng = np.random.default_rng(5)
    out, rows = {"snrs": snrs, "B": B, "results": {}}, []
    for rx, RX in R["receivers"].items():
        for cls, C in RX["classes"].items():
            caps = list(C["captures"].values())
            if not caps:
                continue
            dets = list(caps[0]["counts"].keys())
            n = np.array([c["n"] for c in caps], float)
            fam = {}
            for d in dets:
                fam.setdefault(re.sub(r"-s\d+$", "", d), []).append(d)
            surs = [k for k in ("sur", "sur_win") if k in caps[0]["counts"][dets[0]]]
            for (fname, members), sk in [(fm, s) for fm in fam.items() for s in surs]:
                S = np.array([[c["counts"][d]["sig"] for d in members] for c in caps], float)   # caps x seeds x snr
                U = np.array([[c["counts"][d][sk] for d in members] for c in caps], float)
                def stat(idx):
                    g = [gap_of(S[idx, j].sum(0), U[idx, j].sum(0), n[idx].sum())[0] for j in range(len(members))]
                    return float(np.nanmean(g)) if not np.all(np.isnan(g)) else float("nan")
                full = np.arange(len(caps))
                g0 = stat(full)
                n_snr = [gap_of(S[:, j].sum(0), U[:, j].sum(0), n.sum())[1] for j in range(len(members))]
                bs = np.array([stat(rng.integers(0, len(caps), len(caps))) for _ in range(B)])
                bs = bs[~np.isnan(bs)]
                lo, hi = (np.quantile(bs, [0.025, 0.975]) if len(bs) > 100 else (np.nan, np.nan))
                pd_sig = (S.sum(0) / n.sum()).mean(0)
                pd_sur = (U.sum(0) / n.sum()).mean(0)
                key = f"{rx}|{cls}|{fname}|{sk}"
                out["results"][key] = {"gap": g0, "ci": [float(lo), float(hi)], "n_snr_in_range": n_snr, "seeds": len(members),
                                       "n_captures": len(caps), "n_windows": int(n.sum()),
                                       "pd_sig": pd_sig.round(4).tolist(), "pd_sur": pd_sur.round(4).tolist()}
                rows.append((rx, cls, fname + ("" if sk == "sur" else "[win]"), len(members), len(caps), g0, lo, hi))
    with open(os.path.join(ROOT, "RESULTS", f"real_signal_summary{tag}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    with open(os.path.join(ROOT, "TABLES", f"real_signal_gaps{tag}.csv"), "w") as fh:
        fh.write("receiver,class,detector,seeds,captures,gap,ci_lo,ci_hi\n")
        for r in rows:
            fh.write(",".join(map(str, r[:5])) + f",{r[5]:.4f},{r[6]:.4f},{r[7]:.4f}\n")
    for r in rows:
        print(f"{r[0]:15s} {r[1]:13s} {r[2]:18s} seeds {r[3]} caps {r[4]:3d}  gap {100*r[5]:+6.1f}  [{100*r[6]:+6.1f}, {100*r[7]:+6.1f}]")


if __name__ == "__main__":
    main("_nonets" if len(sys.argv) > 1 and sys.argv[1] == "nonets" else "")
