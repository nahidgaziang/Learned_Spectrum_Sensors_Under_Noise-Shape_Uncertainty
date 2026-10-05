"""Step 5 of the project plan (U4): hardware wall budget. On the SAME recorded RTL-SDR receivers and the SAME
block pairs, how large is the noise-POWER uncertainty that an energy detector faces, and the noise-SHAPE uncertainty
that a power-normalised (scale-invariant) detector faces, and which SNR walls do they imply?

Time scales are aligned (this replaces the preliminary probe novelty_probe_power_vs_shape.py):
  within a capture  : blocks A = windows 1024..2047 and B = 2048..3071 (128-sample windows, after the start-up
                      transient); u_P = |10 log10(P_B / P_A)|, u_S = peak-to-peak ratio of the two smooth shapes
                      (empirical_uncertainty.fit_shape / u_db, identical to the paper's "steady drift");
  between captures  : segment = windows 1024..3023 of each recording, all pairs at the same centre frequency
                      (the paper's "same-frequency" pairs); u_P from the segment powers, u_S from the segment shapes.
Walls: energy detector gamma_ED(u_P) = (R-1)/sqrt(R) (Tandra-Sahai, peak-to-peak); scale-invariant detectors
gamma_w(u_S) for the RRC signal (exact, Theorem 3). Paired per-block/per-pair gain = 10 log10(gamma_ED / gamma_w).
ASSUMPTION: the datasets do not record gain settings; power differences are attributed to the noise level under fixed
gain within a capture and between captures at one frequency (an AGC or gain change would inflate u_P).
Recordings: as in empirical_uncertainty.py (live RTL recordings without narrowband interference).
Writes RESULTS/hardware_wall_budget.json.   usage: py -3.12 hardware_wall_budget.py"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_noise_eval as RN, sim, wall as W
from empirical_uncertainty import fit_shape, u_db, bootstrap_ci

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 128


def summ(x):
    x = np.asarray(x, float)
    return {"n": int(len(x)), "median": float(np.median(x)), "p90": float(np.quantile(x, 0.9)), "max": float(np.max(x)),
            "median_ci": bootstrap_ci(x, np.median), "p90_ci": bootstrap_ci(x, lambda a: np.quantile(a, 0.9))}


def receiver(json_name, ddir, p):
    V = json.load(open(os.path.join(ROOT, "RESULTS", json_name)))["recordings"]
    recs = {k: v for k, v in V.items() if v["group"] == "live_rtl" and not v["shape"]["narrowband"]}
    within_P, within_S, seg = [], [], {}
    for name, v in recs.items():
        iq, _ = RN.load(os.path.join(ddir, name))
        w = RN.windows(iq, N)
        if len(w) >= 3072:
            A, B = w[1024:2048], w[2048:3072]
            within_P.append(abs(10 * np.log10(np.mean(np.abs(B) ** 2) / np.mean(np.abs(A) ** 2))))
            within_S.append(u_db(fit_shape(B), fit_shape(A)))
        s = w[1024:3024] if len(w) >= 3024 else w
        seg[name] = (str(v["meta"].get("center_freq")), float(np.mean(np.abs(s) ** 2)), fit_shape(s))
    pair_P, pair_S = [], []
    names = list(seg)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            (fa, pa, sa), (fb, pb, sb) = seg[names[i]], seg[names[j]]
            if fa == fb:
                pair_P.append(abs(10 * np.log10(pa / pb)))
                pair_S.append(u_db(sa, sb))
    ed = lambda u: 10 * np.log10(sim.ed_power_wall(u)) if u > 0 else -np.inf
    sw = lambda u: 10 * np.log10(W.wall(p, u)) if u > 0 else -np.inf
    out = {"n_recordings": len(recs), "n_within": len(within_P), "n_pairs": len(pair_P),
           "n_frequencies": len({v[0] for v in seg.values()})}
    for lab, P_, S_ in (("within_capture", within_P, within_S), ("same_frequency_pairs", pair_P, pair_S)):
        gain = [ed(a) - sw(b) for a, b in zip(P_, S_)]
        sp, ss = summ(P_), summ(S_)
        out[lab] = {"u_power_db": sp, "u_shape_db": ss,
                    "wall_ed_db": {q: float(ed(sp[q])) for q in ("median", "p90", "max")},
                    "wall_shape_db": {q: float(sw(ss[q])) for q in ("median", "p90", "max")},
                    "paired_gain_db": {"median": float(np.median(gain)), "p10": float(np.quantile(gain, 0.1)),
                                       "p90": float(np.quantile(gain, 0.9)), "frac_positive": float(np.mean(np.array(gain) > 0))},
                    "frac_power_gt_shape": float(np.mean(np.array(P_) > np.array(S_)))}
    return out


def main():
    p = W.signal_psd(N, n=8000)
    res = {"_note": "U4 hardware wall budget (hardware_wall_budget.py); fixed receiver gain assumed (not recorded).",
           "R820T2 (Receiver 1)": receiver("real_noise_v3_N128.json", os.path.join(ROOT, "DATA", "real_noise_hf"), p),
           "R828D (Receiver 2)": receiver("real_noise_trevtron_N128.json", os.path.join(ROOT, "DATA", "trevtron", "noise"), p)}
    json.dump(res, open(os.path.join(ROOT, "RESULTS", "hardware_wall_budget.json"), "w"), indent=1)
    for rx, r in res.items():
        if rx.startswith("_"):
            continue
        print(rx, f"recordings {r['n_recordings']}, within {r['n_within']}, pairs {r['n_pairs']}, freqs {r['n_frequencies']}")
        for lab in ("within_capture", "same_frequency_pairs"):
            v = r[lab]
            print(f"  {lab}: u_P med {v['u_power_db']['median']:.2f} p90 {v['u_power_db']['p90']:.2f} | u_S med {v['u_shape_db']['median']:.2f} "
                  f"p90 {v['u_shape_db']['p90']:.2f} | walls ED med {v['wall_ed_db']['median']:.1f} p90 {v['wall_ed_db']['p90']:.1f} ; "
                  f"shape med {v['wall_shape_db']['median']:.1f} p90 {v['wall_shape_db']['p90']:.1f} | paired gain med {v['paired_gain_db']['median']:.1f} "
                  f"[p10 {v['paired_gain_db']['p10']:.1f}, p90 {v['paired_gain_db']['p90']:.1f}], positive {v['paired_gain_db']['frac_positive']:.2f}")


if __name__ == "__main__":
    main()
