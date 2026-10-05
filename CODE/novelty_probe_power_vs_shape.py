"""Probe P4 (2026-10-05, PRELIMINARY): noise-POWER uncertainty vs noise-SHAPE uncertainty on the same recorded RTL-SDR
receivers, and the SNR walls each implies (energy detector vs scale-invariant detector of the RRC signal, 8 sps).
Assumption (to be stated if used): within one capture, and between captures at the same centre frequency on the same
day, receiver gain settings are unchanged (the datasets do not record gain), so power ratios reflect noise-level drift.
Power uncertainty u_P = |10 log10(P_a / P_b)| (peak-to-peak between two estimates); shape uncertainty from
RESULTS/empirical_uncertainty.json (same scenarios). Walls: ED = sim.ed_power_wall(u_P); shape = wall.wall(p, u_S).
Writes RESULTS/novelty_probe_power_vs_shape.json."""
import collections, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_noise_eval as RN, sim, wall as W

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N, SKIP = 128, 256


def rec_stats(d):
    out = {"within": [], "groups": collections.defaultdict(list)}
    for fp in sorted(glob.glob(os.path.join(d, "*.npy"))):
        iq, meta = RN.load(fp)
        if not meta.get("sample_rate"):
            continue
        sh, _ = RN.psd_shape(iq)
        if sh["narrowband"]:
            continue
        w = RN.windows(iq, N)[SKIP:]
        p = np.mean(np.abs(w) ** 2, 1)
        blocks = np.array_split(p, 4)
        bp = [b.mean() for b in blocks]
        out["within"].append(10 * np.log10(max(bp) / min(bp)))
        day = str(meta.get("timestamp", ""))[:8]
        out["groups"][(meta.get("center_freq"), day)].append(p.mean())
    pair = []
    for g, ps in out["groups"].items():
        ps = np.array(ps)
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                pair.append(abs(10 * np.log10(ps[i] / ps[j])))
    return np.array(out["within"]), np.array(pair), len(out["groups"])


res = {}
p = W.signal_psd(1024, sps=8)
E = json.load(open(os.path.join(ROOT, "RESULTS", "empirical_uncertainty.json")))
for name, d, ek in [("R820T2 (HF composite)", os.path.join(ROOT, "DATA", "real_noise_hf"), "receiver_1_r820t2"),
                    ("R828D (TrevTron)", os.path.join(ROOT, "DATA", "trevtron", "noise"), "receiver_2_r828d")]:
    within, pair, ng = rec_stats(d)
    sh_within = E[ek]["steady_drift"]
    sh_sess = E[ek]["session"]
    r = {"n_within": int(len(within)), "n_pairs": int(len(pair)), "n_groups": ng}
    for lab, arr in [("power_within_capture", within), ("power_same_freq_same_day", pair)]:
        if len(arr):
            r[lab] = {"median_db": float(np.median(arr)), "p90_db": float(np.quantile(arr, 0.9))}
    r["shape_within_capture"] = {"median_db": sh_within["median"], "p90_db": sh_within["p90"]}
    r["shape_same_freq_session"] = {"median_db": sh_sess["median"], "p90_db": sh_sess["p90"]}
    walls = {}
    for lab in ("power_within_capture", "power_same_freq_same_day"):
        if lab in r:
            walls[lab] = {q: float(10 * np.log10(sim.ed_power_wall(r[lab][q]))) for q in ("median_db", "p90_db")}
    for lab in ("shape_within_capture", "shape_same_freq_session"):
        walls[lab] = {q: float(10 * np.log10(W.wall(p, r[lab][q]))) for q in ("median_db", "p90_db")}
    r["walls_db"] = walls
    res[name] = r
    print(name, json.dumps(r, indent=1))
with open(os.path.join(ROOT, "RESULTS", "novelty_probe_power_vs_shape.json"), "w") as fh:
    json.dump(res, fh, indent=1)
