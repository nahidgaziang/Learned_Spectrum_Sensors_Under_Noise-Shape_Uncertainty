"""Step 2 of the project plan (U1 evidence): learned spectrum sensors vs the locally most powerful invariant
(LMPI) test on IDENTICAL test windows, for every saved model (all architectures and seeds, the DetectNet replica,
GapCNN-4X, the coloured/surrogate recipes, N = 128/512/2048/8192, occupancy sweeps) and the classical detectors.

LMPI statistic (white noise, signal PSD p on the N-point grid, circulant approximation of x^H T(p) x / x^H x):
    T(x) = sum_k p_k |X_k|^2 / sum_k |X_k|^2 .
Replay: evaluate.evaluate draws every test window from sim.streams(10_000 + seed) in a fixed order and score functions
consume no randomness, so calling it with the LMPI and the run's own settings (N, sps, seed, beta, mods, signal,
evaluation sizes) reproduces the windows the saved scores refer to. A recording score function keeps the band-aggregated
normalised periodogram (128 contiguous frequency bands) of the windows needed for the template fits; the replayed LMPI
scores are asserted equal to evaluate's raw scores key by key.

Per model: Spearman rho and decision agreement with the LMPI and with the band ratio (same windows, nominal thresholds)
at the SNR where the LMPI's P_d is closest to 0.5 and at -10 dB; the same on surrogate windows and on LP .8 noise;
implicit spectral template (ridge regression of scores on the band-aggregated normalised periodogram over pooled H0
white / H1 white / surrogate windows) with cosine to the LMPI template p and to the band-ratio template, best exponent
a in the family p^a, and best design SNR in the most-powerful-invariant family g p / (1 + g p).
Per (N, seed): reference agreement between networks. Per N: the LMPI's full profile (P_f over all 20 shapes, P_d,
surrogate gap) next to the white-trained GAP CNN's. Low-SNR check: P_d at -20/-17 dB of the LMPI vs every detector.
Per architecture: surrogate gap vs SNR from the run records (STFT CNN check).
Writes RESULTS/lmpi_equivalence.json.   usage: py -3.12 lmpi_equivalence.py [maxN]"""
import glob, inspect, itertools, json, os, sys, time
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as E, sim, wall as W

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "RESULTS")
OUT = os.path.join(RES, "lmpi_equivalence.json")
NB = 128
H1_SHAPES = inspect.signature(E.evaluate).parameters["h1_shapes"].default
KEEP_H0 = ("h0_white", "h0_lpf0.8", "h0_rtl3dB", "h0_pink")
SKIP_TAGS = ("smoke", "amp_check")
_PSD = {}


def psd(N, sps):
    """Signal PSD (unit mean, fft order). For N > 2048 the 2048-point estimate is interpolated (smooth spectrum)."""
    if (N, sps) not in _PSD:
        if N <= 2048:
            p = W.signal_psd(N, sps=sps)
        else:
            p0 = W.signal_psd(2048, sps=sps, n=8000)
            f0 = np.fft.fftshift(np.fft.fftfreq(2048))
            f = np.fft.fftfreq(N)
            p = np.interp(f, np.concatenate([f0 - 1, f0, f0 + 1]), np.tile(np.fft.fftshift(p0), 3))
            p = p / p.mean()
        _PSD[(N, sps)] = p
    return _PSD[(N, sps)]


def band_agg(v, N):
    """fft-ordered per-bin vector(s) -> 128 contiguous bands (ascending frequency)."""
    order = np.argsort(np.fft.fftfreq(N), kind="stable")
    v = np.asarray(v)[..., order]
    return v.reshape(v.shape[:-1] + (NB, N // NB)).sum(-1)


def key_plan(sz):
    n_cal, n_cs, n_h0, n_h1 = sz.get("n_cal", 20000), sz.get("n_cal_shape", 4000), sz.get("n_h0", 4000), sz.get("n_h1", 2000)
    plan = [("cal_white", n_cal)] + [(f"cal_{s}", n_cs) for s in E.all_shapes()] + [(f"h0_{s}", n_h0) for s in E.all_shapes()]
    for d in E.SNRS:
        plan += [(f"h1_white_{d}", n_h0), (f"sur_white_{d}", n_h0)] + [(f"h1_{sh}_{d}", n_h1) for sh in H1_SHAPES if sh != "white"]
    return plan


def replay(N, sps, seed, beta, mods, signal, sz):
    p = psd(N, sps)
    w = p / p.sum()
    plan = key_plan(sz)
    keep = set(KEEP_H0) | {f"h1_white_{d}" for d in E.SNRS} | {f"sur_white_{d}" for d in E.SNRS}
    st = {"i": 0, "left": plan[0][1]}
    Xb, sc = {}, {}

    def lmpi(x):
        key = plan[st["i"]][0]
        P = np.abs(np.fft.fft(x, axis=1)) ** 2
        tot = P.sum(1)
        s = (P @ w) / tot
        if key in keep:
            Xb.setdefault(key, []).append((band_agg(P, N) / tot[:, None]).astype(np.float32))
            sc.setdefault(key, []).append(s)
        st["left"] -= len(x)
        assert st["left"] >= 0, "chunk crosses a key boundary"
        if st["left"] == 0 and st["i"] + 1 < len(plan):
            st["i"] += 1
            st["left"] = plan[st["i"]][1]
        return s

    res, raw = E.evaluate(lmpi, N, sps, seed, beta=beta, mods=mods, signal=signal, **sz)
    assert st["i"] == len(plan) - 1 and st["left"] == 0, "window bookkeeping mismatch"
    Xb = {k: np.concatenate(v) for k, v in Xb.items()}
    for k, v in sc.items():
        assert np.allclose(np.concatenate(v), raw[k], rtol=1e-4, atol=1e-6), f"replay mismatch {k}"
    return res, raw, Xb


def band_template(N, sps=8, beta=0.35):
    """Band-ratio template: occupied band |f| <= (1 + beta) / (2 sps) (= 0.0844 at sps 8, as in detectors.band_ratio)."""
    f = np.fft.fftfreq(N)
    return band_agg((np.abs(f) <= (1 + beta) / (2 * sps)).astype(float), N)


def fit_template(Xs, y):
    Xc = Xs - Xs.mean(0)
    yc = y - y.mean()
    G = Xc.T @ Xc
    beta = np.linalg.solve(G + 1e-6 * np.trace(G) / G.shape[0] * np.eye(G.shape[0]), Xc.T @ yc)
    r2 = 1 - np.sum((yc - Xc @ beta) ** 2) / np.sum(yc ** 2)
    return beta - beta.mean(), float(r2)


def cos(u, v):
    u, v = u - u.mean(), v - v.mean()
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-300))


def family_fits(t, p_bins, N):
    out = {}
    a_grid = np.round(np.arange(0.05, 3.001, 0.05), 3)
    ca = [cos(t, band_agg(p_bins ** a, N)) for a in a_grid]
    out["best_a"], out["cos_best_a"] = float(a_grid[int(np.argmax(ca))]), float(max(ca))
    g_grid = np.arange(-40.0, 20.01, 0.5)
    pu = p_bins / p_bins.mean()
    cg = [cos(t, band_agg((10 ** (g / 10)) * pu / (1 + (10 ** (g / 10)) * pu), N)) for g in g_grid]
    out["best_design_snr_db"], out["cos_best_design"] = float(g_grid[int(np.argmax(cg))]), float(max(cg))
    out["cos_lmpi_limit"] = float(cg[0])
    return out


def runs():
    """Run records with saved scores (modulated signals; OFDM skipped: its LMPI template would need its own PSD)."""
    out = []
    for f in sorted(glob.glob(os.path.join(RES, "runs", "*.json"))):
        name = os.path.basename(f)[:-5]
        if not os.path.exists(os.path.join(RES, "scores", name + ".npz")) or any(t in name for t in SKIP_TAGS):
            continue
        r = json.load(open(f))
        c = r.get("config", {})
        if c.get("signal", "modulated") != "modulated" or "eval" not in r or "thresholds" not in r["eval"]:
            continue
        out.append((name, r))
    return out


def main(maxN=8192, merge=False):
    """merge=True: keep the existing JSON and replace only the groups replayed now (e.g. after new N = 128 models)."""
    t0 = time.time()
    allruns = [(n, r) for n, r in runs() if r["config"]["N"] <= maxN]
    prev = json.load(open(OUT)) if (merge and os.path.exists(OUT)) else None
    groups = {}
    for name, r in allruns:
        c = r["config"]
        sz = r.get("eval_sizes")
        sz = {} if sz in (None, "default") else sz
        key = (c["N"], c["sps"], c["seed"], c.get("beta", 0.35), tuple(c.get("mods") or sim.MODS), json.dumps(sz, sort_keys=True))
        groups.setdefault(key, []).append((name, r))
    out = {"_note": "U1 evidence (lmpi_equivalence.py); per-window comparisons on identical test windows.",
           "models": {}, "lmpi_profile": {}, "reference_agreement": {}, "gap_vs_snr": {}}
    if prev:
        for k in ("models", "lmpi_profile", "reference_agreement"):
            out[k].update(prev.get(k, {}))
    for gk, members in sorted(groups.items()):
        N, sps, seed, beta, mods, szj = gk
        sz = json.loads(szj)
        t1 = time.time()
        res, raw, Xb = replay(N, sps, seed, beta, mods, "modulated", sz)
        thr_l = res["thresholds"]["nominal"]
        pd_l = {d: float(np.mean(raw[f"h1_white_{d}"] > thr_l)) for d in E.SNRS}
        dstar = min(E.SNRS, key=lambda d: abs(pd_l[d] - 0.5))
        p_bins = psd(N, sps)
        pb, bt = band_agg(p_bins, N), band_template(N, sps, beta)
        gid = f"N{N}_sps{sps}_s{seed}" + ("" if len(mods) == len(sim.MODS) else "_" + "+".join(mods))
        out["lmpi_profile"][gid] = {"Pf": {k: v["nominal"] for k, v in res["Pf"].items()},
                                    "Pd": {str(d): pd_l[d] for d in E.SNRS},
                                    "Pd_surrogate": {k: v["nominal"] for k, v in res["Pd_surrogate"].items()},
                                    "surrogate_gap": res["surrogate_gap"], "dstar": dstar, "n_models": len(members)}
        pool_keys = ["h0_white", f"h1_white_{dstar}", f"sur_white_{dstar}"]
        Xs = np.concatenate([Xb[k] for k in pool_keys]).astype(np.float64)
        dec = {"LMPI": {k: raw[k] > thr_l for k in raw}}
        for name, r in members:
            z = np.load(os.path.join(RES, "scores", name + ".npz"))
            thr = r["eval"]["thresholds"]["nominal"]
            m = {"group": gid, "dstar": dstar, "config": {k: r["config"].get(k) for k in ("kind", "arch", "detector", "recipe", "N", "sps", "seed", "tag")}}
            for lab, d in (("dstar", dstar), ("m10", -10)):
                ks, ku = f"h1_white_{d}", f"sur_white_{d}"
                m[lab] = {"snr": d,
                          "rho_sig": float(stats.spearmanr(z[ks], raw[ks])[0]),
                          "rho_sur": float(stats.spearmanr(z[ku], raw[ku])[0]),
                          "agree_sig": float(np.mean((z[ks] > thr) == (raw[ks] > thr_l))),
                          "pd_model": float(np.mean(z[ks] > thr)), "pd_lmpi": pd_l[d],
                          "pd_sur_model": float(np.mean(z[ku] > thr)), "pd_sur_lmpi": float(np.mean(raw[ku] > thr_l))}
            for k in ("h0_lpf0.8", "h0_rtl3dB"):
                m[k] = {"rho": float(stats.spearmanr(z[k], raw[k])[0]), "pf_model": float(np.mean(z[k] > thr)),
                        "pf_lmpi": float(np.mean(raw[k] > thr_l))}
            m["pd_low"] = {str(d): {"model": float(np.mean(z[f"h1_white_{d}"] > thr)), "lmpi": pd_l[d]} for d in (-20, -17, -14)}
            y = np.concatenate([np.asarray(z[k], float) for k in pool_keys])
            t, r2 = fit_template(Xs, y)
            m["template"] = {"r2_linear": r2, "cos_p": cos(t, pb), "cos_band": cos(t, bt), **family_fits(t, p_bins, N)}
            m["template_vec"] = t.tolist()
            out["models"][name] = m
            dec[name] = {k: z[k] > thr for k in (f"h1_white_{dstar}", f"h1_white_-10")}
            dec[name]["_score"] = z[f"h1_white_{dstar}"]
        # reference agreement among learned WHITE models of this group (untagged) and LMPI/BAND
        learned = [n for n, r in members if r["config"].get("kind") == "learned" and r["config"].get("recipe") == "WHITE"
                   and not r["config"].get("tag")]
        band = [n for n, r in members if r["config"].get("detector") == "BAND"]
        ks = f"h1_white_{dstar}"
        pairs = {}
        for a, b in itertools.combinations(learned + band, 2):
            pairs[f"{a}|{b}"] = {"rho": float(stats.spearmanr(dec[a]["_score"], dec[b]["_score"])[0]),
                                 "agree": float(np.mean(dec[a][ks] == dec[b][ks]))}
        for a in learned + band:
            pairs[f"{a}|LMPI"] = {"rho": float(stats.spearmanr(dec[a]["_score"], raw[ks])[0]),
                                  "agree": float(np.mean(dec[a][ks] == dec["LMPI"][ks]))}
        out["reference_agreement"][gid] = {"snr": dstar, "pairs": pairs}
        out["lmpi_profile"][gid]["template_p"] = (pb - pb.mean()).tolist()
        out["lmpi_profile"][gid]["template_band"] = (bt - bt.mean()).tolist()
        print(f"{gid}: {len(members)} models, d*={dstar} dB, LMPI Pd(d*)={pd_l[dstar]:.3f}, gap={res['surrogate_gap']}, "
              f"Pf LP.8={res['Pf']['lpf0.8']['nominal']:.3f}  ({time.time() - t1:.0f}s, total {time.time() - t0:.0f}s)", flush=True)
        with open(OUT + ".tmp", "w") as fh:
            json.dump(out, fh)
        os.replace(OUT + ".tmp", OUT)
    # surrogate gap vs SNR per architecture (run records, untagged WHITE N=128)
    for arch in ("CLDNN", "ResNet1D", "CovCNN", "STFTCNN", "GapCNN", "DetectNetExact", "Transformer"):
        rs = [r for n, r in allruns if r["config"].get("kind") == "learned" and r["config"].get("arch") == arch
              and r["config"].get("recipe") == "WHITE"
              and r["config"]["N"] == 128 and r["config"]["sps"] == 8 and (not r["config"].get("tag") or arch == "DetectNetExact")]
        if not rs:
            continue
        g = {}
        for d in E.SNRS:
            v = [r["eval"]["Pd"][str(d)]["white"]["nominal"] - r["eval"]["Pd_surrogate"][str(d)]["nominal"] for r in rs]
            g[str(d)] = {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None, "n": len(v),
                         "pd_mean": float(np.mean([r["eval"]["Pd"][str(d)]["white"]["nominal"] for r in rs]))}
        out["gap_vs_snr"][arch] = g
    out["runtime_s"] = time.time() - t0
    with open(OUT + ".tmp", "w") as fh:
        json.dump(out, fh)
    os.replace(OUT + ".tmp", OUT)
    print(f"done in {time.time() - t0:.0f} s -> {OUT}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8192, merge="--merge" in sys.argv)
