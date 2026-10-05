"""Step 2 of the project plan (U1 evidence on real data): the LMPI test with the TRAINING template (the PSD of
the simulated RRC signal at 8 samples per symbol, i.e. what the white-trained networks were trained on) applied to the
real-signal surrogate pipeline of real_signal_surrogate.py, on exactly the same windows.

Replay: real_signal_surrogate.main draws every noise pool, capture selection, surrogate and mixture from one
np.random.default_rng(2027) in a fixed order; detectors consume no randomness. Running the same loop with only BAND and
LMPI reproduces the windows of the published run, which is checked by asserting that the BAND counts equal those in
RESULTS/real_signal_surrogate.json for every capture and SNR.
Writes RESULTS/real_signal_lmpi.json (per-capture counts + pooled curves and a comparison with the GAP CNN and the band
ratio from real_signal_summary.json).   usage: py -3.12 real_signal_lmpi.py"""
import glob, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_noise_eval as RN
import real_signal_surrogate as RS
import wall as W

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "RESULTS", "real_signal_lmpi.json")


def main():
    t0 = time.time()
    p = W.signal_psd(RS.N, sps=8)
    w = p / p.sum()

    def lmpi(x):
        P = np.abs(np.fft.fft(x, axis=1)) ** 2
        return (P @ w) / P.sum(1)

    fns = {"BAND": RN.classical_score_fn(dict(RN.DEFAULTS, kind="classical", detector="BAND", N=RS.N, sps=8)), "LMPI": lmpi}
    ref = json.load(open(os.path.join(ROOT, "RESULTS", "real_signal_surrogate.json")))
    rng = np.random.default_rng(2027)
    out = {"snrs": RS.SNRS, "receivers": {}}
    n_checked = 0
    for rx, (ndir, classes) in RS.SOURCES.items():
        if not glob.glob(os.path.join(ndir, "*.npy")):
            continue
        pool = RS.noise_pool(ndir, RS.N_CAL + RS.N_MIX, rng)
        cal, mix = pool[:RS.N_CAL], pool[RS.N_CAL:RS.N_CAL + RS.N_MIX]
        thr = {k: float(np.quantile(f(cal), 1 - RS.ALPHA)) for k, f in fns.items()}
        assert abs(thr["BAND"] - ref["receivers"][rx]["thresholds"]["BAND"]) < 1e-9, "calibration pool differs"
        R = {"thresholds": thr, "classes": {}}
        for cls, sdir in classes.items():
            files = sorted(glob.glob(os.path.join(sdir, "*.npy")))
            C = {"captures": {}}
            for fp in files:
                iq, meta = RN.load(fp)
                sel, info = RS.select(iq, rng)
                name = os.path.basename(fp)
                if sel is None:
                    continue
                s, idx = sel
                s = RS.unit(s)
                sur_w = RS.unit(RS.surrogate(s, rng))
                sur_c = RS.unit(RS.surrogate_ctx(iq, idx, rng))
                n = mix[rng.choice(len(mix), len(s) * len(RS.SNRS), replace=False)].reshape(len(RS.SNRS), len(s), RS.N)
                cnt = {k: {} for k in fns}
                for lab, z in [("sig", s), ("sur", sur_c), ("sur_win", sur_w)]:
                    x = np.concatenate([np.sqrt(10 ** (g / 10)) * z + n[i] for i, g in enumerate(RS.SNRS)])
                    for k, f in fns.items():
                        cnt[k][lab] = (f(x) > thr[k]).reshape(len(RS.SNRS), len(s)).sum(1).tolist()
                refc = ref["receivers"][rx]["classes"][cls]["captures"][name]["counts"]["BAND"]
                assert cnt["BAND"] == refc, f"replay mismatch {rx} {cls} {name}"
                n_checked += 1
                C["captures"][name] = {"n": int(len(s)), "counts": cnt}
            R["classes"][cls] = C
            print(f"{rx} {cls}: {len(C['captures'])} captures replayed ({time.time() - t0:.0f}s)", flush=True)
        out["receivers"][rx] = R
    out["n_captures_checked"] = n_checked
    # pooled curves and comparison with the published GAP CNN / BAND curves
    M = json.load(open(os.path.join(ROOT, "RESULTS", "real_signal_summary.json")))["results"]
    comp = {}
    for rx, R in out["receivers"].items():
        for cls, C in R["classes"].items():
            if not C["captures"]:
                continue
            tot = sum(c["n"] for c in C["captures"].values())
            pool = lambda det, lab: np.sum([c["counts"][det][lab] for c in C["captures"].values()], 0) / tot
            pl, pls = pool("LMPI", "sig"), pool("LMPI", "sur")
            pb = pool("BAND", "sig")
            key = lambda d: f"{rx}|{cls}|{d}|sur"
            cnn = np.array(M[key("GapCNN-WHITE")]["pd_sig"])
            band_pub = np.array(M[key("BAND")]["pd_sig"])
            sel = (pl > 0.1) & (pl < 0.9)
            comp[f"{rx}|{cls}"] = {"n_windows": int(tot), "pd_lmpi": pl.tolist(), "pd_lmpi_sur": pls.tolist(),
                                   "max_pd_lmpi": float(pl.max()), "max_pd_cnn": float(cnn.max()), "max_pd_band": float(pb.max()),
                                   "band_matches_published": bool(np.allclose(pb, band_pub, rtol=0, atol=1e-4)),  # summary rounds to 4 d.p.
                                   "max_abs_curve_diff_lmpi_cnn": float(np.max(np.abs(pl - cnn))),
                                   "max_abs_curve_diff_band_cnn": float(np.max(np.abs(pb - cnn))),
                                   "gap_lmpi": float(np.mean((pl - pls)[sel])) if sel.any() else None}
    out["comparison"] = comp
    out["runtime_s"] = time.time() - t0
    with open(OUT, "w") as fh:
        json.dump(out, fh)
    for k, v in comp.items():
        print(f"{k:34s} max Pd LMPI {v['max_pd_lmpi']:.3f} CNN {v['max_pd_cnn']:.3f} BAND {v['max_pd_band']:.3f} | "
              f"max|curve diff| LMPI-CNN {v['max_abs_curve_diff_lmpi_cnn']:.3f} BAND-CNN {v['max_abs_curve_diff_band_cnn']:.3f} | "
              f"gap LMPI {v['gap_lmpi']}")
    print(f"{n_checked} captures replayed exactly; {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
