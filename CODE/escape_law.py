"""Step 4 of the project plan (U3): escape law for the cyclostationary detector below the noise-shape wall.

Theory. CYC computes S = sum_{tau in L} |R^a(tau)|^2 / P0^2 with R^a(tau) = mean_n x[n+tau] x*[n] e^{-j 2 pi a n},
a = 1/sps. Under H0 (white noise of unit power) the R^a(tau) are asymptotically independent, circular Gaussian with
E|R|^2 = 1/(N - tau) ~ 1/N, so T = 2 N S ~ chi^2 with 2|L| degrees of freedom. Under H1 (unit-power signal at SNR g),
E[R^a(tau)] = g r(tau) up to a timing phase, where for i.i.d. zero-mean symbols and pulse h (any constellation)
    |r(tau)| = | sum_m h[m+tau] h*[m] e^{-j 2 pi m / sps} | / sum_m |h[m]|^2 ,
and P0 -> 1 + g, so T is noncentral chi^2 with lambda = kappa N g^2 / (1 + g)^2,   kappa = 2 sum_{tau in L} |r(tau)|^2.
The noise-shape wall does not appear: lambda grows linearly in N at any SNR (escape), but only quadratically in g
(Proposition 3: structure enters at order g^2). Escape length for target P_d* at level alpha:
    N_esc = lambda*(P_d*, alpha, 2|L|) (1 + g)^2 / (kappa g^2).

Part A  kappa from the pulse (5 and 16 lags; roll-off 0.15, 0.35, 0.6).
Part B  prediction of every existing CYC wall-test detection rate (wall_test_long: N = 2048..32768, u = 1 dB;
        wall_test_escape: N = 8192/32768, u = 1/2 dB) from kappa and the experiment's own robust threshold (T = 2 N thr),
        no fitted parameter; plus the kappa implied by inverting each measured rate.
Part C  validation on new data: roll-off beta in {0.15, 0.35, 0.6} (kappa changes by a large factor), u = 1 dB, exact
        extremal pair (wall_exact.exact_pair), N in {2048, 8192, 32768}, SNR = wall + delta; measured vs predicted.
Part D  escape-length table (P_d* = 0.9, alpha = 0.05) in samples and in ms at 1.024 MS/s.
Writes RESULTS/escape_law.json.   usage: py -3.12 escape_law.py [skipC]"""
import glob, json, os, sys, time
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim, detectors as D, wall as W
from wall_exact import exact_pair

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "RESULTS")
OUT = os.path.join(RES, "escape_law.json")
ALPHA, FS = 0.05, 1.024e6
L5, L16 = tuple(range(5)), tuple(range(16))


def r_cyc(beta, sps=8, lags=L5):
    h = sim.rrc(beta, sps)
    m = np.arange(len(h))
    out = []
    for t in lags:
        out.append(abs(np.sum(h[t:] * np.conj(h[: len(h) - t]) * np.exp(-2j * np.pi * m[: len(h) - t] / sps))) / np.sum(np.abs(h) ** 2))
    return np.array(out)


def kappa(beta, sps=8, lags=L5):
    return float(2 * np.sum(r_cyc(beta, sps, lags) ** 2))


def pd_pred(N, g, thr_S, k, dof):
    return float(stats.ncx2.sf(2 * N * thr_S, dof, k * N * g * g / (1 + g) ** 2))


def lam_inv(pd, thr_T, dof):
    lo, hi = 1e-8, 1e5
    if pd <= stats.chi2.sf(thr_T, dof):
        return 0.0
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if stats.ncx2.sf(thr_T, dof, mid) < pd:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def psd_beta(N, beta, N0=2048):
    p0 = W.signal_psd(N0, beta=beta, n=8000)
    if N == N0:
        return p0
    f0 = np.fft.fftshift(np.fft.fftfreq(N0))
    f = np.fft.fftfreq(N)
    p = np.interp(f, np.concatenate([f0 - 1, f0, f0 + 1]), np.tile(np.fft.fftshift(p0), 3))
    return p / p.mean()


def partB(kap):
    rows = []
    for src, pat in (("wall_test_long", "wall_test_long_s*.json"), ("wall_test_escape", "wall_test_escape_s*.json")):
        for f in sorted(glob.glob(os.path.join(RES, pat))):
            d = json.load(open(f))
            for key, v in d.items():
                if not key.startswith("N") or "CYC" not in v.get("thr", {}):
                    continue
                N = v["N"]
                for dl, pdv in v["Pd"].items():
                    g = 10 ** ((v["wall_db"] + float(dl)) / 10)
                    meas = pdv["CYC"]
                    pr = pd_pred(N, g, v["thr"]["CYC"], kap, 10)
                    lam = lam_inv(meas["white"], 2 * N * v["thr"]["CYC"], 10)
                    rows.append({"source": src, "file": os.path.basename(f), "N": N, "u_db": float(key.split("_R")[1]),
                                 "delta_db": float(dl), "snr_db": v["wall_db"] + float(dl), "pd_white": meas["white"],
                                 "pd_worst": meas["worst"], "pd_pred": pr,
                                 "kappa_implied": lam * (1 + g) ** 2 / (N * g * g) if lam > 0 else None})
    return rows


def partC(betas=(0.15, 0.35, 0.6), Ns=(2048, 8192, 32768), deltas=(-2, 0, 2, 4), n_cal=2000, n_te=1500, seed=0, u=1.0):
    out = []
    t0 = time.time()
    for beta in betas:
        k5 = kappa(beta)
        for N in Ns:
            chunk = max(50, int(2_000_000 // N))
            p = psd_beta(N, beta)
            g_w, c0, c1 = exact_pair(p, u)
            wall_db = 10 * np.log10(g_w)
            _, rcal, rte = sim.streams(70_000 + seed)
            f = lambda x: D.cyc(x, sps=8)

            def sc(gen, n):
                return np.concatenate([f(gen(min(chunk, n - i))) for i in range(0, n, chunk)])
            cal = [sc(gen, n_cal) for gen in (lambda m: sim.white(m, N, rcal), lambda m: sim.from_psd(m, N, rcal, c0),
                                              lambda m: sim.from_psd(m, N, rcal, c1))]
            thr = float(max(np.quantile(c, 1 - ALPHA) for c in cal))
            for dl in deltas:
                g = 10 ** ((wall_db + dl) / 10)
                a = np.sqrt(g)
                r = {}
                for hn, ng in (("white", lambda m: sim.white(m, N, rte)), ("c1", lambda m: sim.from_psd(m, N, rte, c1))):
                    s = sc(lambda m: sim.modulated(m, N, rte, beta=beta) * a + ng(m), n_te)
                    r[hn] = float(np.mean(s > thr))
                row = {"beta": beta, "kappa5": k5, "N": N, "u_db": u, "wall_db": float(wall_db), "delta_db": dl, "thr_S": thr,
                       "pd_white": r["white"], "pd_c1": r["c1"], "pd_worst": min(r.values()), "pd_pred": pd_pred(N, g, thr, k5, 10)}
                out.append(row)
                print(f"C beta={beta} N={N} wall={wall_db:.2f} d={dl:+d}: meas white {r['white']:.3f} c1 {r['c1']:.3f} "
                      f"pred {row['pd_pred']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    return out


def partD(kap5, kap16):
    rows = []
    for u in (1.0, 2.0):
        g_w = W.wall(W.signal_psd(1024), u)
        for dl in (-2.0, 0.0):
            g = g_w * 10 ** (dl / 10)
            for lab, k, dof in (("CYC5", kap5, 10), ("CYC16", kap16, 32)):
                lam = lam_inv(0.9, stats.chi2.ppf(1 - ALPHA, dof), dof)
                n = lam * (1 + g) ** 2 / (k * g * g)
                rows.append({"u_db": u, "delta_db": dl, "snr_db": 10 * np.log10(g), "detector": lab, "lambda_star": lam,
                             "N_esc": n, "ms": n / FS * 1e3})
    return rows


def main(skipC=False):
    t0 = time.time()
    A = {f"beta{b}": {"kappa5": kappa(b), "kappa16": kappa(b, lags=L16), "r5": r_cyc(b).tolist()} for b in (0.15, 0.35, 0.6)}
    k5, k16 = A["beta0.35"]["kappa5"], A["beta0.35"]["kappa16"]
    print("kappa (beta 0.35): CYC5", k5, "CYC16", k16, flush=True)
    B = partB(k5)
    # B2: 16-lag CYC (cyc_lag_ablation.json, N = 32768, thresholds not stored): prediction with the nominal chi^2_32
    # threshold (CYC's false-alarm rate is insensitive to the noise shape, so robust ~ nominal; see B's thresholds)
    B2 = []
    cl = json.load(open(os.path.join(RES, "cyc_lag_ablation.json")))
    for ukey, v in cl["mean"].items():
        u = float(ukey[1:])
        gw = 10 * np.log10(W.wall(W.signal_psd(1024), u))
        for dl, m in v.items():
            g = 10 ** ((gw + float(dl)) / 10)
            for lab, k, dof in (("CYC5", k5, 10), ("CYC16", k16, 32)):
                pr = float(stats.ncx2.sf(stats.chi2.ppf(1 - ALPHA, dof), dof, k * cl["N"] * g * g / (1 + g) ** 2))
                B2.append({"u_db": u, "delta_db": float(dl), "detector": lab, "pd_meas_worst": m[lab], "pd_pred_nominal_thr": pr})
    thr_ratio = [2 * json.load(open(f))[k]["N"] * json.load(open(f))[k]["thr"]["CYC"] / stats.chi2.ppf(1 - ALPHA, 10)
                 for f in sorted(glob.glob(os.path.join(RES, "wall_test_long_s*.json"))) for k in json.load(open(f)) if k.startswith("N")]
    out = {"_note": "U3 escape law (escape_law.py).", "A": A, "B": B, "B2": B2,
           "robust_over_nominal_threshold_CYC5": [min(thr_ratio), max(thr_ratio)], "D": partD(k5, k16)}
    if not skipC:
        out["C"] = partC()
    out["runtime_s"] = time.time() - t0
    json.dump(out, open(OUT, "w"), indent=1)
    sel = [r for r in B if 0.08 < r["pd_white"] < 0.97]
    err = [abs(r["pd_white"] - r["pd_pred"]) for r in sel]
    ki = [r["kappa_implied"] for r in sel if r["kappa_implied"]]
    print(f"B: {len(B)} points; in 0.08-0.97: {len(sel)}, max |meas-pred| {max(err):.3f}, implied kappa median {np.median(ki):.4f} "
          f"(theory {k5:.4f})")
    if "C" in out:
        e2 = [abs(r["pd_white"] - r["pd_pred"]) for r in out["C"]]
        print(f"C: max |meas(white) - pred| {max(e2):.3f}")
    print(f"runtime {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main(skipC=len(sys.argv) > 1 and sys.argv[1] == "skipC")
