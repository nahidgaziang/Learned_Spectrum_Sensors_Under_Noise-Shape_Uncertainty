"""Feasibility probes for PROJECT/the project plan (2026-10-05). PRELIMINARY: these numbers decide which
upgrade ideas are real; they are NOT paper results until re-run under the plan's protocol and wired into the
macro/claim pipeline.

P1  Exact noise-shape wall by linear programming (LP) for arbitrary signal spectra, compared with Proposition 2's
    constructed pair (wall.wall) and, for flat spectra, with Theorem 1's closed form.
    For fixed R the pair condition (1+g)c0 = g p + c1, c0, c1 in U_R becomes linear after d0 = (1+g) c0:
      d0 - c1 - g p = 0,  mean(d0) = 1 + g,  mean(c1) = 1,  m0 <= d0 <= R m0,  m1 <= c1 <= R m1,  maximise g.
P2  Decision-level equivalence on identical test windows (same evaluation seed => same windows):
    white-trained networks vs the band ratio vs the locally most powerful invariant (LMPI) test of a weak Gaussian
    signal with PSD p in white noise of unknown level, T(x) = sum_k p_k |X_k|^2 / sum_k |X_k|^2 (King's test,
    circulant approximation of x^H T(p) x / x^H x).
P3  Escape scaling: CYC statistic = sum over 5 lags of |cyclic autocorrelation|^2 / P0^2 ~ chi2 with 10 d.o.f. under
    stationary noise; invert the measured worst-case P_d of the long wall tests to a noncentrality lambda and test
    lambda ∝ N * gamma^2 (gamma = SNR).
Writes RESULTS/novelty_probes.json.   usage: py -3.12 novelty_probes.py [p1] [p2] [p3]   (default: all)"""
import glob, json, os, sys
import numpy as np
from scipy import sparse, stats
from scipy.optimize import linprog

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim, wall as W

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "RESULTS", "novelty_probes.json")


def lp_wall(p, R_db):
    """Exact wall of U_R for signal PSD p on a grid (largest g with a pair), via one LP."""
    R = 10 ** (R_db / 10)
    p = np.asarray(p, float) / np.mean(p)
    K = len(p)
    nv = 2 * K + 3                      # d0 (K), c1 (K), m0, m1, g
    i0, i1, im0, im1, ig = 0, K, 2 * K, 2 * K + 1, 2 * K + 2
    I = sparse.identity(K, format="csr")
    # equalities
    A_eq = sparse.hstack([I, -I, sparse.csr_matrix((K, 2)), sparse.csr_matrix(-p[:, None])], format="csr")
    row_d0 = sparse.hstack([sparse.csr_matrix(np.full((1, K), 1.0 / K)), sparse.csr_matrix((1, K)),
                            sparse.csr_matrix([[0.0, 0.0, -1.0]])], format="csr")
    row_c1 = sparse.hstack([sparse.csr_matrix((1, K)), sparse.csr_matrix(np.full((1, K), 1.0 / K)),
                            sparse.csr_matrix([[0.0, 0.0, 0.0]])], format="csr")
    A_eq = sparse.vstack([A_eq, row_d0, row_c1], format="csr")
    b_eq = np.concatenate([np.zeros(K), [1.0, 1.0]])
    # inequalities: m0 - d0 <= 0 ; d0 - R m0 <= 0 ; m1 - c1 <= 0 ; c1 - R m1 <= 0
    Z = sparse.csr_matrix((K, K))
    e = np.ones((K, 1))
    blocks = [
        sparse.hstack([-I, Z, sparse.csr_matrix(e), sparse.csr_matrix((K, 1)), sparse.csr_matrix((K, 1))]),
        sparse.hstack([I, Z, sparse.csr_matrix(-R * e), sparse.csr_matrix((K, 1)), sparse.csr_matrix((K, 1))]),
        sparse.hstack([Z, -I, sparse.csr_matrix((K, 1)), sparse.csr_matrix(e), sparse.csr_matrix((K, 1))]),
        sparse.hstack([Z, I, sparse.csr_matrix((K, 1)), sparse.csr_matrix(-R * e), sparse.csr_matrix((K, 1))]),
    ]
    A_ub = sparse.vstack(blocks, format="csr")
    b_ub = np.zeros(4 * K)
    c = np.zeros(nv); c[ig] = -1.0
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * nv, method="highs")
    if res.status != 0:
        return float("nan"), res.message
    return float(res.x[ig]), "ok"


def p1():
    out = {}
    for sps in (4, 8, 16):
        p = W.signal_psd(1024, sps=sps)
        for u in (0.5, 1.0, 2.0, 3.0):
            g_lp, msg = lp_wall(p, u)
            g_c = W.wall(p, u)
            out[f"sps{sps}_u{u}"] = {"lp_db": 10 * np.log10(g_lp), "prop2_db": 10 * np.log10(g_c), "diff_db": 10 * np.log10(g_lp / g_c), "msg": msg}
            print(f"P1 sps={sps:2d} u={u}: LP wall {10*np.log10(g_lp):7.3f} dB  Prop.2 {10*np.log10(g_c):7.3f} dB  "
                  f"diff {10*np.log10(g_lp/g_c):+.3f} dB", flush=True)
    # validation on a flat band (Theorem 1 closed form)
    for rho in (0.169, 0.5):
        f = np.fft.fftfreq(1024)
        p = (np.abs(f) <= rho / 2).astype(float)
        rr = p.mean()
        for u in (1.0, 2.0):
            R = 10 ** (u / 10)
            closed = rr * (R ** 2 - 1) / (rr + R * (1 - rr))
            g_lp, _ = lp_wall(p, u)
            out[f"flat_rho{rho}_u{u}"] = {"lp_db": 10 * np.log10(g_lp), "thm1_db": 10 * np.log10(closed)}
            print(f"P1 flat rho={rr:.3f} u={u}: LP {10*np.log10(g_lp):.4f} dB vs Theorem 1 {10*np.log10(closed):.4f} dB", flush=True)
    return out


def lmpi_fn(N, sps=8):
    p = W.signal_psd(N, sps=sps)
    w = p / p.sum()

    def f(x):
        P = np.abs(np.fft.fft(x, axis=1)) ** 2
        return (P @ w) / P.sum(1)
    return f


def p2(seeds=(0, 1, 2)):
    import evaluate as E
    keys = ["h0_white", "h0_lpf0.8", "h0_rtl3dB", "h0_pink", "h0_ar+0.1", "h0_ar-0.1",
            "h1_white_-14", "h1_white_-10", "sur_white_-10", "h1_lpf0.8_-10"]
    nets = ["CLDNN", "ResNet1D", "CovCNN", "STFTCNN", "GapCNN"]
    out = {"lmpi": {}, "agreement": {}}
    for s in seeds:
        res, raw = E.evaluate(lmpi_fn(128), 128, 8, s)
        out["lmpi"][f"s{s}"] = {"gap": res.get("surrogate_gap"), "Pd_-10": res["Pd"]["-10"]["white"]["nominal"],
                                "Pf": {k: res["Pf"][k]["nominal"] for k in ("white", "lpf0.8", "rtl3dB", "pink", "ar+0.1", "ar-0.1")}}
        thr_l = res["thresholds"]["nominal"]
        band = np.load(os.path.join(ROOT, "RESULTS", "scores", f"cls_BAND_N128_sps8_s{s}.npz"))
        thr_b = json.load(open(os.path.join(ROOT, "RESULTS", "runs", f"cls_BAND_N128_sps8_s{s}.json")))["eval"]["thresholds"]["nominal"]
        for net in nets:
            fz = os.path.join(ROOT, "RESULTS", "scores", f"{net}_WHITE_N128_sps8_s{s}.npz")
            if not os.path.exists(fz):
                continue
            z = np.load(fz)
            thr_n = json.load(open(fz.replace("scores", "runs").replace(".npz", ".json")))["eval"]["thresholds"]["nominal"]
            row = {}
            for k in keys:
                a, b, l = z[k], band[k], raw[k]
                da, db, dl = a > thr_n, b > thr_b, l > thr_l
                row[k] = {"rho_net_band": float(stats.spearmanr(a, b)[0]), "rho_net_lmpi": float(stats.spearmanr(a, l)[0]),
                          "agree_net_band": float(np.mean(da == db)), "agree_net_lmpi": float(np.mean(da == dl)),
                          "pd_net": float(da.mean()), "pd_band": float(db.mean()), "pd_lmpi": float(dl.mean())}
            out["agreement"][f"{net}_s{s}"] = row
        print(f"P2 seed {s}: LMPI gap {res.get('surrogate_gap')}, Pd(-10) {res['Pd']['-10']['white']['nominal']:.3f}, "
              f"Pf LP.8 {res['Pf']['lpf0.8']['nominal']:.3f}, pink {res['Pf']['pink']['nominal']:.3f}", flush=True)
        g = out["agreement"].get(f"GapCNN_s{s}", {})
        for k in ("h1_white_-10", "sur_white_-10", "h0_lpf0.8"):
            if k in g:
                v = g[k]
                print(f"   GapCNN {k}: rho(net,band) {v['rho_net_band']:.3f}  rho(net,LMPI) {v['rho_net_lmpi']:.3f}  "
                      f"agree band {v['agree_net_band']:.3f}  agree LMPI {v['agree_net_lmpi']:.3f}  "
                      f"Pd net/band/LMPI {v['pd_net']:.3f}/{v['pd_band']:.3f}/{v['pd_lmpi']:.3f}", flush=True)
    return out


def p3():
    files = sorted(glob.glob(os.path.join(ROOT, "RESULTS", "wall_test_long_s*.json")))
    D = [json.load(open(f)) for f in files]
    thr = stats.chi2.ppf(0.95, 10)
    pts = []
    for R in ("1.0", "2.0"):
        for N in (2048, 8192, 32768):
            key = f"N{N}_R{R}"
            if not all(key in d for d in D):
                continue
            wall_db = D[0][key]["wall_db"]
            for dl in D[0][key]["Pd"]:
                pd = float(np.mean([d[key]["Pd"][dl]["CYC"]["worst"] for d in D]))
                if not (0.08 < pd < 0.97):
                    continue
                lam = float(_inv_ncx2(pd, 10, thr))
                g = 10 ** ((wall_db + float(dl)) / 10)
                pts.append({"R": R, "N": N, "delta": float(dl), "pd": pd, "lambda": lam, "lam_over_Ng2": lam / (N * g * g)})
    if pts:
        k = np.array([q["lam_over_Ng2"] for q in pts])
        print(f"P3 {len(pts)} points; lambda/(N g^2): median {np.median(k):.3f}, range {k.min():.3f}-{k.max():.3f}, "
              f"CV {k.std()/k.mean():.2f}", flush=True)
        for q in pts:
            print(f"   u={q['R']} N={q['N']:6d} delta={q['delta']:+.0f}  Pd {q['pd']:.3f}  lambda {q['lambda']:.2f}  "
                  f"lambda/(N g^2) {q['lam_over_Ng2']:.3f}", flush=True)
    return {"points": pts}


def _inv_ncx2(pd, df, thr):
    lo, hi = 1e-6, 1e4
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if stats.ncx2.sf(thr, df, mid) < pd:
            lo = mid
        else:
            hi = mid
    return np.sqrt(lo * hi)


if __name__ == "__main__":
    which = sys.argv[1:] or ["p1", "p2", "p3"]
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    out["_note"] = "PRELIMINARY feasibility probes (novelty_probes.py); not paper results."
    for w in which:
        out[w] = {"p1": p1, "p2": p2, "p3": p3}[w]()
        with open(OUT + ".tmp", "w") as fh:
            json.dump(out, fh, indent=1)
        os.replace(OUT + ".tmp", OUT)
