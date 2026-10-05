"""Step 3 of the project plan (U2): exact noise-shape SNR wall for arbitrary signal spectra.

Theorem (supplement S-I). Let p >= 0 have unit mean, q = p / p_max, q_min = min q, R = 10^(u/10) > 1, and
    D(s) = mean_f max(1, (s+1)/R - s q(f)),   s_max = (R - 1/R) / (1/R - q_min)   (s_max = inf if q_min >= 1/R).
Then gamma_w(U_R, p) = s_max / (p_max D(s_max)); in particular gamma_w = inf iff p_max / p_min <= R, and gamma_w equals
Proposition 2's constructed value iff q_min = 0. Every SNR in [0, gamma_w] admits a confusable pair; above gamma_w a
scale-invariant linear periodogram test separates H0 from H1 uniformly as N -> inf.

Part A  closed form vs linear programme (primal: largest gamma with a confusable pair) on 34 spectra x 4 uncertainties,
        plus Theorem 1 for flat bands and Proposition 2 for comparison.
Part B  infinite-wall boundary: spectra whose dynamic range lies just below / above u.
Part C  dual check: for gamma on a grid, the best margin delta*(gamma) of a separating linear functional w
        (max delta s.t. <w, c> <= tau for every unit-mean c in U_R and <w, (gamma p + c)/(1 + gamma)> >= tau + delta for
        every unit-mean c in U_R; sum w = 0, |w| <= 1), solved by cutting planes (extreme points of U_R are two-level
        shapes, so the worst shapes have closed forms). delta* must cross zero at gamma_w.
Part D  finite-N achievability on a spectrum where Proposition 2 is NOT exact (Gaussian bump on a floor, u = 1 dB):
        the test T_N = <w, P_hat> / mean(P_hat) with w from Part C at wall + 1 dB and a FIXED threshold (midpoint of the
        margin, no calibration); H0 over the exact extremal pair, the worst-case shape for w and 40 random admissible
        shapes; H1 = Gaussian signal with PSD p at wall + delta in noise c1*, white and the worst-case H1 shape;
        N = 2048, 8192, 32768. Prediction: max P_f -> 0 and worst P_d(+1 dB) -> 1; P_d at and below the exact wall -> 0
        (the exact pair makes the hypotheses identical there), although Proposition 2's value is 0.73 dB lower.
Writes RESULTS/wall_exact.json.   usage: py -3.12 wall_exact.py"""
import json, os, sys, time
import numpy as np
from scipy import sparse
from scipy.optimize import linprog

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim, wall as W
from novelty_probes import lp_wall

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "RESULTS", "wall_exact.json")
K = 1024
US = (0.5, 1.0, 2.0, 3.0)


def exact_wall(p, u_db):
    R = 10 ** (u_db / 10)
    p = np.asarray(p, float) / np.mean(p)
    q = p / p.max()
    qmin = q.min()
    if qmin >= 1 / R:
        return np.inf, np.inf
    s = (R - 1 / R) / (1 / R - qmin)
    return s / (p.max() * np.mean(np.maximum(1.0, (s + 1) / R - s * q))), s


def exact_pair(p, u_db):
    """Extremal pair (c0, c1) at the exact wall (unit means)."""
    R = 10 ** (u_db / 10)
    p = np.asarray(p, float) / np.mean(p)
    q = p / p.max()
    g, s = exact_wall(p, u_db)
    gs = np.maximum(1.0, (s + 1) / R - s * q)
    c1 = gs / gs.mean()
    c0 = (g * p + c1) / (1 + g)
    return g, c0, c1


def spectra(Kg=K):
    f = np.fft.fftfreq(Kg)
    S = {}
    for sps in (2, 4, 8, 16):
        for beta in (0.15, 0.35, 0.6):
            S[f"RRC sps{sps} beta{beta}"] = W.signal_psd(Kg, sps=sps, beta=beta)
    rng = np.random.default_rng(5)
    x = sim.ofdm(3000, Kg, rng)
    P = np.mean(np.abs(np.fft.fft(x * np.hanning(Kg)[None, :], axis=1)) ** 2, 0)
    S["OFDM 48/64 (multipath, CFO)"] = P / P.mean()
    for rho in (0.1, 0.169, 0.3, 0.5, 0.75):
        S[f"flat band rho{rho}"] = (np.abs(f) <= rho / 2).astype(float)
    for fl in (0.01, 0.05, 0.2):
        S[f"flat band 0.2 + floor {fl}"] = fl + (np.abs(f) <= 0.1)
    for fl in (0.05, 0.2, 0.5):
        S[f"Gaussian bump + floor {fl}"] = fl + np.exp(-(f / 0.08) ** 2)
    for a in (0.3, 0.5, 0.9):
        S[f"cosine ripple {a}"] = 1 + a * np.cos(2 * np.pi * f)
    S["two unequal bands + floor 0.02"] = 0.02 + 1.0 * (np.abs(f - 0.2) <= 0.05) + 0.4 * (np.abs(f + 0.25) <= 0.1)
    S["three bands + floor 0.1"] = 0.1 + (np.abs(f) <= 0.05) + 0.5 * (np.abs(f - 0.3) <= 0.05) + 0.25 * (np.abs(f + 0.3) <= 0.05)
    r2 = np.random.default_rng(11)
    for i in range(5):
        a = r2.normal(0, 1, 8) / np.arange(1, 9)
        g = sum(a[k] * np.cos(2 * np.pi * (k + 1) * f + r2.uniform(0, 2 * np.pi)) for k in range(8))
        g = (g - g.min()) / np.ptp(g)
        S[f"random smooth {i} ({[3, 6, 10, 15, 20][i]} dB range)"] = 10 ** ([3, 6, 10, 15, 20][i] * g / 10)
    return {k: v / v.mean() for k, v in S.items()}


# ---------------------------------------------------------------- Part C: separating functional by cutting planes
def worst_h0(w, R):
    """max over unit-mean c in U_R of <w, c>/K: two-level c = m (1 + (R-1) 1_S), S = top-k bins of w."""
    Kg = len(w)
    ws = np.sort(w)[::-1]
    cs = np.concatenate([[0.0], np.cumsum(ws)])
    k = np.arange(Kg + 1)
    m = 1.0 / (1.0 + (R - 1) * k / Kg)
    val = m * (w.sum() + (R - 1) * cs) / Kg
    kb = int(np.argmax(val))
    c = np.full(Kg, m[kb])
    c[np.argsort(w)[::-1][:kb]] *= R
    return float(val[kb]), c


def worst_h1(w, R):
    """min over unit-mean c in U_R of <w, c>/K: S = bottom-k bins of w raised by R."""
    v, c = worst_h0(-w, R)
    return -v, c


def margin(p, g, u_db):
    """Largest separation margin delta of a linear spectral functional w (sum w = 0, |w| <= 1):
        max_{c in U_R, mean 1} <w, c>/K <= tau   and   (<w, g p>/K + min_{c in U_R, mean 1} <w, c>/K)/(1 + g) >= tau + delta.
    The inner max/min are LPs; their duals make the robust problem one LP (no iteration):
        max <w,c>/K <= tau  <=>  exists a, b >= 0, nu <= tau:  w/K - a + b - nu/K <= 0,  R sum(a) - sum(b) <= 0.
    delta = 0 is always feasible (w = 0); delta > 0 iff the normalised H0 and H1 sets are strictly separated."""
    R = 10 ** (u_db / 10)
    Kg = len(p)
    I = sparse.identity(Kg, format="csr")
    Z = sparse.csr_matrix((Kg, Kg))
    col = lambda v: sparse.csr_matrix(np.asarray(v, float).reshape(-1, 1))
    zc = sparse.csr_matrix((Kg, 1))
    # variable order: w, a, b, a2, b2 (K each), nu, nu2, tau, delta
    r1 = sparse.hstack([I / Kg, -I, I, Z, Z, col(np.full(Kg, -1.0 / Kg)), zc, zc, zc])
    r2 = sparse.hstack([sparse.csr_matrix((1, Kg)), col(np.full(Kg, R)).T, col(np.full(Kg, -1.0)).T,
                        sparse.csr_matrix((1, 2 * Kg)), sparse.csr_matrix((1, 4))])
    r3 = sparse.hstack([sparse.csr_matrix((1, 5 * Kg)), sparse.csr_matrix([[1.0, 0.0, -1.0, 0.0]])])
    r4 = sparse.hstack([-I / Kg, Z, Z, -I, I, zc, col(np.full(Kg, -1.0 / Kg)), zc, zc])
    r5 = sparse.hstack([sparse.csr_matrix((1, 3 * Kg)), col(np.full(Kg, R)).T, col(np.full(Kg, -1.0)).T, sparse.csr_matrix((1, 4))])
    r6 = sparse.hstack([col(-g * p / ((1 + g) * Kg)).T, sparse.csr_matrix((1, 4 * Kg)),
                        sparse.csr_matrix([[0.0, 1.0 / (1 + g), 1.0, 1.0]])])
    A_ub = sparse.vstack([r1, r2, r3, r4, r5, r6], format="csr")
    b_ub = np.zeros(A_ub.shape[0])
    A_eq = sparse.hstack([sparse.csr_matrix(np.ones((1, Kg))), sparse.csr_matrix((1, 4 * Kg + 4))], format="csr")
    cost = np.zeros(5 * Kg + 4); cost[-1] = -1.0
    bounds = [(-1, 1)] * Kg + [(0, None)] * (4 * Kg) + [(None, None)] * 3 + [(None, 1.0)]
    r = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[0.0], bounds=bounds, method="highs")
    assert r.status == 0, r.message
    x = r.x
    return {"delta": float(x[-1]), "tau": float(x[-2]), "iters": 1, "w": x[:Kg]}


def partC(p, u_db, name):
    gw, _ = exact_wall(p, u_db)
    out = {"spectrum": name, "u_db": u_db, "wall_db": float(10 * np.log10(gw)), "grid": []}
    for d in (-1.0, -0.5, -0.2, -0.05, 0.05, 0.2, 0.5, 1.0):
        m = margin(p, gw * 10 ** (d / 10), u_db)
        out["grid"].append({"delta_db": d, "margin": m["delta"], "iters": m["iters"]})
        print(f"  C {name} u={u_db} wall{d:+.2f} dB: margin {m['delta']:+.3e} ({m['iters']} it)", flush=True)
    return out


# ---------------------------------------------------------------- Part D: finite-N achievability
def partD(u_db=1.0, Ns=(2048, 8192, 32768)):
    rng = np.random.default_rng(2027)
    out = {"u_db": u_db, "spectrum": "Gaussian bump + floor 0.2", "N": {}}
    for N in Ns:
        f = np.fft.fftfreq(N)
        p = 0.2 + np.exp(-(f / 0.08) ** 2); p /= p.mean()
        R = 10 ** (u_db / 10)
        gw, c0s, c1s = exact_pair(p, u_db)
        g_p2 = W.wall(p, u_db)
        g1 = gw * 10 ** 0.1
        m = margin(p, g1, u_db)
        w, tau, delta = m["w"], m["tau"], m["delta"]
        thr = tau + 0.5 * delta
        _, h0w = worst_h0(w, R)
        _, h1w = worst_h1(w, R)
        r2 = np.random.default_rng(7)
        rand = []
        for i in range(40):
            if i % 2 == 0:
                lo, wd = r2.uniform(-0.5, 0.5), r2.uniform(0.05, 0.6)
                mm = np.abs(((f - lo + 0.5) % 1.0) - 0.5) <= wd / 2
                s = np.where(mm, R, 1.0) if r2.random() < 0.5 else np.where(mm, 1.0, R)
            else:
                a = r2.normal(0, 1, 6) / np.arange(1, 7)
                gg = sum(a[k] * np.cos(2 * np.pi * (k + 1) * f + r2.uniform(0, 2 * np.pi)) for k in range(6))
                gg = (gg - gg.min()) / max(np.ptp(gg), 1e-12)
                s = 10 ** (u_db * gg / 10)
            rand.append(s / s.mean())

        def T(x):
            P = np.abs(np.fft.fft(x, axis=1)) ** 2
            return (P @ w) / P.sum(1)
        n = max(200, int(4_000_000 // N))
        shapes = {"c0*": c0s, "c1*": c1s, "worst-H0 for w": h0w / h0w.mean()}
        pf = {k: float(np.mean(T(sim.from_psd(n, N, rng, s)) > thr)) for k, s in shapes.items()}
        pf_rand = [float(np.mean(T(sim.from_psd(n, N, rng, s)) > thr)) for s in rand]
        pd = {}
        for d in (-1.0, -0.4, 0.0, 0.5, 1.0):
            a = np.sqrt(gw * 10 ** (d / 10))
            r = {}
            for lab, noise in (("c1*", c1s), ("white", np.ones(N)), ("worst-H1 for w", h1w / h1w.mean())):
                x = a * sim.from_psd(n, N, rng, p) + sim.from_psd(n, N, rng, noise)
                r[lab] = float(np.mean(T(x) > thr))
            pd[str(d)] = {"by_noise": r, "worst": min(r.values())}
        out["N"][str(N)] = {"wall_db": float(10 * np.log10(gw)), "prop2_db": float(10 * np.log10(g_p2)),
                            "margin_at_plus1": delta, "threshold": thr, "n": n, "Pf": pf,
                            "max_Pf_random40": max(pf_rand), "max_Pf_all": max(max(pf.values()), max(pf_rand)), "Pd": pd}
        print(f"D N={N}: wall {10*np.log10(gw):.3f} dB (Prop.2 {10*np.log10(g_p2):.3f}); max Pf {out['N'][str(N)]['max_Pf_all']:.4f}; "
              + " ".join(f"{d}:{v['worst']:.3f}" for d, v in pd.items()), flush=True)
    return out


def main():
    t0 = time.time()
    S = spectra()
    A = []
    for name, p in S.items():
        for u in US:
            g_lp, msg = lp_wall(p, u)
            g_f, _ = exact_wall(p, u)
            rr = {"spectrum": name, "u_db": u, "q_min": float(p.min() / p.max()),
                  "range_db": float(10 * np.log10(p.max() / max(p.min(), 1e-300))),
                  "lp_db": float(10 * np.log10(g_lp)) if msg == "ok" else None, "lp_msg": msg,
                  "formula_db": float(10 * np.log10(g_f)) if np.isfinite(g_f) else None,
                  "prop2_db": float(10 * np.log10(W.wall(p, u)))}
            if name.startswith("flat band rho"):
                rho = float(np.mean(p > 0)); R = 10 ** (u / 10)
                rr["thm1_db"] = float(10 * np.log10(rho * (R * R - 1) / (rho + R * (1 - rho))))
            rr["ok"] = (rr["lp_db"] is None and rr["formula_db"] is None) or (
                rr["lp_db"] is not None and rr["formula_db"] is not None and abs(rr["lp_db"] - rr["formula_db"]) < 1e-3)
            A.append(rr)
        print(f"A {name}: " + " ".join(f"u{r['u_db']}:{r['formula_db'] if r['formula_db'] is None else round(r['formula_db'], 3)}"
                                       for r in A[-4:]), flush=True)
    # Part B: infinite-wall boundary
    f = np.fft.fftfreq(K)
    B = []
    for amp in (0.02, 0.05, 0.1, 0.2, 0.3):
        p = 1 + amp * np.cos(2 * np.pi * f)
        rng_db = 10 * np.log10(p.max() / p.min())
        for u in (0.5 * rng_db, 0.95 * rng_db, 1.05 * rng_db, 2 * rng_db):
            g_lp, msg = lp_wall(p, u)
            g_f, _ = exact_wall(p, u)
            B.append({"amp": amp, "range_db": rng_db, "u_db": u, "predicted_infinite": bool(rng_db <= u),
                      "lp_bounded": msg == "ok", "formula_infinite": bool(not np.isfinite(g_f)),
                      "lp_db": float(10 * np.log10(g_lp)) if msg == "ok" else None,
                      "formula_db": float(10 * np.log10(g_f)) if np.isfinite(g_f) else None})
            B[-1]["ok"] = (B[-1]["predicted_infinite"] == (not B[-1]["lp_bounded"]) == B[-1]["formula_infinite"]) and (
                B[-1]["lp_db"] is None or abs(B[-1]["lp_db"] - B[-1]["formula_db"]) < 1e-3)
    # Part C: dual margin crosses zero at the exact wall
    C = [partC(S["RRC sps8 beta0.35"], 1.0, "RRC sps8 beta0.35"), partC(S["Gaussian bump + floor 0.2"], 1.0, "Gaussian bump + floor 0.2"),
         partC(S["cosine ripple 0.5"], 2.0, "cosine ripple 0.5"), partC(S["OFDM 48/64 (multipath, CFO)"], 1.0, "OFDM 48/64 (multipath, CFO)")]
    # Part D: finite-N achievability on a spectrum where Proposition 2 is not exact
    Dd = partD()
    gaps = [r["prop2_db"] - r["lp_db"] for r in A if r["lp_db"] is not None]
    summ = {"n_spectra": len(S), "n_cases": len(A), "formula_eq_lp": f"{sum(r['ok'] for r in A)}/{len(A)}",
            "max_abs_formula_minus_lp_db": max(abs(r["lp_db"] - r["formula_db"]) for r in A if r["lp_db"] is not None),
            "prop2_minus_exact_db_range": [min(gaps), max(gaps)],
            "prop2_exact_when_qmin_tiny": all(abs(r["prop2_db"] - r["lp_db"]) < 1e-3 for r in A if r["lp_db"] is not None and r["q_min"] < 1e-6),
            "thm1_agree": all(abs(r["thm1_db"] - r["lp_db"]) < 1e-3 for r in A if "thm1_db" in r),
            "boundary_ok": f"{sum(r['ok'] for r in B)}/{len(B)}",
            "margin_sign_ok": all((g["margin"] > 1e-9) == (g["delta_db"] > 0) for c in C for g in c["grid"])}
    out = {"_note": "U2 exact wall (wall_exact.py).", "summary": summ, "A": A, "B": B, "C": C, "D": Dd, "runtime_s": time.time() - t0}
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(summ, indent=1))
    print(f"runtime {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
