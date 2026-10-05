"""Probe (2026-10-05, PRELIMINARY): candidate closed form of the exact noise-shape wall for ARBITRARY spectra,
checked against the LP of novelty_probes.lp_wall.

Derivation sketch (the project plan, idea U2). With q = p/p_max, q_min = min q, s >= 0 and
  D(s) = mean_f max(1, (s+1)/R - s q(f)),
the pair c1 = max(1, (s+1)/R - s q)/D(s), c0 ∝ g p + c1 with g = s/(p_max D(s)) lies in U_R whenever
(s+1)/R - s q_min <= R, i.e. s <= s_max = (R - 1/R)/(1/R - q_min) (s_max = inf if q_min >= 1/R), and every pair
satisfies g <= s/(p_max D(s)) for some admissible s. h(s) = s/D(s) is increasing, so
  gamma_w = s_max / (p_max D(s_max))   (= Proposition 2 when q_min = 0; infinite when p_max/p_min <= R).

Part A: six spectra x four uncertainties (formula vs LP vs Proposition 2).
Part B: infinite-wall boundary (added 2026-10-05, previously a scratch check): cosine ripples whose peak-to-peak
spectral ratio lies just below or above u; the LP must be unbounded exactly when p_max/p_min <= R.
Writes RESULTS/novelty_probe_wallformula.json.   usage: py -3.12 novelty_probe_wallformula.py"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from novelty_probes import lp_wall
import wall as W

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "RESULTS", "novelty_probe_wallformula.json")
TOL_DB = 0.01


def formula(p, R_db):
    R = 10 ** (R_db / 10)
    p = np.asarray(p, float) / np.mean(p)
    q = p / p.max()
    qmin = q.min()
    if qmin >= 1 / R:
        return np.inf
    s = (R - 1 / R) / (1 / R - qmin)
    D = np.mean(np.maximum(1.0, (s + 1) / R - s * q))
    return s / (p.max() * D)


def db(g):
    return float(10 * np.log10(g)) if np.isfinite(g) else None


f = np.fft.fftfreq(1024)
tests = {
    "cosine ripple 0.5": 1 + 0.5 * np.cos(2 * np.pi * f),
    "cosine ripple 0.9": 1 + 0.9 * np.cos(2 * np.pi * f),
    "pedestal 0.05 + flat band": 0.05 + (np.abs(f) <= 0.1),
    "two bands unequal": 0.02 + 1.0 * (np.abs(f - 0.2) <= 0.05) + 0.4 * (np.abs(f + 0.25) <= 0.1),
    "RRC sps 8 (q_min ~ 0)": W.signal_psd(1024, sps=8),
    "Gaussian bump + floor 0.2": 0.2 + np.exp(-(f / 0.08) ** 2),
}
fmt = lambda g: "inf" if g is None else f"{g:8.3f}"
res = {"_note": "PRELIMINARY probe (novelty_probe_wallformula.py); not paper results.", "tol_db": TOL_DB,
       "part_a": [], "part_b": []}
for name, p in tests.items():
    pn = np.asarray(p, float) / np.mean(p)
    for u in (0.5, 1.0, 2.0, 3.0):
        g_lp, msg = lp_wall(p, u)
        r = {"spectrum": name, "u_db": u, "ripple_db": float(10 * np.log10(pn.max() / pn.min())),
             "q_min": float(pn.min() / pn.max()), "lp_db": db(g_lp) if msg == "ok" else None, "lp_msg": msg,
             "formula_db": db(formula(p, u)), "prop2_db": db(W.wall(pn, u))}
        r["formula_eq_lp"] = (r["lp_db"] is not None and r["formula_db"] is not None
                              and abs(r["lp_db"] - r["formula_db"]) <= TOL_DB)
        r["prop2_minus_lp_db"] = None if r["lp_db"] is None else r["prop2_db"] - r["lp_db"]
        res["part_a"].append(r)
        print(f"{name:28s} u={u}: LP {fmt(r['lp_db']) if msg == 'ok' else 'unbounded':>9s} dB | formula "
              f"{fmt(r['formula_db']):>8s} dB | Prop.2 {fmt(r['prop2_db']):>8s} dB", flush=True)

for amp, u in [(0.1, 1.0), (0.1, 0.5), (0.2, 2.0), (0.06, 0.3), (0.2, 1.0)]:
    p = 1 + amp * np.cos(2 * np.pi * f)
    ripple = float(10 * np.log10(p.max() / p.min()))
    g_lp, msg = lp_wall(p, u)
    g_f = formula(p, u)
    r = {"amp": amp, "u_db": u, "ripple_db": ripple, "predicted_infinite": bool(ripple <= u),
         "lp_bounded": msg == "ok", "lp_db": db(g_lp) if msg == "ok" else None, "lp_msg": msg,
         "formula_db": db(g_f)}
    if r["predicted_infinite"]:
        r["consistent"] = (not r["lp_bounded"]) and r["formula_db"] is None
    else:
        r["consistent"] = (r["lp_bounded"] and r["formula_db"] is not None
                           and abs(r["lp_db"] - r["formula_db"]) <= TOL_DB)
    res["part_b"].append(r)
    print(f"ripple {ripple:.2f} dB, u = {u} dB: LP {fmt(r['lp_db']) if r['lp_bounded'] else 'unbounded (' + msg[:40] + ')'}"
          f"; formula {fmt(r['formula_db'])}; consistent={r['consistent']}", flush=True)

A, B = res["part_a"], res["part_b"]
gaps = [r["prop2_minus_lp_db"] for r in A if r["prop2_minus_lp_db"] is not None]
res["summary"] = {"part_a_formula_eq_lp": f"{sum(r['formula_eq_lp'] for r in A)}/{len(A)}",
                  "part_a_max_abs_formula_minus_lp_db": max(abs(r["lp_db"] - r["formula_db"]) for r in A
                                                            if r["lp_db"] is not None and r["formula_db"] is not None),
                  "part_a_prop2_minus_lp_db_range": [min(gaps), max(gaps)],
                  "part_b_consistent": f"{sum(r['consistent'] for r in B)}/{len(B)}",
                  "part_b_infinite_cases": sum(r["predicted_infinite"] for r in B)}
print(json.dumps(res["summary"], indent=1))
with open(OUT, "w") as fh:
    json.dump(res, fh, indent=1)
