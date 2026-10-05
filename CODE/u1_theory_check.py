"""Step 1 of the project plan (U1): numerical check of the first-order surrogate indistinguishability
(Proposition "surrogate gap is O(gamma^2)") and of the LMPI characterisation.

Theory (supplement S-I): for x = sqrt(g) s + n, n ~ CN(0, I), s zero-mean, circularly symmetric, E||s||^4 < inf, and
s_G ~ CN(0, Sigma_s) with Sigma_s = E[s s^H], every test phi has P_d(s, g) - P_d(s_G, g) = O(g^2) as g -> 0.

Check: N = 256 windows of the paper's modulated signal (sps 8, power-normalised per window as in all experiments).
The *ideal* surrogate is drawn from CN(0, Sigma_s) with Sigma_s estimated from M = 4e5 signal windows (Cholesky); the
paper's surrogate (sim.surrogate, also power-normalised) is evaluated as well. Signal and surrogate share the same
noise draw (paired), which makes the gap estimate precise. Detectors: CYC (5 and 16 lags), the one-sided fourth moment
kappa_lo = -E|x|^4/(E|x|^2)^2, the LMPI statistic (white noise; template = diag of F Sigma_s F^H) and the band ratio.
Thresholds: 95 % quantile on 2e5 white-noise windows. For every detector, the log-log slope of the gap vs gamma is fitted
over the SNRs where the gap exceeds 4 standard errors and P_d(signal) <= 0.6 (small-signal regime).
Writes RESULTS/u1_theory_check.json.   usage: py -3.12 u1_theory_check.py"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim, detectors as D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "RESULTS", "u1_theory_check.json")
N, SPS, M_COV, N_CAL, N_TE, CH = 256, 8, 400_000, 200_000, 100_000, 20_000
SNRS = list(range(-22, 1, 2))
ALPHA = 0.05


def main():
    t0 = time.time()
    rng = np.random.default_rng(20261005)
    # ---------------- covariance of the (power-normalised) signal windows
    S = np.zeros((N, N), complex)
    for i in range(M_COV // CH):
        x = sim.modulated(CH, N, rng, sps=SPS)
        S += x.T @ x.conj()                     # sum of x x^H (columns = samples)
    S /= M_COV
    S = 0.5 * (S + S.conj().T)
    L = np.linalg.cholesky(S + 1e-10 * np.eye(N))
    F = np.fft.fft(np.eye(N)) / np.sqrt(N)
    p_bins = np.real(np.einsum("ij,jk,ik->i", F, S, F.conj()))      # diag(F S F^H): mean periodogram of the signal
    w = p_bins / p_bins.sum()

    def lmpi(x):
        P = np.abs(np.fft.fft(x, axis=1)) ** 2
        return (P @ w) / P.sum(1)

    def kappa_lo(x):
        p2 = np.mean(np.abs(x) ** 2, 1)
        return -np.mean(np.abs(x) ** 4, 1) / p2 ** 2

    dets = {"CYC5": lambda x: D.cyc(x, SPS), "CYC16": lambda x: D.cyc(x, SPS, lags=tuple(range(16))),
            "KAPPALO": kappa_lo, "LMPI": lmpi, "BAND": D.band_ratio}

    def ideal_sur(n):
        z = (rng.standard_normal((n, N)) + 1j * rng.standard_normal((n, N))) / np.sqrt(2)
        return z @ L.T

    # ---------------- thresholds on white noise
    cal = {k: [] for k in dets}
    for i in range(N_CAL // CH):
        x = sim.white(CH, N, rng)
        for k, f in dets.items():
            cal[k].append(f(x))
    thr = {k: float(np.quantile(np.concatenate(v), 1 - ALPHA)) for k, v in cal.items()}
    # ---------------- paired H1 draws
    res = {k: {"snr": SNRS, "pd_sig": [], "pd_ideal": [], "pd_paper": [], "gap_ideal": [], "se_ideal": [],
               "gap_paper": [], "se_paper": []} for k in dets}
    for d in SNRS:
        g = np.sqrt(10 ** (d / 10))
        acc = {k: {"s": [], "i": [], "p": []} for k in dets}
        for i in range(N_TE // CH):
            n0 = sim.white(CH, N, rng)
            xs = g * sim.modulated(CH, N, rng, sps=SPS) + n0
            xi = g * ideal_sur(CH) + n0
            xp = g * sim.surrogate(CH, N, rng, sps=SPS) + n0
            for k, f in dets.items():
                acc[k]["s"].append(f(xs) > thr[k])
                acc[k]["i"].append(f(xi) > thr[k])
                acc[k]["p"].append(f(xp) > thr[k])
        for k in dets:
            s, i_, p_ = (np.concatenate(acc[k][c]).astype(float) for c in ("s", "i", "p"))
            r = res[k]
            r["pd_sig"].append(s.mean()); r["pd_ideal"].append(i_.mean()); r["pd_paper"].append(p_.mean())
            r["gap_ideal"].append((s - i_).mean()); r["se_ideal"].append((s - i_).std() / np.sqrt(len(s)))
            r["gap_paper"].append((s - p_).mean()); r["se_paper"].append((s - p_).std() / np.sqrt(len(s)))
        print(f"SNR {d:+3d} dB: " + "  ".join(f"{k} Pd {res[k]['pd_sig'][-1]:.3f} gap {100*res[k]['gap_ideal'][-1]:+.2f}"
                                               for k in dets), flush=True)
    # ---------------- slopes (log gap vs log gamma)
    for k, r in res.items():
        gam = 10 ** (np.array(SNRS) / 10)
        for which in ("ideal", "paper"):
            gp, se = np.array(r[f"gap_{which}"]), np.array(r[f"se_{which}"])
            sel = (gp > 4 * se) & (np.array(r["pd_sig"]) <= 0.6)
            if sel.sum() >= 3:
                b = np.polyfit(np.log10(gam[sel]), np.log10(gp[sel]), 1)
                r[f"slope_{which}"] = float(b[0]); r[f"slope_{which}_n"] = int(sel.sum())
                r[f"slope_{which}_snrs"] = [int(s) for s in np.array(SNRS)[sel]]
            else:
                r[f"slope_{which}"] = None; r[f"slope_{which}_n"] = int(sel.sum())
        # first-order sensitivity: Pd - alpha at the two lowest SNRs (LMPI/BAND should be ~linear in gamma)
        r["max_abs_gap_ideal_over_4se"] = float(np.max(np.abs(r["gap_ideal"]) / np.maximum(np.array(r["se_ideal"]), 1e-12)))
    out = {"_note": "U1 numerical check (u1_theory_check.py).", "N": N, "sps": SPS, "M_cov": M_COV, "n_cal": N_CAL,
           "n_test_per_snr": N_TE, "alpha": ALPHA, "thresholds": thr, "results": res, "runtime_s": time.time() - t0}
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    for k, r in res.items():
        print(k, "slope ideal", r["slope_ideal"], "(n", r["slope_ideal_n"], ")", "slope paper", r["slope_paper"],
              "max|gap|/se", round(r["max_abs_gap_ideal_over_4se"], 2))
    print(f"runtime {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
