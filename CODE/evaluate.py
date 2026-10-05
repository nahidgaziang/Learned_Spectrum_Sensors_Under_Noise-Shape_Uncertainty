"""Evaluation suite shared by classical and learned detectors.

score_fn: callable(complex ndarray (n, N)) -> float ndarray (n,)

Protocol (Pf target alpha = 0.05):
- nominal threshold  : (1-alpha)-quantile of scores on white noise from the CALIBRATION stream
- robust threshold(U): max over shapes c in U of the (1-alpha)-quantile on c-noise from the CALIBRATION stream
- Pf(shape)          : fraction of TEST-stream c-noise above a threshold
- Pd(SNR, shape)     : fraction of TEST-stream (signal + c-noise) above a threshold
- worst-case Pd(U)   : min over c in U of Pd(SNR, c) at the robust threshold of U
- surrogate Pd       : same as Pd but with the PSD-matched Gaussian surrogate in place of the signal
Calibration and test draws come from independent RNG streams (sim.streams).
"""
import numpy as np
import sim

ALPHA = 0.05
SNRS = [-20, -17, -14, -12, -10, -8, -6, -3, 0]
TILTS_DB = [0.5, 1.0, 2.0]


def _gen_shape(name, n, N, rng):
    if name.startswith("tilt"):
        # tilt{R}{+/-}: two-level band tilt with peak-to-peak R dB around the signal band
        R = float(name[4:-1]); sign = +1 if name.endswith("+") else -1
        return sim.band_tilt(n, N, rng, R, sign=sign)
    return sim.TEST_SHAPES[name](n, N, rng)


def all_shapes():
    return list(sim.TEST_SHAPES) + [f"tilt{R}{s}" for R in TILTS_DB for s in "+-"]


def tilt_sets():
    return {f"U_tilt{R}": ["white", f"tilt{R}+", f"tilt{R}-"] for R in TILTS_DB}


def _scores(score_fn, gen, n, chunk=2000):
    out = []
    done = 0
    while done < n:
        m = min(chunk, n - done)
        out.append(score_fn(gen(m)))
        done += m
    return np.concatenate(out).astype(np.float32)


def evaluate(score_fn, N, sps, seed, n_cal=20000, n_cal_shape=4000, n_h0=4000, n_h1=2000,
             h1_shapes=("white", "lpf0.8", "rtl3dB", "ar+0.1", "ar-0.1", "pink", "tilt1.0+", "tilt1.0-", "tilt2.0-", "tilt0.5-"),
             snrs=SNRS, beta=0.35, mods=sim.MODS, signal="modulated", multipath=True):
    _, rcal, rte = sim.streams(10_000 + seed)
    res = {"N": N, "sps": sps, "alpha": ALPHA, "signal": signal}
    raw = {}
    # ---------------- calibration
    cal_white = _scores(score_fn, lambda m: sim.white(m, N, rcal), n_cal)
    thr_nom = float(np.quantile(cal_white, 1 - ALPHA))
    cal_q = {}
    for sh in all_shapes():
        s = _scores(score_fn, lambda m: _gen_shape(sh, m, N, rcal), n_cal_shape)
        cal_q[sh] = float(np.quantile(s, 1 - ALPHA))
    sets = dict(sim.UNCERTAINTY_SETS)
    sets.update(tilt_sets())
    thr = {"nominal": thr_nom}
    for U, members in sets.items():
        thr[U] = max(cal_q[m] for m in members)
    res["thresholds"] = thr
    res["cal_q95"] = cal_q
    # ---------------- H0 (test)
    pf = {}
    for sh in all_shapes():
        s = _scores(score_fn, lambda m: _gen_shape(sh, m, N, rte), n_h0)
        raw[f"h0_{sh}"] = s
        pf[sh] = {k: float(np.mean(s > t)) for k, t in thr.items()}
    res["Pf"] = pf
    # ---------------- H1 (test): signals and surrogates in white noise, signals in shaped noise
    pd, pd_sur = {}, {}
    if signal == "OFDM":
        def sig_gen(m):
            return sim.ofdm(m, N, rte, multipath=multipath)
        def sur_gen(m):
            return sim.ofdm_surrogate(m, N, rte, multipath=multipath)
    else:
        def sig_gen(m):
            return sim.modulated(m, N, rte, sps=sps, beta=beta, mods=mods)
        def sur_gen(m):
            return sim.surrogate(m, N, rte, sps=sps, beta=beta)

    for d in snrs:
        g = np.sqrt(10 ** (d / 10))
        s = _scores(score_fn, lambda m: sig_gen(m) * g + sim.white(m, N, rte), n_h0)
        u = _scores(score_fn, lambda m: sur_gen(m) * g + sim.white(m, N, rte), n_h0)
        raw[f"h1_white_{d}"] = s
        raw[f"sur_white_{d}"] = u
        pd[str(d)] = {"white": {k: float(np.mean(s > t)) for k, t in thr.items()}}
        pd_sur[str(d)] = {k: float(np.mean(u > t)) for k, t in thr.items()}
        for sh in h1_shapes:
            if sh == "white":
                continue
            s2 = _scores(score_fn, lambda m: sig_gen(m) * g + _gen_shape(sh, m, N, rte), n_h1)
            raw[f"h1_{sh}_{d}"] = s2
            pd[str(d)][sh] = {k: float(np.mean(s2 > t)) for k, t in thr.items()}
    res["Pd"] = pd
    res["Pd_surrogate"] = pd_sur
    # worst-case Pd over the H1 shapes that belong to each set (only shapes evaluated under H1)
    wc = {}
    for U, members in sets.items():
        mem = [m for m in members if m in h1_shapes]
        wc[U] = {str(d): float(min(pd[str(d)][m][U] for m in mem)) for d in snrs}
        wc[U + "_members"] = mem
    res["Pd_worstcase"] = wc
    # surrogate gap at the nominal threshold (mean over SNRs where signal Pd in (0.1, 0.9))
    gaps = [pd[str(d)]["white"]["nominal"] - pd_sur[str(d)]["nominal"] for d in snrs
            if 0.1 < pd[str(d)]["white"]["nominal"] < 0.9]
    res["surrogate_gap"] = float(np.mean(gaps)) if gaps else None
    res["surrogate_gap_n"] = len(gaps)
    return res, raw


# ============================================================================ remedy: reference whitening
def whitener(ref, smooth=3, method="movavg", n_cos=8):
    """Noise-PSD estimate from K reference windows (K, N) -> function whitening inputs (n, N).
    method 'movavg': circular moving average of the averaged periodogram over `smooth` bins.
    method 'cos'   : least-squares fit of the log-periodogram with n_cos even cosine terms (smooth receiver
                     shapes need few terms; the fit suppresses the per-bin estimation noise of a short reference).
                     The DC bin is excluded from the fit (DC blocking removes it)."""
    P = np.mean(np.abs(np.fft.fft(ref, axis=1)) ** 2, axis=0)
    N = len(P)
    if method == "cos":
        f = np.fft.fftfreq(N)
        A = np.stack([np.cos(2 * np.pi * k * f) for k in range(n_cos)] + [np.sin(2 * np.pi * k * f) for k in range(1, n_cos)], 1)
        use = np.abs(f) > 0.5 / N
        coef, *_ = np.linalg.lstsq(A[use], np.log(np.maximum(P[use], 1e-12)), rcond=None)
        P = np.exp(A @ coef)
    elif smooth > 1:
        # circular moving average over DFT bins (the fft-ordered grid is circularly contiguous)
        k = np.ones(smooth) / smooth
        P = np.convolve(np.concatenate([P[-smooth:], P, P[:smooth]]), k, mode="same")[smooth:-smooth]
    P = P / P.mean()
    inv = 1.0 / np.sqrt(np.maximum(P, 1e-6))

    def w(x):
        return np.fft.ifft(np.fft.fft(x, axis=1) * inv[None, :], axis=1)
    return w


def evaluate_whitening(score_fn, N, sps, seed, Ks=(16, 64, 256, 1024), methods=("cos", "movavg"),
                       shapes=("white", "lpf0.8", "rtl3dB", "ar+0.2", "pink", "tilt1.0+"),
                       snrs=(-14, -12, -10, -8), n_h0=4000, n_h1=2000, beta=0.35):
    """For each shape: draw K reference noise windows (calibration stream), estimate the noise PSD
    (method 'cos': 16-term cosine fit of the log-periodogram; 'movavg': 3-bin moving average), whiten test
    inputs, apply the detector with its nominal (white-noise) threshold; report Pf and Pd (signal + same-shape noise)."""
    _, rcal, rte = sim.streams(20_000 + seed)
    cal_white = _scores(score_fn, lambda m: sim.white(m, N, rcal), 20000)
    thr = float(np.quantile(cal_white, 1 - ALPHA))
    out = {"thr": thr}
    for method in methods:
        for K in Ks:
            if method == "movavg" and K not in (16, 1024):
                continue
            for sh in shapes:
                ref = _gen_shape(sh, K, N, rcal)
                w = whitener(ref, method=method, n_cos=16)
                f = lambda x: score_fn(w(x))
                s0 = _scores(f, lambda m: _gen_shape(sh, m, N, rte), n_h0)
                r = {"Pf": float(np.mean(s0 > thr))}
                for d in snrs:
                    g = np.sqrt(10 ** (d / 10))
                    s1 = _scores(f, lambda m: sim.modulated(m, N, rte, sps=sps, beta=beta) * g + _gen_shape(sh, m, N, rte), n_h1)
                    r[str(d)] = float(np.mean(s1 > thr))
                out[f"K{K}_{method}_{sh}"] = r
    return out

