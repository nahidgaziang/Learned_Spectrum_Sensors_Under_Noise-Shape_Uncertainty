"""Plan F.1/F.3 (the project plan T6): surrogate test on REAL signals mixed into REAL noise of the same receiver.

Signals: TrevTron/rtl-ml-dataset v2 captures (RTL-SDR Blog V4, R828D; MIT licence) in DATA/trevtron/<class>/ and HF
composite captures (evaluation only) in DATA/hf_signals/<class>/. Noise: the same dataset's noise captures
(DATA/trevtron/noise; DATA/real_noise_hf live recordings with metadata).
Per capture (first 256 windows skipped = RTL start-up transient; windows of N = 128 with per-window DC removal):
  occupancy mask S = bins whose long-term periodogram exceeds the floor (20th percentile over non-DC bins) by >= 6 dB;
  signal windows = windows whose power in S is >= 4x (6 dB) the floor's expected power in S; at most W per capture.
  A capture with an empty S or fewer than MIN_WIN signal windows is reported as 'no detectable signal' and skipped.
  The windows still contain the capture's own receiver noise (reported as the capture's in-mask SNR estimate).
Surrogates (structure removed, second-order content kept): MAIN 'sur' = context FT surrogate (phase-randomise the
contiguous 5-window segment around the window, keep the middle window: a linear segment with the local PSD, like the
paper's PSD-matched Gaussian surrogate); 'sur_win' = per-window FT surrogate IFFT(FFT(s) * exp(j*phi_k)) (identical
periodogram, but the window becomes circular, which window-based detectors such as MME can notice).
Mixture: x = sqrt(g) * s / rms(s) + n / rms(n), n a real noise window of the same receiver (mixing pool), g on a grid
(nominal SNR; the true signal share is lower by the capture's own noise).
Thresholds: (1-ALPHA) quantile of each detector's score on a separate calibration pool of the same receiver's noise.
Detectors: real_noise_eval.detectors_for(128) (classical MME/SFM/CYC/BAND + learned nets, seeds 0-2) and the normalised
fourth moment kappa = mean|x|^4 / (mean|x|^2)^2 (Gaussian: 2) as structure references: KAPPA = |kappa - 2| (two-sided)
and KAPPALO = -kappa (constant-envelope signals). Real RTL noise windows are heavy-tailed (TrevTron: kappa 2.87 +- 0.76
vs 1.99 +- 0.17 for Gaussian), so the two-sided version has a high threshold; KAPPALO is the positive control for FM/FSK.
Stores per capture and SNR the detection COUNTS for signal and surrogate (so pooled curves, gaps and bootstrap CIs can
be recomputed without re-running). Gap per class = mean over SNRs with 0.1 < pooled Pd_sig < 0.9 of (Pd_sig - Pd_sur).
usage: py -3.12 real_signal_surrogate.py [--no-nets] [--max-captures M]   -> RESULTS/real_signal_surrogate[_nonets].json"""
import glob, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_noise_eval as RN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "DATA")
N, SKIP, ALPHA = 128, 256, 0.05
W, MIN_WIN = 100, 20                       # signal windows per capture (max), minimum to keep the capture
SNRS = list(range(-24, 5, 2))              # nominal dB
N_CAL, N_MIX = 20000, 20000                # noise windows per receiver for calibration / mixing (disjoint)
SOURCES = {  # receiver -> (noise dir, {class: signal dir})
    "trevtron_R828D": (os.path.join(DATA, "trevtron", "noise"),
                       {c: os.path.join(DATA, "trevtron", c) for c in
                        ["FM_broadcast", "NOAA_weather", "APRS", "pager", "ISM_sensors", "FRS_GMRS"]}),
    "hf_composite": (os.path.join(DATA, "real_noise_hf"),
                     {c: os.path.join(DATA, "hf_signals", c) for c in ["ADS_B", "FM_broadcast", "ISM_sensors"]}),
}


def kappa(x):
    p = np.abs(x) ** 2
    return np.mean(p ** 2, 1) / np.mean(p, 1) ** 2


def kappa_score(x):          # two-sided: any departure from the Gaussian value 2
    return np.abs(kappa(x) - 2.0)


def kappa_lo_score(x):       # one-sided: sub-Gaussian / constant envelope (FM, FSK). Real RTL noise windows are
    return -kappa(x)         # heavy-tailed (kappa ~2.9 +- 0.8 on TrevTron noise), which defeats the two-sided version


def surrogate(s, rng):
    """Per-window FT surrogate: exactly the same periodogram, but the window becomes periodic (circular)."""
    S = np.fft.fft(s, axis=1)
    return np.fft.ifft(S * np.exp(2j * np.pi * rng.random(S.shape)), axis=1)


def surrogate_ctx(iq, idx, rng, ctx=2):
    """Context FT surrogate: phase-randomise the contiguous segment of 2*ctx+1 windows around window idx (raw samples)
    and keep the middle window (per-window DC removed as for real windows). It is a linear (non-circular) segment of a
    Gaussian-like process with the local PSD, i.e. the PSD-matched surrogate of the paper, estimated locally."""
    out = np.empty((len(idx), N), complex)
    for j, i in enumerate(idx):
        a = (SKIP + i - ctx) * N
        seg = iq[a: a + (2 * ctx + 1) * N].astype(np.complex128)
        seg = seg - seg.mean()
        Sg = np.fft.fft(seg)
        sur = np.fft.ifft(Sg * np.exp(2j * np.pi * rng.random(len(Sg))))
        m = sur[ctx * N: (ctx + 1) * N]
        out[j] = m - m.mean()
    return out


def unit(x):
    return x / np.sqrt(np.mean(np.abs(x) ** 2, 1, keepdims=True))


def noise_pool(ndir, n_total, rng):
    """Real noise windows of one receiver, after the start-up transient; metadata-free files are skipped."""
    out = []
    files = sorted(glob.glob(os.path.join(ndir, "*.npy")))
    per = int(np.ceil(n_total / max(1, len(files)))) + 1
    for fp in files:
        iq, meta = RN.load(fp)
        if not meta.get("sample_rate"):
            continue
        w = RN.windows(iq, N)[SKIP:]
        out.append(w[rng.choice(len(w), min(per, len(w)), replace=False)])
    pool = np.concatenate(out)
    return unit(pool[rng.permutation(len(pool))])


def select(iq, rng):
    w = RN.windows(iq, N)[SKIP:]
    P = np.abs(np.fft.fft(w, axis=1)) ** 2 / N
    Pm = P.mean(0)
    nondc = np.arange(N) != 0
    floor = np.quantile(Pm[nondc], 0.2)
    S = nondc & (Pm >= floor * 10 ** 0.6)
    info = {"n_mask_bins": int(S.sum()), "peak_db": float(10 * np.log10(Pm[nondc].max() / floor))}
    if S.sum() == 0:
        return None, info
    ps = P[:, S].sum(1)
    keep = np.where(ps >= 4 * floor * S.sum())[0]
    keep = keep[(keep >= 2) & (keep < len(w) - 2)]          # room for the context surrogate
    info["n_signal_windows"] = int(len(keep))
    # in-mask SNR estimate of the capture's own signal windows: (mask power - floor) / total floor power
    if len(keep):
        info["own_snr_db"] = float(10 * np.log10(max(ps[keep].mean() - floor * S.sum(), 1e-12) / (floor * N)))
    if len(keep) < MIN_WIN:
        return None, info
    keep = np.sort(rng.choice(keep, min(W, len(keep)), replace=False))
    return (w[keep], keep), info


def main(nets=True, max_captures=None):
    rng = np.random.default_rng(2027)
    if nets:
        fns = RN.detectors_for(N)
    else:
        fns = {det: RN.classical_score_fn(dict(RN.DEFAULTS, kind="classical", detector=det, N=N, sps=8))
               for det in ["MME", "SFM", "CYC", "BAND"]}
    fns["KAPPA"] = kappa_score
    fns["KAPPALO"] = kappa_lo_score
    out = {"protocol": __doc__, "snrs": SNRS, "alpha": ALPHA, "W": W, "MIN_WIN": MIN_WIN, "receivers": {}}
    t0 = time.time()
    for rx, (ndir, classes) in SOURCES.items():
        if not glob.glob(os.path.join(ndir, "*.npy")):
            print(rx, "no noise files, skipped", flush=True)
            continue
        pool = noise_pool(ndir, N_CAL + N_MIX, rng)
        cal, mix = pool[:N_CAL], pool[N_CAL:N_CAL + N_MIX]
        thr = {k: float(np.quantile(f(cal), 1 - ALPHA)) for k, f in fns.items()}
        R = {"noise_dir": os.path.relpath(ndir, ROOT), "n_cal": int(len(cal)), "n_mix": int(len(mix)), "thresholds": thr,
             "classes": {}}
        for cls, sdir in classes.items():
            files = sorted(glob.glob(os.path.join(sdir, "*.npy")))[:max_captures]
            C = {"n_files": len(files), "captures": {}, "skipped": {}}
            for fp in files:
                iq, meta = RN.load(fp)
                sel, info = select(iq, rng)
                name = os.path.basename(fp)
                if sel is None:
                    C["skipped"][name] = info
                    continue
                s, idx = sel
                s = unit(s)
                sur_w = unit(surrogate(s, rng))                      # exact periodogram, circular
                sur_c = unit(surrogate_ctx(iq, idx, rng))             # local PSD, linear segment (main surrogate)
                chk = np.max(np.abs(np.abs(np.fft.fft(sur_w, axis=1)) - np.abs(np.fft.fft(s, axis=1))) / (np.abs(np.fft.fft(s, axis=1)) + 1e-9))
                assert chk < 1e-5, chk
                info["kappa_sig"] = float(kappa(s).mean())
                info["kappa_sur_ctx"] = float(kappa(sur_c).mean())
                n = mix[rng.choice(len(mix), len(s) * len(SNRS), replace=False)].reshape(len(SNRS), len(s), N)
                cnt = {k: {} for k in fns}
                for lab, z in [("sig", s), ("sur", sur_c), ("sur_win", sur_w)]:
                    x = np.concatenate([np.sqrt(10 ** (g / 10)) * z + n[i] for i, g in enumerate(SNRS)])   # same noise for all three
                    for k, f in fns.items():
                        cnt[k][lab] = (f(x) > thr[k]).reshape(len(SNRS), len(s)).sum(1).tolist()
                C["captures"][name] = {"n": int(len(s)), "info": info, "center_freq": str(meta.get("center_freq")), "counts": cnt}
            C["n_used"] = len(C["captures"])
            R["classes"][cls] = C
            print(f"{rx} {cls}: files {len(files)}, used {C['n_used']}, skipped {len(C['skipped'])}  ({time.time()-t0:.0f}s)", flush=True)
        out["receivers"][rx] = R
    path = os.path.join(ROOT, "RESULTS", "real_signal_surrogate.json" if nets else "real_signal_surrogate_nonets.json")
    with open(path + ".tmp", "w") as fh:
        json.dump(out, fh)
    os.replace(path + ".tmp", path)
    print("written", path, flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    mc = int(a[a.index("--max-captures") + 1]) if "--max-captures" in a else None
    main(nets="--no-nets" not in a, max_captures=mc)
