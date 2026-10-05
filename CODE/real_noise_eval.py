"""X10: false alarms of trained detectors on REAL receiver noise (public RTL-SDR recordings).

Data: noise class of huggingface.co/datasets/ishisan28/rf-signal-dataset (downloaded by fetch_real_noise.py).
Steps
 1. load each recording (dict: complex64 IQ, centre frequency, sample rate, ...), compute its Welch PSD
    (nfft=128) and two shape statistics: band-edge roll-off (dB) and spectral flatness; recordings whose
    PSD is flat within 0.3 dB everywhere are flagged 'flat' (likely synthetic AWGN) and reported separately.
 2. slice into non-overlapping windows of N samples, remove the per-window mean (DC blocking, applied to
    ALL detectors), and score with every detector.
 3. Pf at each detector's nominal threshold (white-noise calibration, simulated, as in the literature);
    Pf after reference whitening with K = 16 windows taken from the first part of the same recording.
 4. semi-synthetic Pd: simulated modulated signals added to real-noise windows at SNR -10 dB (power measured
    per recording), at the nominal threshold and at a threshold recalibrated on that recording's first windows.
Output: RESULTS/real_noise.json, TABLES/tab_real_noise.csv
"""
import glob, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim, detectors as D
from run import learned_score_fn, classical_score_fn, DEFAULTS
from evaluate import whitener

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "DATA", "real_noise_hf")
ALPHA = 0.05


def load(path):
    d = np.load(path, allow_pickle=True)
    if d.dtype == object and d.shape == ():
        d = d.item()
    if isinstance(d, dict):
        iq = None
        for k in ["iq", "IQ", "samples", "data", "signal", "x"]:
            if k in d:
                iq = np.asarray(d[k]); break
        if iq is None:
            iq = next(np.asarray(v) for v in d.values() if isinstance(v, np.ndarray) and v.size > 1000)
        meta = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in d.items() if not isinstance(v, np.ndarray)}
    else:
        iq, meta = np.asarray(d), {}
    iq = iq.astype(np.complex64).ravel()
    return iq, meta


def psd_shape(iq, nfft=128):
    n = len(iq) // nfft
    seg = iq[: n * nfft].reshape(n, nfft)
    seg = seg - seg.mean(1, keepdims=True)
    P = np.fft.fftshift(np.mean(np.abs(np.fft.fft(seg * np.hanning(nfft), axis=1)) ** 2, 0))
    P = P / np.median(P)
    f = np.fft.fftshift(np.fft.fftfreq(nfft))
    centre = P[np.abs(f) < 0.15].mean()
    edge = P[np.abs(f) > 0.42].mean()
    core = P[(np.abs(f) > 0.02)]                       # exclude DC bins
    roll = float(10 * np.log10(centre / edge))
    # narrowband components (spurs or weak emissions): bins exceeding a 9-bin running median by > 2 dB, excluding DC
    from scipy.signal import medfilt
    base = medfilt(P, 9)
    excess = 10 * np.log10(P / np.maximum(base, 1e-12))
    excess[np.abs(f) <= 0.02] = 0.0
    # 'flat' = band-edge roll-off within +-0.5 dB (periodogram estimation noise alone gives ~0.5 dB max/min ripple)
    return {"edge_rolloff_db": roll, "ripple_db": float(10 * np.log10(core.max() / core.min())), "flat": bool(abs(roll) < 0.5),
            "max_narrowband_excess_db": float(excess.max()), "narrowband": bool(excess.max() > 2.0)}, P


def windows(iq, N, max_win=4000):
    n = min(len(iq) // N, max_win)
    w = iq[: n * N].reshape(n, N).astype(np.complex128)
    return w - w.mean(1, keepdims=True)


def detectors_for(N, seeds=(0, 1, 2)):
    fns = {}
    for det in ["MME", "SFM", "CYC", "BAND"]:
        fns[det] = classical_score_fn(dict(DEFAULTS, kind="classical", detector=det, N=N, sps=8))
    for arch, rec in [("CLDNN", "WHITE"), ("ResNet1D", "WHITE"), ("CovCNN", "WHITE"), ("STFTCNN", "WHITE"), ("GapCNN", "WHITE"),
                      ("CLDNN", "PINK"), ("GapCNN", "RAND"), ("GapCNN", "SURR"),
                      ("GapCNN", "SURRX"), ("GapCNN", "SURRXM"), ("GapCNN", "RANDM"), ("GapCNN", "PINKM")]:
        for s in seeds:
            ck = os.path.join(ROOT, "EXPERIMENTS", "ckpt", f"{arch}_{rec}_N{N}_sps8_s{s}.pt")
            if not os.path.exists(ck):
                continue
            m = D.ARCHS[arch]().cuda(); m.load_state_dict(torch.load(ck, map_location="cuda")); m.eval()
            # EVAL_BS (default 4096 = protocol of v2/v3) bounds the GPU memory reserved by PyTorch's cache: with 4096 the
            # CovCNN/STFT forward passes reserve ~5.9 GB of the 6 GB card (observed 2026-10-02, constant, no leak);
            # 1024 keeps it well below. Scores can differ in the last float digits between batch sizes (cuDNN kernels).
            fns[f"{arch}-{rec}-s{s}"] = learned_score_fn(m, arch, bs=int(os.environ.get("EVAL_BS", "4096")))
    return fns


def main(N=128, K=16, ref_start=0, data_dir=None, out_name=None):
    """ref_start = 0 reproduces protocol v2 exactly. ref_start = 256 (v3) skips the RTL start-up transient
    (~32 ms after tuning; see empirical_uncertainty.py): reference = windows ref_start .. ref_start+K_big-1,
    test = the following <= 2000 windows. data_dir/out_name allow other receivers (e.g. DATA/trevtron/noise)."""
    files = sorted(glob.glob(os.path.join(data_dir or DATA, "*.npy")))
    rng = np.random.default_rng(77)
    fns = detectors_for(N)
    # nominal thresholds on simulated white noise (the literature's protocol), with the SAME per-window DC
    # blocking that is applied to the real windows, so that preprocessing cannot create a mismatch
    wcal = sim.white(20000, N, rng)
    wcal = wcal - wcal.mean(1, keepdims=True)
    thr = {k: float(np.quantile(f(wcal), 1 - ALPHA)) for k, f in fns.items()}
    out = {"N": N, "protocol": "calibrate on the first windows of a recording (K=16 or K_big=min(1024, n/2)), test on the "
                                "following windows (at most 2000); whitening = 16-term cosine fit of the log-periodogram",
           "recordings": {}, "thresholds": thr}
    for fp in files:
        iq, meta = load(fp)
        shape, _ = psd_shape(iq)
        w = windows(iq, N)[ref_start:]
        kbig = 1024 if len(w) >= 2048 else len(w) // 2
        test = w[kbig: kbig + 2000]
        wh16 = whitener(w[:16], method="cos", n_cos=16)
        whbig = whitener(w[:kbig], method="cos", n_cos=16)
        pnoise = float(np.mean(np.abs(test) ** 2))
        sig = sim.modulated(len(test), N, rng) * np.sqrt(pnoise * 10 ** (-10 / 10))
        sig = sig - sig.mean(1, keepdims=True)          # same DC blocking as the noise windows
        group = "live_rtl" if meta.get("sample_rate") else "unlabelled"
        rec = {"meta": {k: str(v) for k, v in meta.items()}, "group": group, "shape": shape, "n_test": int(len(test)), "K_big": kbig,
               "ref_start": ref_start,
               "Pf": {}, "Pf_w16": {}, "Pf_wbig": {}, "Pd_m10_nominal": {}, "Pd_m10_wbig": {}}
        for k, f in fns.items():
            rec["Pf"][k] = float(np.mean(f(test) > thr[k]))
            rec["Pf_w16"][k] = float(np.mean(f(wh16(test)) > thr[k]))
            rec["Pf_wbig"][k] = float(np.mean(f(whbig(test)) > thr[k]))
            rec["Pd_m10_nominal"][k] = float(np.mean(f(test + sig) > thr[k]))
            rec["Pd_m10_wbig"][k] = float(np.mean(f(whbig(test + sig)) > thr[k]))
        out["recordings"][os.path.basename(fp)] = rec
        print(os.path.basename(fp), round(shape["edge_rolloff_db"], 2), {k: (round(rec["Pf"][k], 3), round(rec["Pf_wbig"][k], 3))
              for k in ("BAND", "CYC", "CLDNN-WHITE-s0", "GapCNN-SURR-s0") if k in rec["Pf"]}, flush=True)
    out["protocol"] += f"; reference starts at window {ref_start}"
    path = os.path.join(ROOT, "RESULTS", out_name or f"real_noise_v2_N{N}.json")
    with open(path + ".tmp", "w") as fh:
        json.dump(out, fh, indent=1)
    os.replace(path + ".tmp", path)


if __name__ == "__main__":
    # usage: py -3.12 real_noise_eval.py [N] [ref_start] [data_dir] [out_name]
    a = sys.argv[1:]
    main(int(a[0]) if a else 128, ref_start=int(a[1]) if len(a) > 1 else 0,
         data_dir=a[2] if len(a) > 2 else None, out_name=a[3] if len(a) > 3 else None)
