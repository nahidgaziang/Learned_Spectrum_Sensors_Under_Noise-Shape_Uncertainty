"""Feasibility probe (planning): the HF dataset also holds recorded ADS-B, FM-broadcast and ISM-sensor signals from the
same RTL-SDR setup. Download 3 files per class to DATA/real_signals_hf/<class>/ and report per file: sample rate,
centre frequency, occupied bandwidth (fraction of band holding 90 % of the excess power over the noise floor), burstiness
(fraction of N=128 windows whose power exceeds the median window power by > 3 dB), and peak in-band SNR estimate.
Writes RESULTS/real_signals_probe.json."""
import json, os, sys, urllib.request
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_noise_eval as RN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "https://huggingface.co/datasets/ishisan28/rf-signal-dataset/resolve/main"
UA = {"User-Agent": "Mozilla/5.0 (research)"}
CLASSES = {"FM_broadcast": ["FM_01.npy", "FM_02.npy", "FM_03.npy"], "ADS_B": ["ADSB_01.npy", "ADSB_02.npy", "ADSB_03.npy"],
           "ISM_sensors": ["ISM_01.npy", "ISM_03.npy", "ISM_04.npy"]}
out = {}
for cls, files in CLASSES.items():
    d = os.path.join(ROOT, "DATA", "real_signals_hf", cls)
    os.makedirs(d, exist_ok=True)
    for fn in files:
        p = os.path.join(d, fn)
        if not os.path.exists(p):
            with urllib.request.urlopen(urllib.request.Request(f"{BASE}/{cls}/{fn}", headers=UA), timeout=120) as r:
                open(p + ".tmp", "wb").write(r.read())
            os.replace(p + ".tmp", p)
        iq, meta = RN.load(p)
        w = RN.windows(iq, 128)
        pw = np.mean(np.abs(w) ** 2, axis=1)
        burst = float(np.mean(pw > 2 * np.median(pw)))
        P = np.fft.fftshift(np.mean(np.abs(np.fft.fft(w, axis=1)) ** 2, axis=0))
        floor = np.quantile(P, 0.2)
        exc = np.maximum(P - floor, 0)
        srt = np.sort(exc)[::-1]
        occ = float(np.searchsorted(np.cumsum(srt) / max(srt.sum(), 1e-12), 0.9) + 1) / len(P)
        snr_db = float(10 * np.log10(max(exc.sum(), 1e-12) / (floor * len(P))))
        out[f"{cls}/{fn}"] = {"meta": {k: str(v) for k, v in meta.items()}, "n_windows": int(len(w)), "burst_fraction": burst,
                              "occupied_fraction_90": occ, "snr_db_estimate": snr_db}
        print(f"{cls:13s} {fn:12s} fs={meta.get('sample_rate')} fc={meta.get('center_freq')} windows={len(w)} "
              f"burst={burst:.2f} occupancy={occ:.2f} SNR~{snr_db:.1f} dB")
json.dump(out, open(os.path.join(ROOT, "RESULTS", "real_signals_probe.json"), "w"), indent=1)
