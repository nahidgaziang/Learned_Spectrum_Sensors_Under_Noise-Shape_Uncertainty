"""Step 7 of the project plan (U6): off-band blindness and its mechanism.

Question: why do the white-trained networks miss real signals that lie away from the band seen in training
(Section VII-E: NOAA, pager, FRS/GMRS on R828D)? Section VI shows they behave as the LMPI test, whose template is the
training PSD p. A second-order statistic sum_k w_k |X_k|^2 / sum_k |X_k|^2 responds to a signal with PSD p' through
the centred overlap  O = <w - mean(w), p'> / <w - mean(w), p_train>  (first order in SNR), so an off-band signal at
SNR g should be detected like a centred signal at SNR g * O, and not at all if O <= 0.
Test: modulated signals with carrier offset f0 in {0, .05, ..., .35} (cycles/sample) and bandwidth sps in {4, 8, 16}
at SNR in {-10, -6, -2} dB in white noise (N = 128), scored by the white-trained networks (5 architectures x 3 seeds),
the DetectNet replica (3 seeds), the band ratio (training band) and the LMPI test (training template), all at their
nominal thresholds on identical windows. Prediction (no fitting): each detector's centred-signal P_d curve (sps 8,
f0 = 0, SNR -20..+6 dB) evaluated at SNR + 10 log10(O), with O from the detector's own implicit template
(RESULTS/lmpi_equivalence.json; the LMPI uses p_train, the band ratio its indicator).
GPU inference (one job; run driver_watch.ps1). Writes RESULTS/offband_sweep.json.   usage: py -3.12 offband_sweep.py"""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim, detectors as D, wall as W
from run import learned_score_fn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "RESULTS")
N, NW = 128, 2000
OFFS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35)
SPSS = (4, 8, 16)
SNRS = (-10, -6, -2)
REF_SNRS = tuple(range(-20, 7, 2))
ARCHS = ("CLDNN", "ResNet1D", "CovCNN", "STFTCNN", "GapCNN")


def shifted(n, sps, f0, rng):
    x = sim.modulated(n, N, rng, sps=sps)
    return x * np.exp(2j * np.pi * f0 * np.arange(N))[None, :]


def mean_periodogram(sps, f0, n=8000, seed=99):
    x = shifted(n, sps, f0, np.random.default_rng(seed))
    P = np.mean(np.abs(np.fft.fft(x, axis=1)) ** 2, 0)
    return P / P.mean()


def main():
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    LE = json.load(open(os.path.join(RES, "lmpi_equivalence.json")))["models"]
    p_train = mean_periodogram(8, 0.0)
    order = np.argsort(np.fft.fftfreq(N), kind="stable")          # template vectors are stored in ascending frequency
    dets, templ, thr = {}, {}, {}
    for arch in ARCHS + ("DetectNetExact",):
        for s in (0, 1, 2):
            name = f"{arch}_WHITE_N128_sps8_s{s}" + ("_QAM16" if arch == "DetectNetExact" else "")
            ck = os.path.join(ROOT, "EXPERIMENTS", "ckpt", name + ".pt")
            m = D.ARCHS[arch]().to(dev)
            m.load_state_dict(torch.load(ck, map_location=dev))
            m.eval()
            dets[name] = learned_score_fn(m, arch, dev)
            thr[name] = json.load(open(os.path.join(RES, "runs", name + ".json")))["eval"]["thresholds"]["nominal"]
            t = np.empty(N)
            t[order] = np.array(LE[name]["template_vec"])
            templ[name] = t
    w = p_train / p_train.sum()
    dets["LMPI"] = lambda x: (np.abs(np.fft.fft(x, axis=1)) ** 2 @ w) / np.sum(np.abs(np.fft.fft(x, axis=1)) ** 2, 1)
    templ["LMPI"] = p_train
    hw = sim.occupancy(8, 0.35) / 2
    dets["BAND"] = lambda x: D.band_ratio(x, half_width=hw)
    templ["BAND"] = (np.abs(np.fft.fftfreq(N)) <= hw).astype(float)
    rng = np.random.default_rng(31)
    cal = sim.white(20000, N, rng)
    for k in ("LMPI", "BAND"):
        thr[k] = float(np.quantile(dets[k](cal), 0.95))
    # reference curves (centred, sps 8)
    ref = {k: [] for k in dets}
    for d in REF_SNRS:
        x = np.sqrt(10 ** (d / 10)) * shifted(NW, 8, 0.0, rng) + sim.white(NW, N, rng)
        for k, f in dets.items():
            ref[k].append(float(np.mean(f(x) > thr[k])))
    print(f"reference curves done ({time.time() - t0:.0f}s)", flush=True)
    rows = []
    for sps in SPSS:
        for f0 in OFFS:
            pt = mean_periodogram(sps, f0)
            ov = {}
            for k, t in templ.items():
                tc = t - t.mean()
                ov[k] = float(tc @ pt / (tc @ p_train))
            for d in SNRS:
                x = np.sqrt(10 ** (d / 10)) * shifted(NW, sps, f0, rng) + sim.white(NW, N, rng)
                r = {"sps": sps, "f0": f0, "snr_db": d, "overlap": ov, "pd": {}, "pd_pred": {}}
                for k, f in dets.items():
                    r["pd"][k] = float(np.mean(f(x) > thr[k]))
                    if ov[k] > 0:
                        r["pd_pred"][k] = float(np.interp(d + 10 * np.log10(ov[k]), REF_SNRS, ref[k]))
                    else:
                        r["pd_pred"][k] = ref[k][0] if ov[k] == 0 else 0.0
                rows.append(r)
            print(f"sps {sps} f0 {f0:.2f}: overlap LMPI {ov['LMPI']:+.3f}  Pd(-6) LMPI {rows[-2]['pd']['LMPI']:.3f} "
                  f"GapCNN_s0 {rows[-2]['pd']['GapCNN_WHITE_N128_sps8_s0']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    # summary: prediction error per detector family
    summ = {}
    for k in dets:
        e = [abs(r["pd"][k] - r["pd_pred"][k]) for r in rows]
        summ[k] = {"mae": float(np.mean(e)), "max_err": float(np.max(e))}
    out = {"_note": "U6 off-band sweep (offband_sweep.py).", "thresholds": thr, "reference": {"snrs": REF_SNRS, "pd": ref},
           "rows": rows, "summary": summ, "runtime_s": time.time() - t0}
    json.dump(out, open(os.path.join(RES, "offband_sweep.json"), "w"), indent=1)
    for k, v in summ.items():
        print(f"{k:40s} MAE {v['mae']:.3f} max {v['max_err']:.3f}")


if __name__ == "__main__":
    main()
