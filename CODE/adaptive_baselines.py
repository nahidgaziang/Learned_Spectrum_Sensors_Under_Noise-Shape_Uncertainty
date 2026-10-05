"""Plan item D: classical adaptive detectors with K noise-only reference windows (CPU only).
Detectors (signal subspace H = DFT vectors of the in-band bins, |f| <= (1+beta)/(2 sps); sample covariance
R = (1/K) sum y y^H of the reference windows, diagonal loading 1e-3*tr(R)/N, plus 0.1*tr(R)/N when K < N):
  AMF   = x^H R^-1 H (H^H R^-1 H)^-1 H^H R^-1 x                (uses the reference power)
  ACE   = AMF / (x^H R^-1 x)                                     (scale-invariant, CFAR for K >= N)
  KELLY = AMF / (1 + x^H R^-1 x / K)                             (subspace GLRT form)
  ED-ref = x^H x / (tr(R)/N)                                     (energy detector with estimated noise power)
Synthetic part: for each noise shape and K, 20 independent reference draws; per draw 200 H0 test windows and 100 H1
windows per SNR; thresholds from white noise (same K, 20 draws x 1000 windows), P_f target 5 %.
Recorded part: RTL-SDR noise (v3 protocol: reference = windows 256..1279, test = next <= 2000), white-calibrated
thresholds (K = 1024), P_f per detector; P_d at -10 dB with simulated signals (semi-synthetic, as for the networks).
Writes RESULTS/adaptive_baselines.json."""
import glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim, evaluate as E, real_noise_eval as RN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N, SPS = 128, 8
f = np.fft.fftfreq(N)
hw = sim.occupancy(SPS) / 2
H = np.exp(2j * np.pi * np.outer(np.arange(N), f[np.abs(f) <= hw])) / np.sqrt(N)      # N x r
SHAPES = ["white", "lpf0.8", "rtl3dB", "ar+0.2", "pink", "tilt1.0+"]
SNRS = [-14, -12, -10, -8]
DETS = ["AMF", "ACE", "KELLY", "ED-ref"]


def stats(R, X, K):
    N_ = R.shape[0]
    load = 1e-3 * np.trace(R).real / N_ + (0.1 * np.trace(R).real / N_ if K < N_ else 0.0)
    Ri = np.linalg.inv(R + load * np.eye(N_))
    RiH = Ri @ H
    G = H.conj().T @ RiH
    q = X.conj() @ RiH                        # n x r  (rows: x^H R^-1 H)
    amf = np.real(np.einsum("ij,jk,ik->i", q, np.linalg.inv(G), q.conj()))
    norm = np.real(np.einsum("ij,jk,ik->i", X.conj(), Ri, X))
    return {"AMF": amf, "ACE": amf / norm, "KELLY": amf / (1 + norm / K),
            "ED-ref": np.mean(np.abs(X) ** 2, 1) / (np.trace(R).real / N_)}


def scm(Y):
    return (Y.T @ Y.conj()) / len(Y)          # (1/K) sum y y^H


def synthetic(Ks=(16, 64, 256, 1024), draws=20, n0=200, n1=100, seed=0):
    rng = np.random.default_rng(70_000 + seed)
    out = {}
    for K in Ks:
        cal = {d: [] for d in DETS}
        for _ in range(draws):
            R = scm(sim.white(K, N, rng))
            s = stats(R, sim.white(1000, N, rng), K)
            for d in DETS:
                cal[d].append(s[d])
        thr = {d: float(np.quantile(np.concatenate(cal[d]), 0.95)) for d in DETS}
        for sh in SHAPES:
            pf = {d: [] for d in DETS}
            pd = {d: {str(x): [] for x in SNRS} for d in DETS}
            for _ in range(draws):
                R = scm(E._gen_shape(sh, K, N, rng))
                s0 = stats(R, E._gen_shape(sh, n0, N, rng), K)
                for d in DETS:
                    pf[d].append(np.mean(s0[d] > thr[d]))
                for x in SNRS:
                    X = sim.modulated(n1, N, rng) * np.sqrt(10 ** (x / 10)) + E._gen_shape(sh, n1, N, rng)
                    s1 = stats(R, X, K)
                    for d in DETS:
                        pd[d][str(x)].append(np.mean(s1[d] > thr[d]))
            out[f"K{K}_{sh}"] = {d: {"Pf": float(np.mean(pf[d])), **{k: float(np.mean(v)) for k, v in pd[d].items()}} for d in DETS}
            print(f"K={K:5d} {sh:9s} " + "  ".join(f"{d}: Pf {out[f'K{K}_{sh}'][d]['Pf']:.3f} Pd(-10) {out[f'K{K}_{sh}'][d]['-10']:.2f}"
                                                  for d in DETS), flush=True)
    return out


def recorded(ref_start=256, K=1024, data_dir=None):
    rng = np.random.default_rng(77)
    cal = {d: [] for d in DETS}
    for _ in range(20):
        Y = sim.white(K, N, rng); Y = Y - Y.mean(1, keepdims=True)
        X = sim.white(1000, N, rng); X = X - X.mean(1, keepdims=True)
        s = stats(scm(Y), X, K)
        for d in DETS:
            cal[d].append(s[d])
    thr = {d: float(np.quantile(np.concatenate(cal[d]), 0.95)) for d in DETS}
    res = {}
    for fp in sorted(glob.glob(os.path.join(data_dir or RN.DATA, "*.npy"))):
        iq, meta = RN.load(fp)
        if not meta.get("sample_rate"):
            continue                                      # metadata-free files (probably simulated) are skipped here
        sh, _ = RN.psd_shape(iq)
        if sh["narrowband"]:
            continue
        w = RN.windows(iq, N)[ref_start:]
        k = min(K, len(w) // 2)
        R, test = scm(w[:k]), w[k: k + 2000]
        pn = float(np.mean(np.abs(test) ** 2))
        sig = sim.modulated(len(test), N, rng) * np.sqrt(pn * 0.1)
        sig = sig - sig.mean(1, keepdims=True)
        s0, s1 = stats(R, test, k), stats(R, test + sig, k)
        res[os.path.basename(fp)] = {"rolloff_db": sh["edge_rolloff_db"], "K": k,
                                     **{d: {"Pf": float(np.mean(s0[d] > thr[d])), "Pd_m10": float(np.mean(s1[d] > thr[d]))} for d in DETS}}
    summ = {d: {"Pf_mean": float(np.mean([v[d]["Pf"] for v in res.values()])), "Pf_max": float(np.max([v[d]["Pf"] for v in res.values()])),
                "Pd_m10_mean": float(np.mean([v[d]["Pd_m10"] for v in res.values()]))} for d in DETS}
    print("recorded:", json.dumps(summ), flush=True)
    return res, summ


if __name__ == "__main__":
    # usage: py -3.12 adaptive_baselines.py            -> synthetic K sweep + HF composite noise (R820T2 receiver)
    #        py -3.12 adaptive_baselines.py trevtron   -> recorded part only, on DATA/trevtron/noise (R828D receiver)
    if len(sys.argv) > 1 and sys.argv[1] == "trevtron":
        rec, summ = recorded(data_dir=os.path.join(ROOT, "DATA", "trevtron", "noise"))
        out, name = {"recorded": rec, "recorded_summary": summ, "N": N, "detectors": DETS}, "adaptive_baselines_trevtron.json"
    else:
        syn = synthetic()
        rec, summ = recorded()
        out, name = {"synthetic": syn, "recorded": rec, "recorded_summary": summ, "N": N, "detectors": DETS}, "adaptive_baselines.json"
    path = os.path.join(ROOT, "RESULTS", name)
    with open(path + ".tmp", "w") as fh:
        json.dump(out, fh, indent=1)
    os.replace(path + ".tmp", path)
