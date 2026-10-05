"""Step 10 of the project plan (U7): what does a learned sensor learn first? Training dynamics of the GAP CNN
under the WHITE recipe (spectral shape separates the classes), SURRXM (40 % signal-matched surrogates among the
negatives, same noise mixture in both classes) and PAIRED (80 % exact phase-randomised twins of the positives), N = 128,
seeds 0-2, otherwise the settings of run.py (60 000 examples, 12 epochs, Adam 1e-3, cosine schedule, batch 128).
At log-spaced optimisation steps the model is evaluated on a FIXED probe set (independent stream):
  P_d on signals and on Gaussian surrogates at -10 dB and 0 dB (threshold: 95 % quantile on 4000 white windows),
  surrogate gaps, Spearman rho with the LMPI statistic on the -10 dB signal windows, and the implicit spectral template
  (ridge fit on the normalised periodogram; cosine with the signal PSD p and with the band indicator).
Prediction (Proposition 3 + distributional simplicity bias): the WHITE model acquires the LMPI-like second-order
statistic early and never a surrogate gap; recipes that remove the spectral cue acquire a gap later, starting at high
SNR. Training data are generated exactly as in run.py (run.training_set), so the final models correspond to the
recipes in the paper (the dynamics runs are re-trainings; their checkpoints are not used elsewhere).
GPU (one job; driver_watch.ps1). Writes RESULTS/dynamics.json.   usage: py -3.12 dynamics.py"""
import json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim, detectors as D, wall as W, run as RUN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "RESULTS", "dynamics.json")
N = 128


def probe_set():
    rng = np.random.default_rng(424242)
    P = {"h0": sim.white(4000, N, rng)}
    for d in (-10, 0):
        g = np.sqrt(10 ** (d / 10))
        P[f"sig{d}"] = g * sim.modulated(4000, N, rng) + sim.white(4000, N, rng)
        P[f"sur{d}"] = g * sim.surrogate(4000, N, rng) + sim.white(4000, N, rng)
    return P


def main():
    t0 = time.time()
    dev = "cuda"
    P = probe_set()
    p = W.signal_psd(N, sps=8)
    w = p / p.sum()
    per = {k: np.abs(np.fft.fft(v, axis=1)) ** 2 for k, v in P.items()}
    lmpi = {k: (v @ w) / v.sum(1) for k, v in per.items()}
    Xn = {k: v / v.sum(1, keepdims=True) for k, v in per.items()}
    pool = np.concatenate([Xn["h0"], Xn["sig-10"], Xn["sur-10"]])
    Xc = pool - pool.mean(0)
    G = Xc.T @ Xc
    Ginv = np.linalg.inv(G + 1e-6 * np.trace(G) / N * np.eye(N))
    band = (np.abs(np.fft.fftfreq(N)) <= sim.occupancy(8, 0.35) / 2).astype(float)
    cos = lambda u, v: float((u - u.mean()) @ (v - v.mean()) / (np.linalg.norm(u - u.mean()) * np.linalg.norm(v - v.mean())))
    reps = {k: torch.from_numpy(D.norm_iq(v)) for k, v in P.items()}

    def evaluate(model):
        model.eval()
        with torch.no_grad():
            sc = {k: np.concatenate([model(v[i:i + 2048].to(dev)).float().cpu().numpy() for i in range(0, len(v), 2048)])
                  for k, v in reps.items()}
        model.train()
        thr = np.quantile(sc["h0"], 0.95)
        m = {f"pd_{k}": float(np.mean(sc[k] > thr)) for k in sc if k != "h0"}
        m["gap_-10"] = m["pd_sig-10"] - m["pd_sur-10"]
        m["gap_0"] = m["pd_sig0"] - m["pd_sur0"]
        m["rho_lmpi"] = float(stats.spearmanr(sc["sig-10"], lmpi["sig-10"])[0])
        y = np.concatenate([sc["h0"], sc["sig-10"], sc["sur-10"]]).astype(float)
        beta = Ginv @ (Xc.T @ (y - y.mean()))
        t = beta - beta.mean()
        m["cos_p"], m["cos_band"] = cos(t, p), cos(t, band)
        r = y - y.mean() - Xc @ beta
        m["r2"] = float(1 - r @ r / np.sum((y - y.mean()) ** 2))
        return m

    out = {"_note": "U7 training dynamics (dynamics.py).", "runs": {}}
    for recipe in ("WHITE", "SURRXM", "PAIRED"):
        for seed in (0, 1, 2):
            c = dict(RUN.DEFAULTS); c.update({"recipe": recipe, "seed": seed})
            rtr, _, _ = sim.streams(c["seed"])
            torch.manual_seed(c["seed"])
            X, y = RUN.training_set(c, rtr)
            nval = len(y) // 20
            Xt, yt = X[nval:], y[nval:]
            model = D.ARCHS["GapCNN"]().to(dev)
            opt = torch.optim.Adam(model.parameters(), c["lr"])
            per_ep = int(np.ceil(len(yt) / c["batch"]))
            steps_total = c["epochs"] * per_ep
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps_total)
            lf = nn.BCEWithLogitsLoss()
            marks = sorted(set([0, 10, 20, 50, 100, 200, 300] + [per_ep * e for e in range(1, c["epochs"] + 1)]))
            hist, step = [], 0
            hist.append({"step": 0, **evaluate(model)})
            for ep in range(c["epochs"]):
                perm = torch.randperm(len(yt))
                for i in range(0, len(yt), c["batch"]):
                    idx = perm[i:i + c["batch"]]
                    xb, yb = Xt[idx].to(dev), yt[idx].to(dev)
                    opt.zero_grad(set_to_none=True)
                    loss = lf(model(xb), yb)
                    loss.backward()
                    opt.step()
                    sch.step()
                    step += 1
                    if step in marks:
                        hist.append({"step": step, "epoch": step / per_ep, **evaluate(model)})
            key = f"GapCNN_{recipe}_s{seed}"
            out["runs"][key] = hist
            h = hist[-1]
            print(f"{key}: final rho_LMPI {h['rho_lmpi']:.3f} cos_p {h['cos_p']:.3f} gap(-10) {100*h['gap_-10']:+.1f} "
                  f"gap(0) {100*h['gap_0']:+.1f} Pd(-10) {h['pd_sig-10']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
            del X, y, Xt, yt
            json.dump(out, open(OUT + ".tmp", "w"), indent=1)
            os.replace(OUT + ".tmp", OUT)
    out["runtime_s"] = time.time() - t0
    json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
