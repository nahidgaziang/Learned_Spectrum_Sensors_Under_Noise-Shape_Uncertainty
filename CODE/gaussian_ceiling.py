"""Step 9 of the project plan (U5): the Gaussian ceiling, i.e. the detection rate of the most powerful
scale-invariant (MPI) test of the Gaussian surrogate at each SNR (Proposition 4: x^H [I - (I + g Sigma)^-1] x / x^H x,
exact matrix form, Sigma = covariance of the power-normalised signal windows, level alpha under white noise).
For every invariant test phi:  P_d(phi; s_G, g) <= C(g) := P_d(MPI_g; s_G, g)  (Neyman-Pearson on the maximal invariant).
Uses: (i) efficiency of the learned sensors: the SNR loss in dB relative to the ceiling (how much SNR the ceiling
needs to reach the network's detection rate); (ii) certificate: P_d(phi; s, g) > C(g) + margin implies
P_d(phi; s, g) - P_d(phi; s_G, g) > margin, i.e. phi uses non-Gaussian structure, without running surrogates; under
noise-shape uncertainty the ceiling of the worst-case detection rate is alpha at and below the wall (Proposition 1),
so the wall tests' escape results are such certificates (summarised here from RESULTS/wall_test_escape_s*.json).
N = 128 (same SNR grid and nominal 5 % level as the main experiments). Writes RESULTS/gaussian_ceiling.json."""
import glob, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "RESULTS")
N, M_COV, N_CAL, N_TE, CH, ALPHA = 128, 400_000, 200_000, 100_000, 20_000, 0.05
ARCHS = ("CLDNN", "ResNet1D", "CovCNN", "STFTCNN", "GapCNN")


def main():
    t0 = time.time()
    rng = np.random.default_rng(777)
    S = np.zeros((N, N), complex)
    for _ in range(M_COV // CH):
        x = sim.modulated(CH, N, rng)
        S += x.T @ x.conj()
    S = S / M_COV
    S = 0.5 * (S + S.conj().T)
    L = np.linalg.cholesky(S + 1e-10 * np.eye(N))
    grid = np.arange(-24.0, 2.01, 1.0)
    ceil, pd_sig, pd_paper = [], [], []
    for d in grid:
        g = 10 ** (d / 10)
        A = np.eye(N) - np.linalg.inv(np.eye(N) + g * S)
        stat = lambda x: np.real(np.sum((x.conj() @ A) * x, 1)) / np.sum(np.abs(x) ** 2, 1)   # x^H A x / x^H x (rows)
        cal = np.concatenate([stat(sim.white(CH, N, rng)) for _ in range(N_CAL // CH)])
        thr = np.quantile(cal, 1 - ALPHA)
        a = np.sqrt(g)
        sG, sS, sP = [], [], []
        for _ in range(N_TE // CH):
            z = (rng.standard_normal((CH, N)) + 1j * rng.standard_normal((CH, N))) / np.sqrt(2)
            sG.append(stat(a * (z @ L.T) + sim.white(CH, N, rng)) > thr)
            sS.append(stat(a * sim.modulated(CH, N, rng) + sim.white(CH, N, rng)) > thr)
            sP.append(stat(a * sim.surrogate(CH, N, rng) + sim.white(CH, N, rng)) > thr)
        ceil.append(float(np.mean(np.concatenate(sG)))); pd_sig.append(float(np.mean(np.concatenate(sS))))
        pd_paper.append(float(np.mean(np.concatenate(sP))))
        print(f"{d:+.0f} dB: ceiling {ceil[-1]:.4f}  MPI on signals {pd_sig[-1]:.4f}  on paper surrogates {pd_paper[-1]:.4f}", flush=True)
    ceil = np.array(ceil)

    def snr_needed(pd):          # SNR at which the ceiling reaches pd (monotone interpolation)
        if pd <= ceil[0]:
            return float(grid[0])
        if pd >= ceil[-1]:
            return float(grid[-1])
        return float(np.interp(pd, ceil, grid))
    eff = {}
    for arch in ARCHS + ("DetectNetExact",):
        fs = sorted(glob.glob(os.path.join(RES, "runs", f"{arch}_WHITE_N128_sps8_s?.json"))) if arch != "DetectNetExact" else \
            sorted(glob.glob(os.path.join(RES, "runs", "DetectNetExact_WHITE_N128_sps8_s?_QAM16.json")))
        rows = []
        for f in fs:
            e = json.load(open(f))["eval"]
            for d in (-14, -12, -10):
                pd = e["Pd"][str(d)]["white"]["nominal"]
                rows.append({"run": os.path.basename(f)[:-5], "snr_db": d, "pd": pd, "ceiling": float(np.interp(d, grid, ceil)),
                             "snr_loss_db": d - snr_needed(pd)})
        eff[arch] = rows
    # certificates under shape uncertainty: worst-case Pd below the wall vs the ceiling alpha (+ binomial margin)
    cert = {}
    for f in sorted(glob.glob(os.path.join(RES, "wall_test_escape_s*.json"))):
        dd = json.load(open(f))
        for key, v in dd.items():
            if not key.startswith("N"):
                continue
            for det, pdv in v["Pd"]["-2"].items():
                cert.setdefault(f"{key}|{det}", []).append(pdv["worst"])
    cert = {k: {"worst_pd_mean": float(np.mean(v)), "n_seeds": len(v),
                "exceeds_ceiling": bool(np.mean(v) > ALPHA + 2 * np.sqrt(ALPHA * (1 - ALPHA) / 2000))} for k, v in cert.items()}
    out = {"_note": "U5 Gaussian ceiling (gaussian_ceiling.py).", "grid_db": grid.tolist(), "ceiling": ceil.tolist(),
           "mpi_on_signals": pd_sig, "mpi_on_paper_surrogates": pd_paper, "efficiency": eff, "certificates_below_wall": cert,
           "runtime_s": time.time() - t0}
    json.dump(out, open(os.path.join(RES, "gaussian_ceiling.json"), "w"), indent=1)
    for arch, rows in eff.items():
        lo = [r["snr_loss_db"] for r in rows]
        print(f"{arch:16s} SNR loss vs ceiling (dB): {min(lo):+.2f} .. {max(lo):+.2f}")
    for k, v in cert.items():
        print(k, v)


if __name__ == "__main__":
    main()
