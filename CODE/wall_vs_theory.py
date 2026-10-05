"""J2: theory versus experiment for the noise-shape SNR wall, across uncertainty u and occupancy rho (CPU only).

For sps in {4, 8, 16} (rho ~ 0.34, 0.17, 0.084) and u in {0.5, 1, 2, 3} dB: build the extremal pair (c0, c1) of the
non-flat construction from the signal's PSD, set a robust threshold for the oracle band ratio (BAND) over
{white, c0, c1}, and measure its worst-case Pd (H1 noise in {white, c1}) at SNR = predicted wall + delta.
Predictions: (i) Pd <= alpha for delta <= 0 at every N; (ii) the 50 % crossing approaches delta = 0 from above as N grows.
The signal PSD is estimated once per sps at N0 = 2048 (8000 windows) and interpolated onto each grid, so the pair stays
exact on long grids. Output: RESULTS/wall_vs_theory.json (incremental, atomic)."""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim, detectors as D, wall as W

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALPHA = 0.05
DELTAS = (-1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
_P0 = {}


def psd_on_grid(N, sps, N0=2048):
    if sps not in _P0:
        _P0[sps] = W.signal_psd(N0, sps=sps, n=8000)
    f0 = np.fft.fftshift(np.fft.fftfreq(N0)); p0 = np.fft.fftshift(_P0[sps])
    f = np.fft.fftfreq(N)
    p = np.interp(f, np.concatenate([f0 - 1, f0, f0 + 1]), np.tile(p0, 3))
    return p / p.mean()


def scores(fn, gen, n, chunk):
    return np.concatenate([fn(gen(min(chunk, n - i))) for i in range(0, n, chunk)])


def main(seed=0, n_cal=2000, n_te=1500, Ns=(8192, 32768)):
    path = os.path.join(ROOT, "RESULTS", "wall_vs_theory.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    t0 = time.time()
    for sps in (8, 4, 16):
        hw = sim.occupancy(sps, 0.35) / 2
        fn = lambda x: D.band_ratio(x, half_width=hw)
        for u in (0.5, 1.0, 2.0, 3.0):
            for N in Ns:
                key = f"sps{sps}_u{u}_N{N}_s{seed}"
                if key in out:
                    continue
                chunk = max(50, int(2_000_000 // N))
                p = psd_on_grid(N, sps)
                g, c0, c1 = W.extremal_pair(p, u)
                wall_db = 10 * np.log10(g)
                _, rcal, rte = sim.streams(50_000 + 97 * sps + seed)
                thr = max(np.quantile(scores(fn, gen, n_cal, chunk), 1 - ALPHA) for gen in
                          (lambda m: sim.white(m, N, rcal), lambda m: sim.from_psd(m, N, rcal, c0), lambda m: sim.from_psd(m, N, rcal, c1)))
                pd = {}
                for d in DELTAS:
                    a = np.sqrt(10 ** ((wall_db + d) / 10))
                    r = [float(np.mean(scores(fn, lambda m: sim.modulated(m, N, rte, sps=sps) * a + ng(m), n_te, chunk) > thr))
                         for ng in (lambda m: sim.white(m, N, rte), lambda m: sim.from_psd(m, N, rte, c1))]
                    pd[str(d)] = {"white": r[0], "c1": r[1], "worst": min(r)}
                out[key] = {"sps": sps, "rho": float(sim.occupancy(sps, 0.35)), "u_db": u, "N": N, "seed": seed,
                            "wall_db": float(wall_db), "wall_flat_db": float(10 * np.log10(sim.shape_wall(sim.occupancy(sps, 0.35), u))),
                            "ed_wall_db": float(10 * np.log10(sim.ed_power_wall(u))), "Pd": pd}
                with open(path + ".tmp", "w") as fh:
                    json.dump(out, fh, indent=1)
                os.replace(path + ".tmp", path)
                print(f"sps={sps:2d} u={u} N={N:5d} wall={wall_db:6.2f} dB  worst Pd: " +
                      " ".join(f"{d:+.1f}:{pd[str(d)]['worst']:.3f}" for d in DELTAS) + f"  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
