"""Wall test at long observation lengths for the two classical archetypes (CPU only):
BAND (oracle band-power ratio: a second-order, scale-invariant test) and CYC (normalised cyclic autocorrelation at
the symbol rate: uses cyclostationary structure that stationary noise of any shape lacks).

Same protocol as wall_test.py (u = 1 dB extremal pair, robust threshold over {white, c0, c1} on calibration draws,
worst-case Pd over H1 noise in {white, c1} at SNR = wall + delta), extended to N = 8192 and 32768. Theory: BAND stays
at the false-alarm level at and below the wall for every N; CYC is not bound by the shape wall, so its worst-case Pd
at the wall must grow with N.

For long N the signal PSD is NOT re-estimated per bin (a noisy estimate would make the extremal pair inexact, which a
long second-order test could exploit). Instead the smooth PSD estimated at N0 = 2048 from 8000 windows is
interpolated onto the N-point grid. Both detectors are scored on the same draws.
Output: RESULTS/wall_test_long_s{seed}.json      Usage: py -3.12 wall_test_long.py <seed>
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim, detectors as D, wall as W

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALPHA = 0.05
P0 = None


def psd_on_grid(N, N0=2048):
    global P0
    if P0 is None:
        P0 = W.signal_psd(N0, n=8000)
    f0 = np.fft.fftshift(np.fft.fftfreq(N0))
    p0 = np.fft.fftshift(P0)
    f = np.fft.fftfreq(N)
    p = np.interp(f, np.concatenate([f0 - 1, f0, f0 + 1]), np.tile(p0, 3))   # periodic interpolation
    return p / p.mean()


def scores(fns, gen, n, chunk):
    out = {k: [] for k in fns}
    for i in range(0, n, chunk):
        x = gen(min(chunk, n - i))
        for k, f in fns.items():
            out[k].append(f(x))
    return {k: np.concatenate(v) for k, v in out.items()}


def main(seed=0, Ns=(2048, 8192, 32768), R_db=1.0, deltas=(-2, 0, 2, 4), n_cal=3000, n_te=2000):
    path = os.path.join(ROOT, "RESULTS", f"wall_test_long_s{seed}.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    hw = sim.occupancy(8, 0.35) / 2
    fns = {"BAND": lambda x: D.band_ratio(x, half_width=hw), "CYC": lambda x: D.cyc(x, sps=8)}
    t0 = time.time()
    for N in Ns:
        key = f"N{N}_R{R_db}"
        if key in out:
            continue
        chunk = max(50, int(2_000_000 // N))
        p = psd_on_grid(N)
        g, c0, c1 = W.extremal_pair(p, R_db)
        wall_db = 10 * np.log10(g)
        _, rcal, rte = sim.streams(40_000 + seed)
        cal = [scores(fns, gen, n_cal, chunk) for gen in
               (lambda m: sim.white(m, N, rcal), lambda m: sim.from_psd(m, N, rcal, c0), lambda m: sim.from_psd(m, N, rcal, c1))]
        thr = {k: max(np.quantile(c[k], 1 - ALPHA) for c in cal) for k in fns}
        pf = {}
        for name, gen in {"white": lambda m: sim.white(m, N, rte), "c0": lambda m: sim.from_psd(m, N, rte, c0),
                          "c1": lambda m: sim.from_psd(m, N, rte, c1)}.items():
            s = scores(fns, gen, n_te, chunk)
            pf[name] = {k: float(np.mean(s[k] > thr[k])) for k in fns}
        pd = {}
        for d in deltas:
            a = np.sqrt(10 ** ((wall_db + d) / 10))
            r = {}
            for hn, ng in {"white": lambda m: sim.white(m, N, rte), "c1": lambda m: sim.from_psd(m, N, rte, c1)}.items():
                s = scores(fns, lambda m: sim.modulated(m, N, rte) * a + ng(m), n_te, chunk)
                r[hn] = {k: float(np.mean(s[k] > thr[k])) for k in fns}
            pd[str(d)] = {k: {"white": r["white"][k], "c1": r["c1"][k], "worst": min(r["white"][k], r["c1"][k])} for k in fns}
        out[key] = {"N": N, "wall_db": wall_db, "thr": thr, "Pf": pf, "Pd": pd, "n_cal": n_cal, "n_te": n_te}
        with open(path + ".tmp", "w") as fh:
            json.dump(out, fh, indent=1)
        os.replace(path + ".tmp", path)
        print(f"N={N} wall={wall_db:.2f} dB  " + "  ".join(
            f"{k}: Pfmax={max(pf[s][k] for s in pf):.3f} worst " + " ".join(f"{d}:{pd[str(d)][k]['worst']:.3f}" for d in deltas)
            for k in fns) + f"  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
