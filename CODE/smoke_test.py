"""Quick sanity checks for sim/detectors/evaluate (run before any experiment queue)."""
import time
import numpy as np
import sim, detectors as D

rng = np.random.default_rng(0)
N = 128
# 1) powers
for name, gen in sim.TEST_SHAPES.items():
    x = gen(4000, N, rng)
    print(f"{name:10s} power={np.mean(np.abs(x)**2):.3f}")
s = sim.modulated(2000, N, rng); u = sim.surrogate(2000, N, rng)
print("signal power", np.mean(np.abs(s) ** 2).round(3), "surrogate power", np.mean(np.abs(u) ** 2).round(3))
# 2) PSD of signal vs surrogate (should match), band tilt levels
P_s = np.mean(np.abs(np.fft.fft(s, axis=1)) ** 2, 0); P_u = np.mean(np.abs(np.fft.fft(u, axis=1)) ** 2, 0)
print("PSD corr signal vs surrogate", np.corrcoef(P_s, P_u)[0, 1].round(4), " in-band frac", (np.sum(P_s[np.abs(np.fft.fftfreq(N)) <= 0.0844]) / P_s.sum()).round(3))
t = sim.band_tilt(4000, N, rng, 1.0, +1); P_t = np.mean(np.abs(np.fft.fft(t, axis=1)) ** 2, 0)
f = np.fft.fftfreq(N)
print("tilt1dB in/out ratio dB", (10 * np.log10(P_t[np.abs(f) <= 0.08].mean() / P_t[np.abs(f) > 0.1].mean())).round(3))
r = sim.rtl_like(4000, N, rng, 3.0); P_r = np.mean(np.abs(np.fft.fft(r, axis=1)) ** 2, 0)
print("rtl3dB centre/edge dB", (10 * np.log10(P_r[np.abs(f) < 0.05].mean() / P_r[np.abs(f) > 0.45].mean())).round(3))
# 3) classical detectors: Pf at white-calibrated threshold under white (should be ~0.05) and timing
for nm in ["ED", "MME", "SFM", "CYC", "BAND"]:
    fn = {"CYC": lambda x: D.cyc(x, 8), "BAND": lambda x: D.band_ratio(x, 0.0844)}.get(nm, D.CLASSICAL.get(nm))
    t0 = time.time()
    thr = np.quantile(fn(sim.white(8000, N, rng)), 0.95)
    pf_w = np.mean(fn(sim.white(4000, N, rng)) > thr)
    pf_l = np.mean(fn(sim.butter_rolloff(4000, N, rng, 0.8)) > thr)
    pd10 = np.mean(fn(sim.h1(sim.modulated(4000, N, rng), -10, rng)) > thr)
    pds10 = np.mean(fn(sim.h1(sim.surrogate(4000, N, rng), -10, rng)) > thr)
    print(f"{nm:5s} Pf_white={pf_w:.3f} Pf_lpf0.8={pf_l:.3f} Pd(-10)={pd10:.3f} Pd_sur(-10)={pds10:.3f} t={time.time()-t0:.2f}s")
# 4) representations
for arch in D.ARCHS:
    z = D.represent(arch, s[:8]); m = D.ARCHS[arch]()
    import torch
    print(arch, z.shape, m(torch.from_numpy(z)).shape, "params", D.n_params(m))
