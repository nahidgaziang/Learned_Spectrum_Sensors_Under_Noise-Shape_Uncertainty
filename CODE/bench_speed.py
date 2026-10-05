"""Speed benchmark for planning (does not change any result): GAP CNN training steps at N = 2048, batch 128, under
(a) the current pipeline (dataset on CPU, per-batch gather + copy), (b) dataset resident on the GPU, (c) + cudnn.benchmark,
(d) + automatic mixed precision. Also: CPU data generation speed and FFT multithreading. Prints steps/s and speed-ups."""
import os, sys, time
import numpy as np
import torch, torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim, detectors as D

N, B, STEPS, n = 2048, 128, 60, 6000
rng = np.random.default_rng(0)
X = D.represent("GapCNN", np.concatenate([sim.modulated(n // 2, N, rng), sim.white(n // 2, N, rng)]), True)
y = np.r_[np.ones(n // 2), np.zeros(n // 2)].astype(np.float32)
Xc, yc = torch.from_numpy(X), torch.from_numpy(y)
lf = nn.BCEWithLogitsLoss()


def bench(gpu_resident, cudnn_bench, amp):
    torch.backends.cudnn.benchmark = cudnn_bench
    torch.manual_seed(0)
    m = D.ARCHS["GapCNN"]().cuda()
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    Xs, ys = (Xc.cuda(), yc.cuda()) if gpu_resident else (Xc, yc)
    def step():
        idx = torch.randint(0, n, (B,), device="cuda" if gpu_resident else "cpu")
        xb, yb = Xs[idx], ys[idx]
        if not gpu_resident:
            xb, yb = xb.cuda(non_blocking=True), yb.cuda(non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            loss = lf(m(xb), yb)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    for _ in range(10):
        step()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(STEPS):
        step()
    torch.cuda.synchronize()
    return STEPS / (time.time() - t)


res = {}
for name, args in [("a current", (False, False, False)), ("b gpu-resident", (True, False, False)),
                   ("c + cudnn.benchmark", (True, True, False)), ("d + AMP fp16", (True, True, True))]:
    res[name] = bench(*args)
    print(f"{name:22s} {res[name]:7.1f} steps/s   x{res[name] / res['a current']:.2f}")

# CPU side: data generation and FFT threads
t = time.time(); sim.modulated(4000, N, np.random.default_rng(1)); tm = time.time() - t
t = time.time(); sim.from_psd(4000, N, np.random.default_rng(1), lambda f: 1 + 0 * f); tp = time.time() - t
import scipy.fft as sf
a = np.random.default_rng(2).standard_normal((4000, N)) + 0j
t = time.time(); np.fft.fft(a, axis=1); t1 = time.time() - t
t = time.time(); sf.fft(a, axis=1, workers=-1); t8 = time.time() - t
print(f"CPU: modulated 4000x{N}: {tm:.2f} s; from_psd: {tp:.2f} s; FFT numpy {t1:.3f} s vs scipy workers=-1 {t8:.3f} s (x{t1/t8:.1f})")
print(f"torch threads {torch.get_num_threads()}, cpu count {os.cpu_count()}")
