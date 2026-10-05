"""Adds the transformer encoder (added 2026-10-05, U8) to RESULTS/complexity_profiling.json with exactly the
procedure of profile_complexity.py (FLOPs by torch FlopCounterMode, batch-1 latency: CPU 20 warm-up + 100 timed,
GPU 30 warm-up + 150 timed, CUDA events). The other entries are not re-measured, so the published latencies stay
unchanged.   usage: py -3.12 profile_transformer.py"""
import json, os
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from detectors import ARCHS, REPS, n_params
from profile_complexity import benchmark_latency, benchmark_latency_gpu

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, "RESULTS", "complexity_profiling.json")

np.random.seed(42); torch.manual_seed(42)
model = ARCHS["Transformer"](); model.eval()
x = (np.random.randn(1, 128) + 1j * np.random.randn(1, 128)).astype(np.complex128)
inp = torch.from_numpy(REPS[ARCHS["Transformer"].rep](x))
with FlopCounterMode(display=False) as f:
    _ = model(inp)
flops = f.get_total_flops()


def fwd():
    with torch.no_grad():
        _ = model(inp)


cpu_m, cpu_s = benchmark_latency(fwd, warmup=20, iters=100)
gpu_m, gpu_s = 0.0, 0.0
if torch.cuda.is_available():
    mg, ig = model.to("cuda"), inp.to("cuda")
    with torch.no_grad():
        gpu_m, gpu_s = benchmark_latency_gpu(mg, ig, warmup=30, iters=150)
R = json.load(open(P))
R["learned"]["Transformer"] = {"arch": "Transformer", "window_len": 128, "params": n_params(model), "flops": flops,
                               "mflops": round(flops / 1e6, 2), "cpu_latency_ms": round(cpu_m, 3), "cpu_latency_std_ms": round(cpu_s, 3),
                               "gpu_latency_ms": round(gpu_m, 3), "gpu_latency_std_ms": round(gpu_s, 3), "rep": "iq",
                               "note": "added 2026-10-05 by profile_transformer.py (same procedure as profile_complexity.py)"}
json.dump(R, open(P, "w"), indent=2)
print(R["learned"]["Transformer"])
