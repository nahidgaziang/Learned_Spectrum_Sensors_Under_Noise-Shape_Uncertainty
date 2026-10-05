"""Profile computational complexity, parameters, FLOPs, and inference latency
of all learned and classical spectrum sensing detectors for IEEE TCCN.
"""
import time
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from detectors import (
    ARCHS, REPS, CLASSICAL, n_params, ed, mme, sfm, cyc, band_ratio
)

def benchmark_latency(fn, warmup=50, iters=200):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)  # ms
    return float(np.mean(times)), float(np.std(times))

def benchmark_latency_gpu(model, tensor_gpu, warmup=50, iters=200):
    for _ in range(warmup):
        _ = model(tensor_gpu)
    torch.cuda.synchronize()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start_event.record()
        _ = model(tensor_gpu)
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))  # ms
    return float(np.mean(times)), float(np.std(times))

def profile():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Profiling on device: {device}")
    
    results = {}
    N = 128
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 1. Learned Detectors Profiling
    print("\n--- Profiling Learned Detectors ---")
    model_list = [
        ("GAP CNN", "GapCNN", 128),
        ("GAP CNN (N=2048)", "GapCNN", 2048),
        ("GAP CNN 4X", "GapCNN4X", 128),
        ("DetectNet replica", "DetectNetExact", 128),
        ("CNN--LSTM", "CLDNN", 128),
        ("1-D ResNet", "ResNet1D", 128),
        ("Covariance CNN", "CovCNN", 128),
        ("STFT CNN", "STFTCNN", 128),
        ("CycNet", "CycNet", 128),
    ]
    
    results["learned"] = {}
    
    for display_name, arch_name, win_len in model_list:
        model_cls = ARCHS[arch_name]
        if arch_name == "DetectNetExact":
            model = model_cls(N=win_len)
        else:
            model = model_cls()
        model.eval()
        
        params = n_params(model)
        
        # Synthetic input
        x_raw = (np.random.randn(1, win_len) + 1j * np.random.randn(1, win_len)).astype(np.complex128)
        rep_fn = REPS[model_cls.rep]
        x_rep = rep_fn(x_raw)
        inp_tensor = torch.from_numpy(x_rep)
        
        # Count FLOPs
        with FlopCounterMode(display=False) as f:
            _ = model(inp_tensor)
        flops = f.get_total_flops()
        
        # CPU Latency
        def forward_cpu():
            with torch.no_grad():
                _ = model(inp_tensor)
        cpu_mean_ms, cpu_std_ms = benchmark_latency(forward_cpu, warmup=20, iters=100)
        
        # GPU Latency
        gpu_mean_ms, gpu_std_ms = 0.0, 0.0
        if torch.cuda.is_available():
            model_gpu = model.to(device)
            inp_gpu = inp_tensor.to(device)
            with torch.no_grad():
                gpu_mean_ms, gpu_std_ms = benchmark_latency_gpu(model_gpu, inp_gpu, warmup=30, iters=150)
        
        results["learned"][display_name] = {
            "arch": arch_name,
            "window_len": win_len,
            "params": params,
            "flops": flops,
            "mflops": round(flops / 1e6, 2),
            "cpu_latency_ms": round(cpu_mean_ms, 3),
            "cpu_latency_std_ms": round(cpu_std_ms, 3),
            "gpu_latency_ms": round(gpu_mean_ms, 3),
            "gpu_latency_std_ms": round(gpu_std_ms, 3),
            "rep": model_cls.rep
        }
        print(f"[{display_name}] Params: {params:,} | FLOPs: {flops/1e6:.2f} M | CPU: {cpu_mean_ms:.3f} ms | GPU: {gpu_mean_ms:.3f} ms")

    # 2. Classical Detectors Profiling
    print("\n--- Profiling Classical Detectors ---")
    results["classical"] = {}
    classical_list = [
        ("Energy Detector (ED)", ed, 128),
        ("Band-Power Ratio (BAND)", band_ratio, 128),
        ("Spectral Flatness (SFM)", sfm, 128),
        ("Eigenvalue Ratio (MME)", mme, 128),
        ("Cyclic Autocorrelation (CYC)", cyc, 128),
        ("Band-Power Ratio (N=2048)", band_ratio, 2048),
        ("Cyclic Autocorrelation (N=2048)", cyc, 2048),
    ]
    
    for display_name, fn, win_len in classical_list:
        x_raw = (np.random.randn(1, win_len) + 1j * np.random.randn(1, win_len)).astype(np.complex128)
        def run_classical():
            _ = fn(x_raw)
        cpu_mean_ms, cpu_std_ms = benchmark_latency(run_classical, warmup=20, iters=100)
        
        results["classical"][display_name] = {
            "window_len": win_len,
            "cpu_latency_ms": round(cpu_mean_ms, 3),
            "cpu_latency_std_ms": round(cpu_std_ms, 3),
        }
        print(f"[{display_name}] CPU: {cpu_mean_ms:.3f} ms")

    out_file = Path("RESULTS/complexity_profiling.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved profiling results to {out_file}")

if __name__ == "__main__":
    profile()
