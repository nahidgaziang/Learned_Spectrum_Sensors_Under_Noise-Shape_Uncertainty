"""Compute accounting from the run records (RESULTS/runs/*.json: runtime_s, peak_vram_MB, device) plus the
logged durations of the wall tests. Writes RESULTS/compute_summary.json."""
import glob, json, os, re
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = [json.load(open(f)) for f in glob.glob(os.path.join(ROOT, "RESULTS", "runs", "*.json"))]
# the 'device' field records the GPU whenever one is present; classical detectors run on the CPU
gpu = [r for r in R if r["config"]["kind"] in ("learned", "whitening")]
cpu = [r for r in R if r["config"]["kind"] == "classical"]
out = {"n_runs": len(R), "n_gpu_runs": len(gpu), "n_cpu_runs": len(cpu),
       "gpu_run_hours": sum(r.get("runtime_s", 0) for r in gpu) / 3600,
       "cpu_run_hours": sum(r.get("runtime_s", 0) for r in cpu) / 3600,
       "peak_vram_MB_max": max((r.get("peak_vram_MB") or 0) for r in R),
       "devices": sorted({str(r.get("device")) for r in R})}
# wall tests: last timestamp "(NNNs)" printed per seed run in the logs
for log in ["wall_test_final.out", "wall_test_matched.out", "wall_test_long.out", "wall_vs_theory.out", "wall_test_long_nets.out"]:
    p = os.path.join(ROOT, "EXPERIMENTS", log)
    if os.path.exists(p):
        raw = open(p, "rb").read()
        # logs appended by Windows PowerShell 5.1 can be UTF-16 (BOM FF FE); others UTF-8
        t = raw.decode("utf-16", errors="ignore") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8", errors="ignore")
        secs = [int(s) for s in re.findall(r"\((\d+)s\)", t)]
        # each seed restarts its clock: sum the maxima of increasing runs
        tot, prev = 0, 0
        for s in secs:
            if s < prev:
                tot += prev
            prev = s
        tot += prev
        out[f"{log}_hours"] = tot / 3600
json.dump(out, open(os.path.join(ROOT, "RESULTS", "compute_summary.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
