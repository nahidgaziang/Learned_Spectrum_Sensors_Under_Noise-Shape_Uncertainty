"""Diagnostics of the real RTL-SDR noise recordings: PSD shape per group, narrowband components.
Writes FIGURES/diag_real_noise_psd.png and RESULTS/real_noise_psd_stats.json."""
import collections, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import real_noise_eval as R

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
files = sorted(glob.glob(os.path.join(ROOT, "DATA", "real_noise_hf", "*.npy")))
groups = collections.defaultdict(list)
example = {}
for f in files:
    iq, meta = R.load(f)
    sh, P = R.psd_shape(iq)
    key = "unlabelled" if not meta.get("sample_rate") else "%.1f MHz" % (float(meta.get("center_freq", 0)) / 1e6)
    sh["file"] = os.path.basename(f)
    groups[key].append(sh)
    example.setdefault(key, P)
fig, ax = plt.subplots(figsize=(6, 3))
fr = np.fft.fftshift(np.fft.fftfreq(128))
for k, P in example.items():
    ax.plot(fr, 10 * np.log10(P), label=k, lw=1)
ax.legend(fontsize=7)
ax.set_xlabel("f / fs")
ax.set_ylabel("PSD (dB, median-normalised)")
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "FIGURES", "diag_real_noise_psd.png"), dpi=130)
summary = {}
for k, v in sorted(groups.items()):
    ro = [s["edge_rolloff_db"] for s in v]
    nb = sum(s["narrowband"] for s in v)
    mx = max(s["max_narrowband_excess_db"] for s in v)
    summary[k] = {"n": len(v), "rolloff_min": min(ro), "rolloff_max": max(ro), "rolloff_mean": float(np.mean(ro)),
                  "narrowband_flagged": nb, "max_excess_db": mx}
    print(f"{k:12s} n={len(v):3d} narrowband-flagged={nb:3d} max_excess={mx:5.1f} dB rolloff=[{min(ro):.2f},{max(ro):.2f}]")
json.dump({"groups": summary, "files": {k: v for k, v in groups.items()}}, open(os.path.join(ROOT, "RESULTS", "real_noise_psd_stats.json"), "w"), indent=1)
print("files", len(files))
