"""Print validation-loss trajectories (every other epoch) to check that training converged."""
import glob, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
files = sorted(glob.glob(os.path.join(ROOT, "RESULTS", "runs", "*_N128_sps8_s0.json")))
for f in files:
    r = json.load(open(f))
    if "training" not in r or "history" not in r["training"]:
        continue
    h = r["training"]["history"]
    vals = " ".join(f"{e['val_loss']:.3f}" for e in h[::2])
    print(f"{r['id'][:34]:34s} val: {vals} | last train {h[-1]['train_loss']:.3f} | params {r['training']['n_params']}")
