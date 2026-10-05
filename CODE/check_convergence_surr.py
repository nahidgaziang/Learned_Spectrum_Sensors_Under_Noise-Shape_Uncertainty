"""Validation-loss trajectories of the surrogate recipes across N (are the N=2048 runs converged?)."""
import glob, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for pat in ["GapCNN_SURRX_N*_sps8_s*.json", "GapCNN_SURR_N*_sps8_s*.json", "GapCNN_WHITE_N*_sps8_s0.json"]:
    for f in sorted(glob.glob(os.path.join(ROOT, "RESULTS", "runs", pat))):
        r = json.load(open(f))
        h = r["training"]["history"]
        e = r["eval"]
        print(f"{r['id'][:30]:30s} epochs={len(h):2d} val: " + " ".join(f"{x['val_loss']:.3f}" for x in h)
              + f" | train {h[-1]['train_loss']:.3f} | gap {100*(e['surrogate_gap'] or 0):+.1f} | PdSur(0dB) {e['Pd_surrogate']['0']['nominal']:.2f}")
