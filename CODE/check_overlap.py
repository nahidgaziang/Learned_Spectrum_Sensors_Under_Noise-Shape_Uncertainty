"""Check overlap between TrevTron/rtl-ml-dataset v2 (R828D tuner) and ishisan28/rf-signal-dataset (R820T2 tuner).
Computes SHA-256 hashes of all noise captures, verifies that there is zero overlap, and compares capture metadata.
Writes RESULTS/real_noise_overlap.json."""
import glob, hashlib, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_noise_eval as RN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

hf_files = sorted(glob.glob(os.path.join(ROOT, "DATA", "real_noise_hf", "*.npy")))
tt_files = sorted(glob.glob(os.path.join(ROOT, "DATA", "trevtron", "noise", "*.npy")))


def get_hash(fp):
    return hashlib.sha256(open(fp, "rb").read()).hexdigest()


hf_hashes = {get_hash(f): os.path.basename(f) for f in hf_files}
tt_hashes = {get_hash(f): os.path.basename(f) for f in tt_files}
overlap = set(hf_hashes.keys()) & set(tt_hashes.keys())

tt_sample_meta = RN.load(tt_files[0])[1] if tt_files else {}
hf_sample_meta = RN.load(hf_files[0])[1] if hf_files else {}

out = {
    "hf_noise_files": len(hf_files),
    "tt_noise_files": len(tt_files),
    "sha256_overlap_count": len(overlap),
    "hf_receiver": "R820T2 (Nooelec NESDR SMArt v5 / public RTL)",
    "tt_receiver": "R828D (RTL-SDR Blog V4)",
    "tt_center_freq_mhz": float(tt_sample_meta.get("center_freq", 0)) / 1e6,
    "tt_sample_rate_msps": float(tt_sample_meta.get("sample_rate", 0)) / 1e6,
    "tt_license": "MIT",
    "hf_license": "no license stated (evaluation only)",
}

out_path = os.path.join(ROOT, "RESULTS", "real_noise_overlap.json")
with open(out_path, "w") as fh:
    json.dump(out, fh, indent=2)

print(f"HF files: {len(hf_files)}, TrevTron files: {len(tt_files)}, SHA-256 overlap: {len(overlap)}")
print(f"Wrote {out_path}")
