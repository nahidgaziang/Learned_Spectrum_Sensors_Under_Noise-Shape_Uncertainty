"""Scan the headers of every .npy file in the HF dataset (HTTP range requests, first 256 bytes only) to classify each file
as a metadata dict (pickled object: live/public RTL-SDR capture format) or a bare array (no capture metadata; likely
Simulink-simulated per the dataset card). Writes RESULTS/hf_dataset_scan.json."""
import json, os, re, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API = "https://huggingface.co/api/datasets/ishisan28/rf-signal-dataset/tree/main"
BASE = "https://huggingface.co/datasets/ishisan28/rf-signal-dataset/resolve/main"
UA = {"User-Agent": "Mozilla/5.0 (research)"}


def get(url, rng=None):
    h = dict(UA)
    if rng:
        h["Range"] = rng
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=60) as r:
        return r.read()


out = {}
for cls in ["noise", "ADS_B", "FM_broadcast", "ISM_sensors"]:
    files = json.loads(get(f"{API}/{cls}"))
    kinds = {}
    for it in files:
        name = it["path"].split("/")[-1]
        hdr = get(f"{BASE}/{cls}/{urllib.request.quote(name)}", "bytes=0-255").decode("latin-1")
        m = re.search(r"'descr':\s*'([^']+)'.*?'shape':\s*\(([^)]*)\)", hdr)
        descr, shape = (m.group(1), m.group(2)) if m else ("?", "?")
        kind = "dict(metadata)" if descr == "|O" else f"array {descr}"
        kinds.setdefault(kind, []).append((name, it.get("size"), shape))
    out[cls] = {k: {"count": len(v), "examples": v[:4], "sizes": sorted({s for _, s, _ in v})[:6]} for k, v in kinds.items()}
    print(cls, {k: v["count"] for k, v in out[cls].items()})
json.dump(out, open(os.path.join(ROOT, "RESULTS", "hf_dataset_scan.json"), "w"), indent=1)
