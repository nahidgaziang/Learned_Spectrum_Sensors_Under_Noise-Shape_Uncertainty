"""Download real RTL-SDR captures for the real-signal surrogate test and the second-receiver noise test (plan item F).
  TrevTron/rtl-ml-dataset v2 (MIT licence; RTL-SDR Blog V4, R828D tuner): all 100 noise captures + 40 evenly spaced
     captures from each signal class (FM_broadcast, NOAA_weather, APRS, pager, ISM_sensors, FRS_GMRS)
     -> DATA/trevtron/<class>/
  ishisan28/rf-signal-dataset (no licence stated -> evaluation only; Nooelec R820T2 + public RTL captures): 50 evenly
     spaced metadata-carrying captures per signal class (ADS_B, FM_broadcast, ISM_sensors); bare-array files (probably
     simulated) are skipped -> DATA/hf_signals/<class>/
Resumable (existing files are skipped), atomic writes. Writes DATA/real_signals_manifest.json."""
import json, os, re, urllib.parse, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UA = {"User-Agent": "Mozilla/5.0 (research)"}


def get(url, rng=None, retries=5):
    import time
    h = dict(UA)
    if rng:
        h["Range"] = rng
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=180) as r:
                return r.read()
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def tree(repo, path):
    return json.loads(get(f"https://huggingface.co/api/datasets/{repo}/tree/main/{urllib.parse.quote(path)}"))


def spaced(names, k):
    names = sorted(names)
    if len(names) <= k:
        return names
    idx = [round(i * (len(names) - 1) / (k - 1)) for i in range(k)]
    return [names[i] for i in idx]


manifest = {}
jobs = []
for cls in ["noise", "FM_broadcast", "NOAA_weather", "APRS", "pager", "ISM_sensors", "FRS_GMRS"]:
    files = [x["path"] for x in tree("TrevTron/rtl-ml-dataset", f"datasets_validated/{cls}") if x["path"].endswith(".npy")]
    pick = files if cls == "noise" else spaced(files, 40)
    jobs += [("TrevTron/rtl-ml-dataset", p, os.path.join(ROOT, "DATA", "trevtron", cls)) for p in pick]
for cls in ["ADS_B", "FM_broadcast", "ISM_sensors"]:
    files = [x["path"] for x in tree("ishisan28/rf-signal-dataset", cls) if x["path"].endswith(".npy")]
    dicts = []
    for p in sorted(files):
        hdr = get(f"https://huggingface.co/datasets/ishisan28/rf-signal-dataset/resolve/main/{urllib.parse.quote(p)}", "bytes=0-255")
        if "'descr': '|O'" in hdr.decode("latin-1"):
            dicts.append(p)
        if len(dicts) >= 150:
            break
    jobs += [("ishisan28/rf-signal-dataset", p, os.path.join(ROOT, "DATA", "hf_signals", cls)) for p in spaced(dicts, 50)]
print(f"{len(jobs)} files to fetch", flush=True)
for k, (repo, p, d) in enumerate(jobs):
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, os.path.basename(p))
    if not os.path.exists(dst):
        data = get(f"https://huggingface.co/datasets/{repo}/resolve/main/{urllib.parse.quote(p)}")
        open(dst + ".tmp", "wb").write(data)
        os.replace(dst + ".tmp", dst)
    manifest[os.path.relpath(dst, ROOT)] = {"repo": repo, "path": p}
    if k % 25 == 0:
        print(f"{k + 1}/{len(jobs)}", flush=True)
json.dump(manifest, open(os.path.join(ROOT, "DATA", "real_signals_manifest.json"), "w"), indent=1)
print("done", len(manifest), flush=True)
