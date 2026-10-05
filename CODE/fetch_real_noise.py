"""Download the noise class of the public Hugging Face dataset `ishisan28/rf-signal-dataset`
(RTL-SDR / Nooelec NESDR SMArt v5 recordings; README lists no license -> used for evaluation only,
cited by URL). Files go to PROJECT/DATA/real_noise_hf/ (git-ignored)."""
import json, os, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "DATA", "real_noise_hf")
os.makedirs(OUT, exist_ok=True)
API = "https://huggingface.co/api/datasets/ishisan28/rf-signal-dataset/tree/main/noise"
BASE = "https://huggingface.co/datasets/ishisan28/rf-signal-dataset/resolve/main/"
UA = {"User-Agent": "Mozilla/5.0"}

files = json.loads(urllib.request.urlopen(urllib.request.Request(API, headers=UA), timeout=60).read())
json.dump(files, open(os.path.join(OUT, "_listing.json"), "w"), indent=1)
for i, f in enumerate(files):
    dest = os.path.join(OUT, os.path.basename(f["path"]))
    if os.path.exists(dest) and os.path.getsize(dest) == f["size"]:
        continue
    for k in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(BASE + f["path"], headers=UA), timeout=120) as r, open(dest + ".part", "wb") as fh:
                fh.write(r.read())
            os.replace(dest + ".part", dest)
            break
        except Exception as e:
            print("retry", f["path"], e, flush=True)
            time.sleep(5 * (k + 1))
    if i % 20 == 0:
        print(i, "/", len(files), flush=True)
print("done", len(files))
