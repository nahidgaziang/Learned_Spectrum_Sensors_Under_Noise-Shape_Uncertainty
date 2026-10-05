"""Summarise RESULTS/real_noise_N128.json per recording group and detector (mean over recordings; learned
detectors averaged over their seeds). Writes TABLES/tab_real_noise.csv and prints a compact table."""
import collections, json, os, re
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = json.load(open(os.path.join(ROOT, "RESULTS", "real_noise_v2_N128.json")))
rows = []
for name, rec in R["recordings"].items():
    meta = rec["meta"]
    if rec["group"] == "unlabelled":
        grp = "flat control"
    else:
        grp = f"{float(meta.get('center_freq', 0))/1e6:.1f} MHz" + (" (narrowband)" if rec["shape"]["narrowband"] else "")
    dets = collections.defaultdict(list)
    for field in ["Pf", "Pf_w16", "Pf_wbig", "Pd_m10_nominal", "Pd_m10_wbig"]:
        for k, v in rec[field].items():
            det = re.sub(r"-s\d+$", "", k)
            dets[(det, field)].append(v)
    for (det, field), vs in dets.items():
        rows.append({"group": grp, "recording": name, "detector": det, "metric": field, "value": float(np.mean(vs)),
                     "rolloff_db": rec["shape"]["edge_rolloff_db"]})
df = pd.DataFrame(rows)
piv = df.groupby(["group", "detector", "metric"]).value.mean().unstack("metric")
cnt = df.groupby("group").recording.nunique()
os.makedirs(os.path.join(ROOT, "TABLES"), exist_ok=True)
piv.to_csv(os.path.join(ROOT, "TABLES", "tab_real_noise_v2.csv"))
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 400)
print(cnt.to_string())
order = ["CLDNN-WHITE", "ResNet1D-WHITE", "CovCNN-WHITE", "STFTCNN-WHITE", "GapCNN-WHITE", "CLDNN-PINK", "GapCNN-RAND", "GapCNN-SURR", "BAND", "MME", "SFM", "CYC"]
for metric in ["Pf", "Pf_wbig", "Pd_m10_nominal"]:
    t = piv[metric].unstack("detector")[[d for d in order if d in piv[metric].unstack("detector").columns]]
    print(f"\n=== {metric} (mean over recordings in group) ===")
    print((100 * t).round(1).to_string())
# clean live = all live groups except narrowband
live = df[(~df.group.str.contains("control")) & (~df.group.str.contains("narrowband"))]
print("\n=== clean live recordings: mean over recordings ===")
print((100 * live.groupby(["detector", "metric"]).value.mean().unstack("metric")).round(1).reindex(order).to_string())
print("n clean live recordings:", live.recording.nunique())

