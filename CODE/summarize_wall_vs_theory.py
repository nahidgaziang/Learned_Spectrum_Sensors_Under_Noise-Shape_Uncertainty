"""Summarise RESULTS/wall_vs_theory.json (J2): per (sps, u, N) the largest worst-case Pd at delta <= 0 (should be
<= alpha) and the SNR offset delta_50 where worst-case Pd crosses 0.5 (linear interpolation; should approach 0 from
above as N grows). Writes TABLES/wall_vs_theory.csv and RESULTS/wall_vs_theory_summary.json."""
import json, os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = json.load(open(os.path.join(ROOT, "RESULTS", "wall_vs_theory.json")))
rows = []
for k, v in D.items():
    ds = sorted(float(d) for d in v["Pd"])
    pw = np.array([v["Pd"][str(d) if str(d) in v["Pd"] else f"{d}"]["worst"] for d in ds])
    below = max(v["Pd"][d]["worst"] for d in v["Pd"] if float(d) <= 0)
    d50 = float("nan")
    for i in range(1, len(ds)):
        if pw[i - 1] < 0.5 <= pw[i]:
            d50 = ds[i - 1] + (0.5 - pw[i - 1]) * (ds[i] - ds[i - 1]) / (pw[i] - pw[i - 1])
            break
    rows.append({"sps": v["sps"], "rho": round(v["rho"], 3), "u_db": v["u_db"], "N": v["N"], "wall_db": round(v["wall_db"], 2),
                 "wall_flat_db": round(v["wall_flat_db"], 2), "ed_wall_db": round(v["ed_wall_db"], 2),
                 "max_Pd_at_or_below_wall": round(below, 3), "delta50_db": round(d50, 2)})
df = pd.DataFrame(rows).sort_values(["sps", "u_db", "N"])
df.to_csv(os.path.join(ROOT, "TABLES", "wall_vs_theory.csv"), index=False)
summ = {"n_configs": len(df), "max_Pd_at_or_below_wall": float(df.max_Pd_at_or_below_wall.max()),
        "delta50_max_N32768": float(df[df.N == 32768].delta50_db.max()), "delta50_max_N8192": float(df[df.N == 8192].delta50_db.max()),
        "delta50_shrinks_with_N_all": bool(all(
            df[(df.sps == s) & (df.u_db == u) & (df.N == 32768)].delta50_db.iloc[0] <= df[(df.sps == s) & (df.u_db == u) & (df.N == 8192)].delta50_db.iloc[0]
            for s in df.sps.unique() for u in df.u_db.unique()))}
json.dump(summ, open(os.path.join(ROOT, "RESULTS", "wall_vs_theory_summary.json"), "w"), indent=1)
print(df.to_string(index=False))
print(summ)
