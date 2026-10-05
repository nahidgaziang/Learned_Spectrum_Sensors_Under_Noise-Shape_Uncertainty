"""Preview of the wall signature from the main evaluation: worst-case Pd under the two-level band tilt
set U_tilt1.0 = {white, tilt+1 dB, tilt-1 dB} (robust threshold), per detector and N (mean over seeds)."""
import numpy as np
import pandas as pd
from analysis import load_runs

df = load_runs()
df = df[(df.tag == "") & (df.sps == 8)]
rows = []
for (model, recipe, N), g in df.groupby(["model", "recipe", "N"]):
    if model not in ("GapCNN", "BAND", "CYC", "MME", "ED") or recipe not in ("WHITE", "SURR", "-"):
        continue
    r = {"det": f"{model}-{recipe}" if recipe != "-" else model, "N": N, "seeds": len(g)}
    for d in [-20, -17, -14, -12, -10, -8, -6]:
        c = f"wcPd_U_tilt1.0_{d}"
        if c in g:
            r[str(d)] = round(100 * g[c].mean(), 1)
    rows.append(r)
pd.set_option("display.width", 200)
print(pd.DataFrame(rows).sort_values(["det", "N"]).to_string(index=False))
