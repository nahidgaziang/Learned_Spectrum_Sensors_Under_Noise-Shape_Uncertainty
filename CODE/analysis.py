"""Aggregate RESULTS/runs/*.json into tables (CSV + LaTeX) with mean +- sd over seeds.

Writes:
  RESULTS/summary_runs.csv          one row per run (flattened key metrics)
  TABLES/tab_diagnostic.csv/.tex    X1/X2: detectors (WHITE recipe + classical) at N=128, sps=8
  TABLES/tab_recipes.csv/.tex       X3/X8: training recipes x architectures at N=128
  TABLES/tab_occupancy.csv          X5
  TABLES/tab_nsweep.csv             X4 (main evaluate; theory-pair wall test is in wall_test.json)
"""
import glob, json, os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHAPES_MAIN = ["white+1dB", "white-1dB", "ar+0.05", "ar-0.05", "ar+0.1", "ar-0.1", "ar+0.2", "ar-0.2", "lpf0.9", "lpf0.8",
               "rtl1dB", "rtl3dB", "pink"]


def load_runs():
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "RESULTS", "runs", "*.json"))):
        r = json.load(open(f))
        if "eval" not in r:
            continue
        c, e = r["config"], r["eval"]
        name = c["arch"] if c["kind"] == "learned" else c["detector"]
        row = {"id": r["id"], "kind": c["kind"], "model": name, "recipe": c.get("recipe", "-") if c["kind"] == "learned" else "-",
               "N": c["N"], "sps": c["sps"], "seed": c["seed"], "runtime_s": r["runtime_s"], "code": r["code_version"],
               "surgap": e["surrogate_gap"], "tag": c.get("tag", "") or "", "signal": c.get("signal", "modulated")}
        for sh in ["white"] + SHAPES_MAIN + [s for s in e["Pf"] if s.startswith("tilt")]:
            row[f"Pf_{sh}"] = e["Pf"][sh]["nominal"]
        for d, v in e["Pd"].items():
            row[f"Pd_{d}"] = v["white"]["nominal"]
            row[f"PdSur_{d}"] = e["Pd_surrogate"][d]["nominal"]
            # signal embedded in coloured noise, nominal threshold (exposes masking by coloured-negative recipes)
            for sh in ("lpf0.8", "pink", "rtl3dB", "ar+0.1"):
                if sh in v:
                    row[f"PdIn_{sh}_{d}"] = v[sh]["nominal"]
        for U in ["U_power", "U_mild", "U_receiver", "U_broad", "U_tilt1.0", "U_tilt2.0"]:
            if U in e["Pd_worstcase"]:
                for d, v in e["Pd_worstcase"][U].items():
                    row[f"wcPd_{U}_{d}"] = v
            # realised max Pf of the set on the test stream at the robust threshold
            mem = (sim_sets().get(U) or [])
            if mem:
                row[f"maxPf_{U}"] = max(e["Pf"][m][U] for m in mem if m in e["Pf"])
        rows.append(row)
    return pd.DataFrame(rows)


def sim_sets():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import sim, evaluate as E
    s = dict(sim.UNCERTAINTY_SETS); s.update(E.tilt_sets())
    return s


def msd(x, pct=False, nd=2):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return "–"
    m, s = x.mean(), (x.std(ddof=1) if x.size > 1 else 0.0)
    if pct:
        return f"{100*m:.1f}±{100*s:.1f}" if x.size > 1 else f"{100*m:.1f}"
    return f"{m:.{nd}f}±{s:.{nd}f}" if x.size > 1 else f"{m:.{nd}f}"


def agg(df, by, cols, pct_cols=()):
    out = []
    for key, g in df.groupby(by, sort=False):
        row = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        row["n_seeds"] = len(g)
        for c in cols:
            if c in g:
                row[c] = msd(g[c].values, pct=c in pct_cols)
        out.append(row)
    return pd.DataFrame(out)


def to_latex(df, path, caption=""):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(df.to_latex(index=False, escape=True))


def main():
    df = load_runs()
    df = df[df.tag == ""]          # main tables: untagged runs only (ablations are reported separately)
    os.makedirs(os.path.join(ROOT, "TABLES"), exist_ok=True)
    df.to_csv(os.path.join(ROOT, "RESULTS", "summary_runs.csv"), index=False)
    base = df[(df.N == 128) & (df.sps == 8)]
    # ---- X1/X2 diagnostic table
    diag = base[(base.recipe.isin(["WHITE", "-"]))]
    cols = ["surgap", "Pf_white", "Pf_white+1dB", "Pf_ar+0.1", "Pf_ar-0.1", "Pf_lpf0.9", "Pf_lpf0.8", "Pf_rtl3dB", "Pf_pink",
            "Pd_-10", "PdSur_-10", "wcPd_U_receiver_-10", "maxPf_U_receiver"]
    pct = [c for c in cols if c.startswith(("Pf", "Pd", "wcPd", "maxPf"))]
    t1 = agg(diag, ["kind", "model"], cols, pct)
    t1.to_csv(os.path.join(ROOT, "TABLES", "tab_diagnostic.csv"), index=False)
    to_latex(t1, os.path.join(ROOT, "TABLES", "tab_diagnostic.tex"))
    # ---- X3 recipes
    rec = base[base.kind == "learned"]
    cols2 = ["surgap", "Pf_white", "Pf_pink", "Pf_lpf0.8", "Pf_rtl3dB", "Pf_ar+0.2" if "Pf_ar+0.2" in rec else "Pf_ar+0.1",
             "Pd_-12", "Pd_-10", "wcPd_U_broad_-10", "wcPd_U_broad_-6", "maxPf_U_broad"]
    t2 = agg(rec[rec.model.isin(["CLDNN", "GapCNN"])], ["model", "recipe"], cols2, [c for c in cols2 if c != "surgap"])
    t2.to_csv(os.path.join(ROOT, "TABLES", "tab_recipes.csv"), index=False)
    to_latex(t2, os.path.join(ROOT, "TABLES", "tab_recipes.tex"))
    # ---- X5 occupancy
    occ = df[(df.N == 128) & (df.recipe.isin(["WHITE", "-"])) & (df.model.isin(["GapCNN", "ED", "BAND", "MME"]))]
    t3 = agg(occ, ["sps", "model"], ["surgap", "Pd_-14", "Pd_-12", "Pd_-10", "Pd_-8", "Pd_-6", "Pf_lpf0.8", "Pf_rtl3dB"],
             ["Pd_-14", "Pd_-12", "Pd_-10", "Pd_-8", "Pd_-6", "Pf_lpf0.8", "Pf_rtl3dB"])
    t3.to_csv(os.path.join(ROOT, "TABLES", "tab_occupancy.csv"), index=False)
    # ---- X4 N sweep (main evaluate)
    ns = df[(df.sps == 8) & (df.model.isin(["GapCNN", "BAND", "CYC", "MME", "ED"])) & (df.recipe.isin(["WHITE", "SURR", "-"]))]
    t4 = agg(ns, ["model", "recipe", "N"], ["surgap", "Pf_lpf0.8", "Pf_rtl3dB", "Pf_pink", "Pd_-14", "Pd_-10",
                                             "wcPd_U_receiver_-10", "wcPd_U_broad_-10", "wcPd_U_tilt1.0_-10", "wcPd_U_tilt1.0_-14"],
             ["Pf_lpf0.8", "Pf_rtl3dB", "Pf_pink", "Pd_-14", "Pd_-10", "wcPd_U_receiver_-10", "wcPd_U_broad_-10",
              "wcPd_U_tilt1.0_-10", "wcPd_U_tilt1.0_-14"])
    t4.to_csv(os.path.join(ROOT, "TABLES", "tab_nsweep.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(t1.to_string(index=False)); print(); print(t2.to_string(index=False)); print(); print(t4.to_string(index=False))


if __name__ == "__main__":
    main()
