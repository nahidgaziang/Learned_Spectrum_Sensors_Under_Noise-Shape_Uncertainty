"""Supplementary tables: mean +- s.d. over seeds for every cell of the main and recipe tables (N = 128, untagged runs),
plus seed counts. Writes TABLES/supp_main_sd.csv, TABLES/supp_recipes_sd.csv and TABLES/supp_sd.tex."""
import os
import numpy as np
import pandas as pd
from analysis import load_runs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLS = ["surgap", "Pf_white+1dB", "Pf_lpf0.9", "Pf_lpf0.8", "Pf_rtl3dB", "Pf_ar+0.1", "Pf_ar-0.1", "Pf_pink", "Pd_-10",
        "wcPd_U_receiver_-10", "PdIn_pink_-10", "PdIn_lpf0.8_-10", "wcPd_U_broad_-10"]


def summarise(d, keys):
    rows = []
    for k, g in d.groupby(keys):
        r = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        r["seeds"] = len(g)
        for c in COLS:
            if c in g:
                r[c] = f"{100*g[c].mean():.1f} ± {100*g[c].std(ddof=1):.1f}" if len(g) > 1 else f"{100*g[c].mean():.1f}"
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    df = load_runs()
    b = df[(df.N == 128) & (df.sps == 8)]
    main_ = pd.concat([b[(b.kind == "learned") & (b.recipe == "WHITE") & (b.tag == "")],
                       b[(b.model == "DetectNetExact") & (b.tag == "QAM16")],
                       b[(b.kind == "classical") & (b.tag == "")]])
    t1 = summarise(main_, ["kind", "model"])
    t2 = summarise(b[(b.kind == "learned") & (b.tag == "")], ["model", "recipe"])
    t1.to_csv(os.path.join(ROOT, "TABLES", "supp_main_sd.csv"), index=False)
    t2.to_csv(os.path.join(ROOT, "TABLES", "supp_recipes_sd.csv"), index=False)
    def tex(t, cap, lab, cols):
        head = " & ".join(["Detector", "n"] + [c.replace("_", r"\_") for c in cols])
        rows = [" & ".join([f"{r.get('model','')} {r.get('recipe','') if r.get('recipe','') not in ('', '-') else ''}".strip(), str(r["seeds"])]
                           + [str(r.get(c, "--")).replace("±", r"$\pm$") for c in cols]) + r" \\" for _, r in t.iterrows()]
        return ("\\begin{table*}[t]\n\\centering\n\\caption{" + cap + "}\n\\label{" + lab + "}\n\\scriptsize\n\\setlength{\\tabcolsep}{2pt}\n"
                "\\begin{tabular}{l" + "r" * (len(cols) + 1) + "}\n\\toprule\n" + head + " \\\\\n\\midrule\n" + "\n".join(rows) +
                "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    c1 = ["surgap", "Pf_white+1dB", "Pf_lpf0.9", "Pf_lpf0.8", "Pf_rtl3dB", "Pf_ar+0.1", "Pf_ar-0.1", "Pf_pink", "Pd_-10", "wcPd_U_receiver_-10"]
    c2 = ["surgap", "Pf_pink", "Pf_lpf0.8", "Pf_rtl3dB", "Pd_-10", "PdIn_pink_-10", "PdIn_lpf0.8_-10", "wcPd_U_broad_-10"]
    open(os.path.join(ROOT, "TABLES", "supp_sd.tex"), "w", encoding="utf-8").write(
        tex(t1, "Main table with mean $\\pm$ s.d.\\ over seeds (\\%; gap in points). $N=128$.", "tab:supp-main", c1) + "\n" +
        tex(t2, "Recipe table with mean $\\pm$ s.d.\\ over seeds (\\%; gap in points). $N=128$.", "tab:supp-recipes", c2))
    print(t1.to_string(index=False)); print(t2.to_string(index=False))


if __name__ == "__main__":
    main()
