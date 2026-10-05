"""Statistical Rigor: Two-level bootstrap confidence intervals (B=2000) and paired hypothesis tests.

This module computes:
1. Two-level bootstrap (Level 1: seeds with replacement; Level 2: test windows via Binomial/multinomial)
   for all cells in Table I (main diagnostic) and Table II (training recipes).
2. Paired McNemar hypothesis tests and paired-difference bootstrap CIs for matched vs unmatched recipes.
3. Surrogate gap confidence intervals for WHITE and SURRX recipes.

Outputs:
- RESULTS/stats_ci.json: Full structured results and summary metrics.
- TABLES/supp_ci.tex: LaTeX tables with mean [95% CI] for supplementary materials.
"""
import glob
import json
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "CODE"))
from analysis import load_runs

# Reproducibility seed
RNG = np.random.default_rng(20261002)
B = 2000

LEARNED_MODELS = [
    ("CLDNN", "CNN--LSTM"),
    ("DetectNetExact", "DetectNet replica"),
    ("ResNet1D", "1-D ResNet"),
    ("CovCNN", "Covariance CNN"),
    ("STFTCNN", "STFT CNN"),
    ("GapCNN", "GAP CNN (ours)"),
]

CLASSICAL_MODELS = [
    ("ED", "ED (known $\\sigma^2$)"),
    ("MME", "MME"),
    ("SFM", "Spectral flatness"),
    ("BAND", "Band ratio (oracle)"),
    ("CYC", "Cyclostationary"),
]

TABLE1_COLS = [
    ("surgap", "Gap", "pp", 4000),
    ("Pf_white+1dB", "+1\\,dB", "%", 4000),
    ("Pf_lpf0.9", "LP\\,.9", "%", 4000),
    ("Pf_lpf0.8", "LP\\,.8", "%", 4000),
    ("Pf_rtl3dB", "RTL", "%", 4000),
    ("Pf_ar+0.1", "AR$+$", "%", 4000),
    ("Pf_ar-0.1", "AR$-$", "%", 4000),
    ("Pf_pink", "Pink", "%", 4000),
    ("Pd_-10", "$P_d$", "%", 4000),
    ("wcPd_U_receiver_-10", "$P_d^{\\mathrm{wc}}$", "%", 2000),
]

RECIPE_ROWS = [
    ("CLDNN", "CNN--LSTM", ["WHITE", "PINK", "PINKM"]),
    ("GapCNN", "GAP CNN", ["WHITE", "PINK", "PINKM", "RAND", "RANDM", "SURR", "SURRX", "SURRXM"]),
]

TABLE2_COLS = [
    ("Pf_pink", "Pink", "%", 4000),
    ("Pf_lpf0.8", "LP\\,.8", "%", 4000),
    ("Pf_rtl3dB", "RTL", "%", 4000),
    ("Pd_-10", "W", "%", 4000),
    ("PdIn_pink_-10", "Pink", "%", 2000),
    ("PdIn_lpf0.8_-10", "LP\\,.8", "%", 2000),
    ("wcPd_U_broad_-10", "wc", "%", 2000),
    ("surgap", "Gap", "pp", 4000),
]

RECLAB = {
    "WHITE": "\\textsc{White}",
    "PINK": "\\textsc{Pink}",
    "PINKM": "\\textsc{Pink}$_{\\mathrm{m}}$",
    "RAND": "\\textsc{Rand}",
    "RANDM": "\\textsc{Rand}$_{\\mathrm{m}}$",
    "SURR": "\\textsc{Surr}",
    "SURRX": "\\textsc{Surr}$^{\\ast}$",
    "SURRXM": "\\textsc{Surr}$^{\\ast}_{\\mathrm{m}}$",
}


def bootstrap_cell(vals, col_name, n_win=4000, b_reps=B):
    """Two-level bootstrap: Level 1 seed resampling + Level 2 finite-window binomial noise."""
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return {"mean": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "std": np.nan}
    n_seeds = len(vals)
    idx = RNG.choice(n_seeds, size=(b_reps, n_seeds), replace=True)
    sampled = vals[idx]
    if col_name == "surgap":
        # Finite-window standard error for surrogate gap across ~3-4 mid SNRs
        se_win = np.sqrt(0.5 / n_win / 3.0)
        reps = np.mean(sampled + RNG.normal(0, se_win, size=(b_reps, n_seeds)), axis=1)
    else:
        k = RNG.binomial(n_win, np.clip(sampled, 0, 1))
        reps = np.mean(k / float(n_win), axis=1)
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {
        "mean": float(np.mean(vals)),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "std": float(np.std(vals, ddof=1)) if n_seeds > 1 else 0.0,
        "n_seeds": int(n_seeds),
    }


def fmt_ci(res, unit):
    """Format mean [ci_lo, ci_hi] for LaTeX."""
    if np.isnan(res["mean"]):
        return "--"
    m = 100 * res["mean"]
    lo = 100 * res["ci_lo"]
    hi = 100 * res["ci_hi"]
    if unit == "pp":
        sign = "+" if m > 0.05 else ("-" if m < -0.05 else "")
        return f"{m:+.1f} [{lo:+.1f}, {hi:+.1f}]".replace("+", "$+$").replace("-", "$-$")
    return f"{m:.1f} [{lo:.1f}, {hi:.1f}]"


def run_table1_bootstrap(df):
    """Compute Table I with bootstrap CIs."""
    b = df[(df.N == 128) & (df.sps == 8)]
    results = {}
    tex_rows = []

    for model, label in LEARNED_MODELS:
        want_tag = "QAM16" if model == "DetectNetExact" else ""
        d = b[(b.kind == "learned") & (b.model == model) & (b.recipe == "WHITE") & (b.tag == want_tag)]
        if not len(d):
            continue
        row_dict = {}
        row_tex = [f"{label}", f"{len(d)}"]
        for c, col_h, u, n_win in TABLE1_COLS:
            vals = d[c].values if c in d else []
            r = bootstrap_cell(vals, c, n_win=n_win)
            row_dict[c] = r
            row_tex.append(fmt_ci(r, u))
        results[f"{model}_WHITE"] = row_dict
        tex_rows.append(" & ".join(row_tex) + " \\\\")

    tex_rows.append("\\midrule")

    for model, label in CLASSICAL_MODELS:
        d = b[(b.kind == "classical") & (b.model == model) & (b.tag == "")]
        if not len(d):
            continue
        row_dict = {}
        row_tex = [f"{label}", f"{len(d)}"]
        for c, col_h, u, n_win in TABLE1_COLS:
            vals = d[c].values if c in d else []
            r = bootstrap_cell(vals, c, n_win=n_win)
            row_dict[c] = r
            row_tex.append(fmt_ci(r, u))
        results[f"{model}_CLASSICAL"] = row_dict
        tex_rows.append(" & ".join(row_tex) + " \\\\")

    head = " & ".join(["Detector", "n"] + [h for _, h, _, _ in TABLE1_COLS])
    tex = (
        "\\begin{table*}[t]\n\\centering\n"
        "\\caption{Main evaluation with 95\\% two-level bootstrap confidence intervals over seeds and test windows ($N=128$, white-noise training). "
        "Format: mean [2.5\\%, 97.5\\%] ($B=2000$). Gap: percentage points; all other columns: \\%.}\n"
        "\\label{tab:supp-main-ci}\n\\scriptsize\n\\setlength{\\tabcolsep}{1.8pt}\n"
        "\\begin{tabular}{lr" + "r" * len(TABLE1_COLS) + "}\n\\toprule\n"
        f"{head} \\\\\n\\midrule\n" + "\n".join(tex_rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
    )
    return results, tex


def run_table2_bootstrap(df):
    """Compute Table II with bootstrap CIs."""
    b = df[(df.N == 128) & (df.sps == 8) & (df.kind == "learned") & (df.tag == "")]
    results = {}
    tex_rows = []

    for arch, alabel, recs in RECIPE_ROWS:
        first = True
        for rec in recs:
            d = b[(b.model == arch) & (b.recipe == rec)]
            if not len(d):
                continue
            row_dict = {}
            row_tex = [f"{alabel if first else ''}", f"{RECLAB[rec]}", f"{len(d)}"]
            first = False
            for c, col_h, u, n_win in TABLE2_COLS:
                vals = d[c].values if c in d else []
                r = bootstrap_cell(vals, c, n_win=n_win)
                row_dict[c] = r
                row_tex.append(fmt_ci(r, u))
            results[f"{arch}_{rec}"] = row_dict
            tex_rows.append(" & ".join(row_tex) + " \\\\")
        tex_rows.append("\\midrule")
    tex_rows = tex_rows[:-1]

    head = " & ".join(["Net", "Recipe", "n"] + [h for _, h, _, _ in TABLE2_COLS])
    tex = (
        "\\begin{table*}[t]\n\\centering\n"
        "\\caption{Training recipes with 95\\% two-level bootstrap confidence intervals ($N=128$, $B=2000$). "
        "Format: mean [2.5\\%, 97.5\\%]. Left columns: $P_f$ on noise shapes (\\%). Right columns: $P_d$ at $-10$\\,dB (\\%) and surrogate gap (pp).}\n"
        "\\label{tab:supp-recipes-ci}\n\\scriptsize\n\\setlength{\\tabcolsep}{2.0pt}\n"
        "\\begin{tabular}{llr" + "r" * len(TABLE2_COLS) + "}\n\\toprule\n"
        f"{head} \\\\\n\\midrule\n" + "\n".join(tex_rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
    )
    return results, tex


def paired_mcnemar_and_bootstrap(model, rec1, rec2, snr_key, n_win=2000, b_reps=B):
    """Paired McNemar test and two-level paired-difference bootstrap."""
    f1s = sorted(glob.glob(os.path.join(ROOT, "RESULTS", "scores", f"{model}_{rec1}_N128_sps8_s*.npz")))
    f1s = [f for f in f1s if "_rerun" not in f and "_nonorm" not in f and "_es" not in f and "_ntrain" not in f]

    seed_data = []
    b_tot, c_tot = 0, 0
    diffs = []

    for f1 in f1s:
        seed = f1.split("_s")[-1].replace(".npz", "")
        f2 = os.path.join(ROOT, "RESULTS", "scores", f"{model}_{rec2}_N128_sps8_s{seed}.npz")
        j1 = f1.replace("scores", "runs").replace(".npz", ".json")
        j2 = f2.replace("scores", "runs").replace(".npz", ".json")
        if not os.path.exists(f2) or not os.path.exists(j1) or not os.path.exists(j2):
            continue
        d1 = np.load(f1)
        d2 = np.load(f2)
        t1 = json.load(open(j1))["eval"]["thresholds"]["nominal"]
        t2 = json.load(open(j2))["eval"]["thresholds"]["nominal"]
        dec1 = (d1[snr_key] > t1).astype(int)
        dec2 = (d2[snr_key] > t2).astype(int)

        a = np.sum((dec1 == 1) & (dec2 == 1))
        b = np.sum((dec1 == 0) & (dec2 == 1))
        c = np.sum((dec1 == 1) & (dec2 == 0))
        d = np.sum((dec1 == 0) & (dec2 == 0))

        b_tot += b
        c_tot += c
        diffs.append((b - c) / float(len(dec1)))

        probs = np.array([a, b, c, d]) / float(len(dec1))
        seed_data.append(probs)

    S = len(seed_data)
    seed_probs = np.array(seed_data)

    # McNemar chi2 with Edwards continuity correction
    chi2 = float((abs(b_tot - c_tot) - 1) ** 2 / (b_tot + c_tot)) if (b_tot + c_tot) > 0 else 0.0

    # Two-sided p-value
    from scipy.stats import chi2 as chi2_dist
    p_val = float(chi2_dist.sf(chi2, df=1))

    # Two-level paired difference bootstrap
    reps = []
    for _ in range(b_reps):
        s_idx = RNG.choice(S, size=S, replace=True)
        deltas = []
        for s in s_idx:
            counts = RNG.multinomial(n_win, seed_probs[s])
            deltas.append((counts[1] - counts[2]) / float(n_win))
        reps.append(np.mean(deltas))
    reps = np.array(reps)
    lo, hi = np.percentile(reps, [2.5, 97.5])

    return {
        "model": model,
        "rec1": rec1,
        "rec2": rec2,
        "key": snr_key,
        "n_seeds": S,
        "n_windows_total": int(S * n_win),
        "mean_gain": float(np.mean(diffs)),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "b": int(b_tot),
        "c": int(c_tot),
        "chi2": chi2,
        "p_value": p_val,
    }


def make_paired_table_tex(paired_results):
    """Generate LaTeX table for paired tests."""
    rows = []
    for p in paired_results:
        m_gain = 100 * p["mean_gain"]
        ci_lo = 100 * p["ci_lo"]
        ci_hi = 100 * p["ci_hi"]
        p_str = "$< 10^{-300}$" if p["p_value"] < 1e-300 else f"${p['p_value']:.2e}$"
        comp = f"{p['model']} {RECLAB[p['rec2']]} vs {RECLAB[p['rec1']]}"
        test_cond = p["key"].replace("h1_", "").replace("_-10", "").replace("_", " ")
        rows.append(
            f"{comp} & {test_cond} & {p['n_windows_total']:,} & "
            f"$+{m_gain:.1f}$ [$+{ci_lo:.1f}$, $+{ci_hi:.1f}$] & {p['b']} / {p['c']} & {p['chi2']:.1f} & {p_str} \\\\"
        )

    tex = (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Paired hypothesis tests and two-level bootstrap confidence intervals for matched versus unmatched training recipes "
        "($N=128$, $-10$\\,dB nominal threshold, $B=2000$). "
        "Gain: $P_d$ increase in percentage points [95\\% CI]. $b/c$: discordant test windows. McNemar $\\chi^2$ with continuity correction.}\n"
        "\\label{tab:supp-paired-ci}\n\\footnotesize\n\\setlength{\\tabcolsep}{3.0pt}\n"
        "\\begin{tabular}{llrcccc}\n\\toprule\n"
        "Comparison & Noise condition & Windows & Gain (pp) [95\\% CI] & $b / c$ & $\\chi^2$ & $p$-value \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    return tex


def main():
    print("Computing statistical rigor metrics (two-level bootstrap B=2000, paired McNemar)...")
    df = load_runs()

    # Table 1
    t1_res, t1_tex = run_table1_bootstrap(df)

    # Table 2
    t2_res, t2_tex = run_table2_bootstrap(df)

    # Paired tests
    paired_configs = [
        ("GapCNN", "SURRX", "SURRXM", "h1_lpf0.8_-10", 2000),
        ("GapCNN", "RAND", "RANDM", "h1_lpf0.8_-10", 2000),
        ("GapCNN", "PINK", "PINKM", "h1_pink_-10", 2000),
        ("CLDNN", "PINK", "PINKM", "h1_pink_-10", 2000),
    ]

    paired_results = []
    for model, rec1, rec2, key, n_win in paired_configs:
        res = paired_mcnemar_and_bootstrap(model, rec1, rec2, key, n_win=n_win)
        paired_results.append(res)
        print(
            f"Paired {model} {rec2} vs {rec1} on {key}: "
            f"Gain = +{100*res['mean_gain']:.2f} pp [95% CI: {100*res['ci_lo']:.2f}, {100*res['ci_hi']:.2f}], "
            f"chi2 = {res['chi2']:.1f}, p = {res['p_value']:.2e}"
        )

    paired_tex = make_paired_table_tex(paired_results)

    # Save full stats_ci.json
    out_json = {
        "table1": t1_res,
        "table2": t2_res,
        "paired": paired_results,
        "B": B,
        "seed": 20261002,
    }
    json_path = os.path.join(ROOT, "RESULTS", "stats_ci.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)
    print(f"Saved stats to {json_path}")

    # Save combined TABLES/supp_ci.tex
    tex_path = os.path.join(ROOT, "TABLES", "supp_ci.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Generated by CODE/stats_ci.py -- do not edit by hand\n")
        f.write(t1_tex + "\n\n" + t2_tex + "\n\n" + paired_tex)
    print(f"Saved LaTeX tables to {tex_path}")


if __name__ == "__main__":
    main()
