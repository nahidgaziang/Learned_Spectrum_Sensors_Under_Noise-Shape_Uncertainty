"""Paper-ready LaTeX tables (booktabs) generated from RESULTS/ (never hand-edited).
TABLES/tab_main.tex    : Table I  - surrogate gap, Pf under noise shapes, Pd, worst-case Pd (N=128, WHITE recipe + classical)
TABLES/tab_main_col.tex: the same table as a single-column float (journal version)
TABLES/tab_recipes.tex : Table II - training recipes (CNN-LSTM, GAP-CNN), N=128
"""
import os
import numpy as np
import pandas as pd
from analysis import load_runs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEARNED = [("CLDNN", "CNN--LSTM~\\cite{gao2019deep}"), ("DetectNetExact", "DetectNet replica$^{\\dagger}$"),
           ("ResNet1D", "1-D ResNet~\\cite{zheng2020spectrum}"), ("CovCNN", "Covariance CNN~\\cite{liu2019cmcnn}"),
           ("STFTCNN", "STFT CNN~\\cite{chen2021stft}"), ("GapCNN", "GAP CNN (ours)")]
CLASSICAL = [("ED", "ED (known $\\sigma^2$)"), ("MME", "MME~\\cite{zeng2009eigen}"), ("SFM", "Spectral flatness"),
             ("BAND", "Band ratio (oracle)"), ("CYC", "Cyclostationary")]
COLS = [("surgap", "Gap", "pp"), ("Pf_white+1dB", "+1\\,dB", "%"), ("Pf_lpf0.9", "LP\\,.9", "%"), ("Pf_lpf0.8", "LP\\,.8", "%"),
        ("Pf_rtl3dB", "RTL", "%"), ("Pf_ar+0.1", "AR$+$", "%"), ("Pf_ar-0.1", "AR$-$", "%"), ("Pf_pink", "Pink", "%"),
        ("Pd_-10", "$P_d$", "%"), ("wcPd_U_receiver_-10", "$P_d^{\\mathrm{wc}}$", "%")]


def fmt(v, unit):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    if unit == "pp":
        return f"{100*v:+.1f}".replace("+", "$+$").replace("-", "$-$") if abs(100 * v) >= 0.05 else "0.0"
    return f"{100*v:.0f}"


def main_table(df):
    b = df[(df.N == 128) & (df.sps == 8) & (df.signal != "OFDM")]
    rows, sds = [], []
    for key, label in LEARNED:
        want_tag = "QAM16" if key == "DetectNetExact" else ""
        d = b[(b.kind == "learned") & (b.model == key) & (b.recipe == "WHITE") & (b.tag == want_tag)]
        if not len(d):
            continue
        vals = [fmt(d[c].mean() if c in d else np.nan, u) for c, _, u in COLS]
        sds += [(d[c].std(ddof=1), label.split("~")[0].split("$")[0].strip(), h) for c, h, u in COLS if c in d and len(d) > 1 and u == "%"]
        rows.append(f"{label} & " + " & ".join(vals) + f" \\\\")
    rows.append("\\midrule")
    for key, label in CLASSICAL:
        d = b[(b.kind == "classical") & (b.model == key) & (b.tag == "")]
        if not len(d):
            continue
        vals = [fmt(d[c].mean() if c in d else np.nan, u) for c, _, u in COLS]
        rows.append(f"{label} & " + " & ".join(vals) + f" \\\\")
    worst = max(sds, key=lambda t: t[0]) if sds else (float("nan"), "", "")
    maxsd = f"{100 * worst[0]:.1f} points ({worst[1]}, {worst[2]})"
    head = " & ".join(h for _, h, _ in COLS)
    # caption shortened 2026-10-05 for the 10-page version; the noise shapes are defined in the setup section
    cap = ("\\caption{Surrogate gap (points), false alarms $P_f$ (\\%) at the nominal threshold under "
           "noise power $+1$\\,dB and noise shapes absent from training, and detection $P_d$ (\\%) at $-10$\\,dB, nominal and worst case "
           "over the receiver set with $P_f\\le5\\%$ enforced; $N=128$, white-noise training, mean over 5 seeds (classical detectors: "
           f"3 test draws), largest seed s.d.\\ {maxsd}. $^{{\\dagger}}$Replica of DetectNet trained on QAM16 as in~\\cite{{gao2019deep}}.}}\n"
           "\\label{tab:main}\n")
    body = ("\\begin{tabular}{l" + "r" * len(COLS) + "}\n\\toprule\n"
            f"Detector & {head} \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    # two-column float (conference version, MANUSCRIPT/)
    tex = "\\begin{table*}[t]\n\\centering\n" + cap + "\\setlength{\\tabcolsep}{4.2pt}\n" + body + "\\end{table*}\n"
    open(os.path.join(ROOT, "TABLES", "tab_main.tex"), "w", encoding="utf-8").write(tex)
    # single-column float (journal version, 2026-10-05): the table is narrower than the text width, so a column float
    # saves space; footnotesize is the usual IEEE table size
    col = ("\\begin{table}[t]\n\\centering\n" + cap + "\\footnotesize\n\\setlength{\\tabcolsep}{1.5pt}\n"
           + body.replace("Covariance CNN~", "Cov.\\ CNN~").replace("DetectNet replica", "DetectNet rep.")
           .replace("Spectral flatness", "Spectral flat.") + "\\end{table}\n")
    open(os.path.join(ROOT, "TABLES", "tab_main_col.tex"), "w", encoding="utf-8").write(col)
    return tex


RECLAB = {"WHITE": "\\textsc{White}", "PINK": "\\textsc{Pink}", "PINKM": "\\textsc{Pink}$_{\\mathrm{m}}$", "RAND": "\\textsc{Rand}",
          "RANDM": "\\textsc{Rand}$_{\\mathrm{m}}$", "SURR": "\\textsc{Surr}", "SURRX": "\\textsc{Surr}$^{\\ast}$",
          "SURRXM": "\\textsc{Surr}$^{\\ast}_{\\mathrm{m}}$"}
RECIPE_ROWS = [("CLDNN", "CNN--LSTM", ["WHITE", "PINK", "PINKM"]),
               ("GapCNN", "GAP CNN", ["WHITE", "PINK", "PINKM", "RAND", "RANDM", "SURR", "SURRX", "SURRXM"])]


def recipe_table(df):
    """False alarms on coloured noise versus missed detections in the same colours (masking)."""
    b = df[(df.N == 128) & (df.sps == 8) & (df.kind == "learned") & (df.tag == "") & (df.signal != "OFDM")]
    cols = [("Pf_pink", "Pink", "%"), ("Pf_lpf0.8", "LP\\,.8", "%"), ("Pf_rtl3dB", "RTL", "%"),
            ("Pd_-10", "W", "%"), ("PdIn_pink_-10", "Pink", "%"), ("PdIn_lpf0.8_-10", "LP\\,.8", "%"),
            ("wcPd_U_broad_-10", "wc", "%"), ("surgap", "Gap", "pp")]
    rows, seeds = [], set()
    for arch, alabel, recs in RECIPE_ROWS:
        first = True
        for rec in recs:
            d = b[(b.model == arch) & (b.recipe == rec)]
            if not len(d):
                continue
            seeds.add(len(d))
            vals = [fmt(d[c].mean() if c in d else np.nan, u) for c, _, u in cols]
            rows.append(f"{alabel if first else ''} & {RECLAB[rec]} & " + " & ".join(vals) + " \\\\")
            first = False
        rows.append("\\midrule")
    rows = rows[:-1]
    ns = "/".join(str(s) for s in sorted(seeds, reverse=True))
    tex = ("\\begin{table}[t]\n\\centering\n\\caption{Training recipes ($N=128$). Left: $P_f$ (\\%) at the nominal threshold on coloured "
           "noise. Right: $P_d$ (\\%) at $-10$\\,dB with the signal in white (W), pink or LP\\,.8 noise at the nominal threshold, and "
           "worst case over the broad set with $P_f\\le5\\%$ enforced (wc). Subscript m: positives drawn in the same noise mixture "
           f"as the negatives. Mean over {ns} seeds.}}\n"
           "\\label{tab:recipes}\n\\footnotesize\n\\setlength{\\tabcolsep}{2.3pt}\n\\begin{tabular}{llrrrrrrrr}\n\\toprule\n"
           "& & \\multicolumn{3}{c}{$P_f$ on noise} & \\multicolumn{4}{c}{$P_d$, signal in} & \\\\\n"
           "\\cmidrule(lr){3-5}\\cmidrule(lr){6-9}\n"
           "Net & Recipe & " + " & ".join(h for _, h, _ in cols) + " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    open(os.path.join(ROOT, "TABLES", "tab_recipes.tex"), "w", encoding="utf-8").write(tex)
    return tex


# ============================================================================= journal-only tables
def _rng(vals, nd=0, pct=True):
    vals = [v for v in vals if v == v]
    lo, hi = min(vals), max(vals)
    f = (lambda x: f"{100*x:.{nd}f}") if pct else (lambda x: f"{x:.{nd}f}")
    return f(lo) if f(lo) == f(hi) else f"{f(lo)}--{f(hi)}"


def alpha_table():
    """Main findings at alpha = 1, 5, 10 % for the five white-trained architectures (ranges over architectures)."""
    import json
    A = pd.DataFrame(json.load(open(os.path.join(ROOT, "RESULTS", "alpha_sensitivity.json"))))
    archs = ["CLDNN", "ResNet1D", "CovCNN", "STFTCNN", "GapCNN"]
    rows = []
    for al in (0.01, 0.05, 0.10):
        d = A[(A.recipe == "WHITE") & (A.model.isin(archs)) & np.isclose(A.alpha, al)]
        b = A[(A.model == "BAND") & np.isclose(A.alpha, al)].iloc[0]
        gap = max(abs(x) for x in d.surgap)
        rows.append(f"{100*al:.0f} & {_rng(d.Pf_white, 1)} & {_rng(d['Pf_white+1dB'], 1)} & {_rng(d['Pf_lpf0.8'])} & {_rng(d['Pf_rtl3dB'])} & "
                    f"{_rng(d['Pf_pink'])} & {100*gap:.1f} & {_rng(d['Pd_-10'])} & {100*b['Pf_lpf0.8']:.0f} & {100*b['Pd_-10']:.0f} \\\\")
    tex = ("\\begin{table*}[t]\n\\centering\n\\caption{Sensitivity to the false-alarm target $\\alpha$ ($N=128$, white-trained networks, "
           "ranges over the five architectures; seed means). Thresholds from 2000 held-out white-noise test draws; $P_f$ in \\%. "
           "Gap: largest $|P_d^{\\mathrm{signal}}-P_d^{\\mathrm{surrogate}}|$ in points.}\n\\label{tab:alpha}\n"
           "\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{rrrrrrrrrr}\n\\toprule\n"
           "& \\multicolumn{7}{c}{White-trained networks} & \\multicolumn{2}{c}{Band ratio} \\\\\n\\cmidrule(lr){2-8}\\cmidrule(lr){9-10}\n"
           "$\\alpha$ (\\%) & $P_f$ white & $+1$\\,dB & LP\\,.8 & RTL & Pink & Gap & $P_d(-10)$ & LP\\,.8 & $P_d(-10)$ \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    open(os.path.join(ROOT, "TABLES", "tab_alpha.tex"), "w", encoding="utf-8").write(tex)
    return tex


def whitening_table():
    """Pf (%) after reference whitening versus the number K of reference windows (GAP CNN and CNN-LSTM, white-trained,
    3 seeds each): mean (max over the 6 runs)."""
    import glob, json
    WH = [json.load(open(f))["whitening"] for f in sorted(glob.glob(os.path.join(ROOT, "RESULTS", "runs", "wht_*.json")))]
    shapes = [("white", "White"), ("lpf0.8", "LP\\,.8"), ("rtl3dB", "RTL"), ("ar+0.2", "AR$+.2$"), ("pink", "Pink"), ("tilt1.0+", "Tilt")]
    rows = []
    for meth, Ks in [("cos", [16, 64, 256, 1024]), ("movavg", [16, 1024])]:
        for K in Ks:
            cells = []
            for s, _ in shapes:
                v = [w[f"K{K}_{meth}_{s}"]["Pf"] for w in WH]
                cells.append(f"{100*np.mean(v):.1f} ({100*max(v):.0f})")
            lab = "cosine fit" if meth == "cos" else "3-bin average"
            rows.append(f"{lab} & {K} & " + " & ".join(cells) + " \\\\")
    tex = ("\\begin{table}[t]\n\\centering\n\\caption{False alarms (\\%) after whitening with $K$ reference noise windows of the same "
           f"shape ($N=128$; mean (max) over {len(WH)} white-trained networks). Nominal $P_f=5\\%$.}}\n\\label{{tab:whitening}}\n"
           "\\footnotesize\n\\setlength{\\tabcolsep}{2.2pt}\n\\begin{tabular}{lr" + "r" * len(shapes) + "}\n\\toprule\n"
           "PSD estimate & $K$ & " + " & ".join(h for _, h in shapes) + " \\\\\n\\midrule\n" + "\n".join(rows) +
           "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    open(os.path.join(ROOT, "TABLES", "tab_whitening.tex"), "w", encoding="utf-8").write(tex)
    return tex


def impl_table():
    """Architectures: parameters, FLOPs, CPU/GPU latency, input representation, design mirrored."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import json
    import detectors as D
    
    prof_path = os.path.join(ROOT, "RESULTS", "complexity_profiling.json")
    prof = {}
    if os.path.exists(prof_path):
        prof = json.load(open(prof_path)).get("learned", {})
        
    info = [
        ("GapCNN", "GAP CNN", "I/Q, any $N$", "ours", "GAP CNN"),
        ("GapCNN4X", "GAP CNN 4X", "I/Q", "ours (scaled)", "GAP CNN 4X"),
        ("DetectNetExact", "DetectNet replica", "I/Q", "code of~\\cite{gao2019deep}", "DetectNet replica"),
        ("CLDNN", "CNN--LSTM", "I/Q", "\\cite{gao2019deep}", "CNN--LSTM"),
        ("ResNet1D", "1-D ResNet", "I/Q", "\\cite{zheng2020spectrum}", "1-D ResNet"),
        ("CovCNN", "Covariance CNN", "covariance", "\\cite{liu2019cmcnn}", "Covariance CNN"),
        ("STFTCNN", "STFT CNN", "STFT", "\\cite{chen2021stft}", "STFT CNN"),
        ("Transformer", "Transformer", "I/Q (16 tokens)", "ours (breadth)", "Transformer"),
        ("CycNet", "CycNet", "cyclic lags", "ours (escape)", "CycNet"),
    ]
    
    rows = []
    for key, name, rep, src, prof_key in info:
        m = D.ARCHS[key]()
        n = sum(p.numel() for p in m.parameters())
        p_data = prof.get(prof_key, {})
        mflops = p_data.get("mflops", "-")
        cpu_lat = p_data.get("cpu_latency_ms", 0.0)
        gpu_lat = p_data.get("gpu_latency_ms", 0.0)
        if isinstance(mflops, (int, float)):
            mflops_str = f"{mflops:.1f}"
        else:
            mflops_str = "-"
        if cpu_lat and gpu_lat:
            lat_str = f"{cpu_lat:.2f} / {gpu_lat:.2f}"
        else:
            lat_str = "-"
        rows.append(f"{name} & {n/1000:.0f}\\,k & {mflops_str} & {lat_str} & {rep} & {src} \\\\")
        
    tex = ("\\begin{table}[t]\n\\centering\n"
           "\\caption{Computational complexity and specifications of learned detectors ($N=128$). "
           "All inputs are power-normalised per example. Latency: forward pass of the network for one window (batch 1), in ms, "
           "on an Intel Core i5-13420H CPU / RTX 3050 GPU, excluding input preprocessing (for \\textsc{CycNet}, the cyclic features). "
           "Covariance: $16\\times16$ smoothed sample covariance. STFT: log-power, 16-point Hann window, hop 8. "
           "Training: Adam, binary cross-entropy, cosine learning-rate decay, 60\\,k examples (half signals), 12 epochs ($N=128$), "
           "batch 128, learning rate $10^{-3}$ ($3\\times10^{-4}$ for the CNN--LSTM).}\n"
           "\\label{tab:impl}\n\\scriptsize\n\\setlength{\\tabcolsep}{1.6pt}\n"
           "\\begin{tabular}{lrrccl}\n\\toprule\n"
           "Detector & Params & MFLOPs & Latency (ms) & Input & Ref. \\\\\n"
           " & & & (CPU / GPU) & & \\\\\n\\midrule\n" + "\n".join(rows) +
           "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    open(os.path.join(ROOT, "TABLES", "tab_impl.tex"), "w", encoding="utf-8").write(tex)
    return tex


def adaptive_table():
    """Classical adaptive baselines (AMF, ACE, Kelly, ED-ref) with K=1024 reference windows on both recorded receivers."""
    import json
    p_hf = os.path.join(ROOT, "RESULTS", "adaptive_baselines.json")
    p_tt = os.path.join(ROOT, "RESULTS", "adaptive_baselines_trevtron.json")
    if not (os.path.exists(p_hf) and os.path.exists(p_tt)):
        return ""
    d_hf = json.load(open(p_hf))
    d_tt = json.load(open(p_tt))
    s_hf = d_hf.get("summary", d_hf.get("recorded_summary", {}))
    s_tt = d_tt.get("summary", d_tt.get("recorded_summary", {}))

    rows_def = [
        ("AMF", "AMF~\\cite{robey1992amf}", "CFAR (hom.~noise)"),
        ("ACE", "ACE~\\cite{kraut2001ace}", "Scale-inv.~CFAR"),
        ("KELLY", "Kelly~\\cite{kelly1986adaptive}", "GLRT CFAR"),
        ("ED-ref", "ED-ref", "Power-ref.~ED")
    ]
    lines = []
    for k, name, prop in rows_def:
        vh = s_hf.get(k, {})
        vt = s_tt.get(k, {})
        pf_h = f"{100*vh.get('Pf_mean', 0):.1f}"
        pd_h = f"{100*vh.get('Pd_m10_mean', 0):.1f}"
        pf_t = f"{100*vt.get('Pf_mean', 0):.1f}"
        pd_t = f"{100*vt.get('Pd_m10_mean', 0):.1f}"
        lines.append(f"{name} & {prop} & {pf_h} & {pd_h} & {pf_t} & {pd_t} \\\\")

    tex = ("\\begin{table}[t]\n\\centering\n\\caption{Classical adaptive baselines ($N=128$, $K=1024$ reference windows) on recorded "
           "noise from two distinct receivers. Nominal $P_f=5\\%$. Mean $P_f$ and mean $P_d(-10\\,\\mathrm{dB})$ over clean recordings. "
           "Receiver 1: Nooelec NESDR (R820T2 tuner); Receiver 2: RTL-SDR Blog V4 (R828D tuner).}\n\\label{tab:adaptive}\n"
           "\\footnotesize\n\\setlength{\\tabcolsep}{2.2pt}\n\\begin{tabular}{llrrrr}\n\\toprule\n"
           "& & \\multicolumn{2}{c}{Receiver 1 (R820T2)} & \\multicolumn{2}{c}{Receiver 2 (R828D)} \\\\\n\\cmidrule(lr){3-4} \\cmidrule(lr){5-6}\n"
           "Detector & Invariance Property & $P_f$ (\\%) & $P_d$ (\\%) & $P_f$ (\\%) & $P_d$ (\\%) \\\\\n\\midrule\n"
           + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    out_p = os.path.join(ROOT, "TABLES", "tab_adaptive.tex")
    with open(out_p, "w", encoding="utf-8") as f:
        f.write(tex)
    return tex
def real_signals_table():
    """Surrogate gap (points, 95% bootstrap CI) on real over-the-air signals in real receiver noise."""
    import json
    p_rs = os.path.join(ROOT, "RESULTS", "real_signal_summary.json")
    if not os.path.exists(p_rs):
        return ""
    RS = json.load(open(p_rs))["results"]

    # Every class/receiver combination that was evaluated is listed (fix 2026-10-05, internal notes 1.2: NOAA,
    # FRS/GMRS and APRS were missing). "n.d." = not detected: the pooled P_d never leaves [0, 0.1], so the gap (mean over
    # SNRs with 0.1 < P_d < 0.9) is undefined.
    S = json.load(open(os.path.join(ROOT, "RESULTS", "real_signal_surrogate.json")))
    specs = [
        ("hf_composite", "FM_broadcast", "FM broadcast", "R820T2", "WBFM", "KAPPALO"),
        ("trevtron_R828D", "FM_broadcast", "FM broadcast", "R828D", "WBFM", "KAPPALO"),
        ("trevtron_R828D", "NOAA_weather", "NOAA weather", "R828D", "NBFM", "KAPPALO"),
        ("trevtron_R828D", "FRS_GMRS", "FRS/GMRS", "R828D", "NBFM", "KAPPALO"),
        ("trevtron_R828D", "pager", "Pager", "R828D", "FSK", "KAPPALO"),
        ("hf_composite", "ISM_sensors", "ISM sensors", "R820T2", "FSK/OOK", "KAPPALO"),
        ("trevtron_R828D", "ISM_sensors", "ISM sensors", "R828D", "FSK/OOK", "KAPPALO"),
        ("hf_composite", "ADS_B", "ADS-B", "R820T2", "PPM", "KAPPA"),
        ("trevtron_R828D", "APRS", "APRS", "R828D", "AFSK", None),
    ]

    def cell(rx, cname, det):
        v = RS.get(f"{rx}|{cname}|{det}|sur")
        if not v:
            return "--"
        g = v["gap"]
        if g == g:
            return "0.0" if abs(100 * g) < 0.05 else f"{100*g:+.1f}".replace("+", "$+$").replace("-", "$-$")
        return "n.d." if max(v["pd_sig"]) <= 0.1 else "--"

    def maxpd(rx, cname, det):
        v = RS.get(f"{rx}|{cname}|{det}|sur")
        return f"{max(v['pd_sig']):.2f}" if v else "--"

    lines = []
    for rx, cname, dname, recv, mod, sdet in specs:
        C = S["receivers"][rx]["classes"][cname]
        used, nfiles = len(C["captures"]), C["n_files"]
        if used == 0:
            lines.append(f"{dname} & {recv} & {mod} & 0/{nfiles} & 0 & \\multicolumn{{6}}{{l}}{{no capture with a detectable signal}} \\\\")
            continue
        wins = sum(c["n"] for c in C["captures"].values())
        own = np.median([c["info"]["own_snr_db"] for c in C["captures"].values()])
        sname = "$\\kappa_{\\mathrm{lo}}$" if sdet == "KAPPALO" else "$|\\kappa-2|$"
        lines.append(f"{dname} & {recv} & {mod} & {used}/{nfiles} & {wins:,} & ${own:.0f}$ & "
                     f"{maxpd(rx, cname, 'GapCNN-WHITE')} / {maxpd(rx, cname, 'BAND')} & {cell(rx, cname, 'GapCNN-WHITE')} & "
                     f"{cell(rx, cname, 'BAND')} & {cell(rx, cname, sdet)} ({sname}) & {cell(rx, cname, 'GapCNN-SURRX')} \\\\")

    tex = ("\\begin{table*}[t]\n\\centering\n\\caption{Real over-the-air signals mixed into real noise of the same receiver ($N=128$; "
           "thresholds at $P_f=5\\%$ on that receiver's noise). Captures: used/downloaded (a capture is used if it has at least 20 windows "
           "6~dB above its noise floor in its occupied bins). Own SNR: median in-band SNR of the selected signal windows (dB), i.e.\\ the "
           "share of the capture's own receiver noise. Max $P_d$: largest detection rate over $\\gamma\\in[-24,4]$~dB for the white-trained "
           "GAP CNN and the centred band ratio. Gap: $P_d^{\\mathrm{signal}}-P_d^{\\mathrm{surrogate}}$ in points, mean over SNRs with "
           "$0.1<P_d<0.9$ (context Fourier surrogates); n.d.: not detected (max $P_d\\le0.1$), gap undefined. Structure statistic: "
           "normalised fourth moment, one-sided ($\\kappa_{\\mathrm{lo}}$, constant envelope) or two-sided ($|\\kappa-2|$, pulsed); "
           "negative values arise when the capture's own heavy-tailed receiver noise dominates (low own SNR). Networks: mean over 3 seeds.}\n"
           "\\label{tab:realsig}\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{lllrrrcrrrr}\n\\toprule\n"
           "Class & Receiver & Mod. & Captures & Windows & Own SNR & Max $P_d$ CNN / band & Gap CNN & Gap band & Gap structure & "
           "Gap \\textsc{Surr}$^{\\ast}$ \\\\\n\\midrule\n"
           + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    out_p = os.path.join(ROOT, "TABLES", "tab_real_signals.tex")
    with open(out_p, "w", encoding="utf-8") as f:
        f.write(tex)
    return tex


def fit_supp_ci():
    """TABLES/supp_ci_fit.tex: the tables of supp_ci.tex (written by stats_ci.py) with every tabular scaled to the line
    width, for the one-column supplement. Text transform only; no statistic is recomputed."""
    import re
    src = os.path.join(ROOT, "TABLES", "supp_ci.tex")
    if not os.path.exists(src):
        return ""
    t = open(src, encoding="utf-8").read()
    # only the wide (table*) tables are scaled down; the narrow paired-test table is left at its natural size
    t = re.sub(r"(\\begin\{table\*\}.*?\\end\{table\*\})",
               lambda m: re.sub(r"(\\begin\{tabular\}.*?\\end\{tabular\})",
                                lambda n: "\\resizebox{\\linewidth}{!}{%\n" + n.group(1) + "}", m.group(1), flags=re.S),
               t, flags=re.S)
    with open(os.path.join(ROOT, "TABLES", "supp_ci_fit.tex"), "w", encoding="utf-8") as fh:
        fh.write("% generated by paper_tables.py: fit_supp_ci() from supp_ci.tex (stats_ci.py) -- do not edit\n" + t)
    return t


if __name__ == "__main__":
    df = load_runs()
    print(main_table(df))
    print(recipe_table(df))
    print(alpha_table())
    print(whitening_table())
    print(impl_table())
    print(adaptive_table())
    print(real_signals_table())
    fit_supp_ci()



