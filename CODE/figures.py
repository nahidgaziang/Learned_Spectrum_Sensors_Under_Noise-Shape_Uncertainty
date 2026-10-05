"""Publication figures, generated only from machine-readable results.

Palette (validated with the dataviz validator, light mode, white surface):
  categorical  #2a78d6 blue, #eb6834 orange, #1baf7a aqua, #4a3aa7 violet  (all-pairs CVD >= 9.2, normal >= 16.3)
  ordinal (N)  #86b6ef, #2a78d6, #104281                                    (monotone, single hue)
Aqua is below 3:1 contrast on white -> every series also carries a marker / line style and a legend.

Writes FIGURES/fig_theory.pdf, fig_surrogate.pdf, fig_wall.pdf (+ .png previews).
"""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim, wall as W

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "FIGURES")
os.makedirs(FIG, exist_ok=True)
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
ORD = ["#86b6ef", "#2a78d6", "#104281"]
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#9a998f"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.0, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
    "xtick.labelsize": 8.0, "ytick.labelsize": 8.0, "legend.fontsize": 8.0, "axes.edgecolor": AXIS,
    "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-", "legend.frameon": False,
    "lines.linewidth": 1.3, "lines.markersize": 3.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300, "figure.dpi": 150,
})
W1 = 3.45   # IEEE single-column width (in), conference figure
WC = 3.5    # IEEEtran journal \columnwidth = 252 pt = 3.5 in: journal figures are drawn at exactly this width
WT = 6.8    # 0.95 \textwidth (7.16 in) for the two-column wall figure
FS = 8.0    # minimum font size at printed size (IEEE recommends >= 8 pt); journal figures are included at 100 %


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)


def save_exact(fig, name):
    """Journal figures (2026-10-05): saved at exactly the figure size (constrained layout, no tight cropping), so that
    \\includegraphics at the same width prints every font at its nominal size (>= FS)."""
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=200)
    plt.close(fig)


def new_fig(w, h, **kw):
    fig, ax = plt.subplots(figsize=(w, h), layout="constrained", **kw)
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.04, hspace=0.02)
    return fig, ax


# ----------------------------------------------------------------------------- Fig. 1: theory
def fig_theory():
    u = np.linspace(0.25, 3.0, 56)
    ed = 10 * np.log10([sim.ed_power_wall(x) for x in u])
    p = W.signal_psd(512)
    sh = 10 * np.log10([W.wall(p, x) for x in u])
    flat05 = 10 * np.log10([sim.shape_wall(0.5, x) for x in u])
    fig, ax = new_fig(WC, 1.5)
    ax.plot(u, ed, color=ORANGE, ls="-", marker="o", markevery=(5, 11), label="Energy det., power unc.")
    ax.plot(u, flat05, color=VIOLET, ls="--", marker="s", markevery=(5, 11), label=r"Scale-inv., shape unc., $\rho=0.5$")
    ax.plot(u, sh, color=BLUE, ls="-", marker="^", markevery=(5, 11), label=r"Scale-inv., shape unc., RRC")
    ax.set_xlabel(r"Peak-to-peak uncertainty $u$ (dB)")
    ax.set_ylabel("SNR wall (dB)")
    ax.set_xlim(0.2, 3.05)
    ax.set_ylim(-19, 9)
    ax.legend(loc="upper left", handlelength=2.0, labelspacing=0.2, borderaxespad=0.2)
    save_exact(fig, "fig_theory")


# ----------------------------------------------------------------------------- data helpers
def runs(pattern):
    out = []
    for f in glob.glob(os.path.join(ROOT, "RESULTS", "runs", pattern)):
        out.append(json.load(open(f)))
    return out


def curve(rs, key="Pd", sub="white", thr="nominal"):
    snrs = [int(s) for s in rs[0]["eval"]["Pd"].keys()]
    vals = []
    for r in rs:
        if key == "Pd":
            vals.append([r["eval"]["Pd"][str(s)][sub][thr] for s in snrs])
        else:
            vals.append([r["eval"]["Pd_surrogate"][str(s)][thr] for s in snrs])
    v = np.array(vals)
    return np.array(snrs), v.mean(0), (v.std(0, ddof=1) if len(v) > 1 else np.zeros(v.shape[1]))


# ----------------------------------------------------------------------------- Fig. 2: surrogate diagnostic
def fig_surrogate():
    spec = [("cls_BAND_N128_sps8_s*.json", "Band ratio (oracle)", AQUA, "D"),
            ("CLDNN_WHITE_N128_sps8_s*.json", "CNN-LSTM, white neg.", BLUE, "o"),
            ("GapCNN_SURRX_N128_sps8_s*.json", r"CNN, Surr$^{\ast}$ neg.", ORANGE, "s"),
            ("cls_CYC_N128_sps8_s*.json", "Cyclostationary", VIOLET, "^")]
    fig, ax = new_fig(WC, 2.15)
    for pat, lab, col, mk in spec:
        rs = runs(pat)
        if not rs:
            continue
        x, m, s = curve(rs, "Pd")
        xs, ms, ss = curve(rs, "sur")
        ax.plot(x, m, color=col, marker=mk, ls="-", label=lab)
        ax.fill_between(x, m - s, m + s, color=col, alpha=0.12, lw=0)
        ax.plot(xs, ms, color=col, marker=mk, ls=":", mfc="white", mew=0.9)
    ax.axhline(0.05, color=INK2, lw=0.6)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel(r"$P_d$ (nominal threshold)")
    ax.set_ylim(0, 1.02)
    ax.set_xticks([-20, -14, -10, -6, 0])
    ax.set_xticks([-17, -12, -8, -3], minor=True)   # unlabelled: neighbouring labels would overlap at 8 pt
    h, l = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    h += [Line2D([], [], color=INK2, ls="-"), Line2D([], [], color=INK2, ls=":", marker="o", mfc="white")]
    l += ["solid: signal", "dotted: surrogate"]
    ax.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, handlelength=1.6, labelspacing=0.15,
              columnspacing=0.8, borderaxespad=0.0, handletextpad=0.4)
    save_exact(fig, "fig_surrogate")


# ----------------------------------------------------------------------------- Fig. 3: wall test
STRUCT_NET = "GapCNN-SURRXM"     # structure-forced network in panel (b): the matched exact-surrogate recipe (decided before its wall results)


def fig_wall(R="1.0", struct_net=None):
    """(a) worst-case Pd versus SNR - wall for the white-trained CNN and the band ratio, N = 128/512/2048.
    (b) worst-case Pd AT the wall versus N (128 ... 32768): second-order tests stay at alpha, structure need not."""
    struct_net = struct_net or STRUCT_NET
    files = sorted(glob.glob(os.path.join(ROOT, "RESULTS", "wall_test_s*.json")))
    if not files:
        return
    data = [json.load(open(f)) for f in files]
    longd = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(ROOT, "RESULTS", "wall_test_long_s*.json")))]
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 2, figsize=(W1, 1.95), gridspec_kw={"width_ratios": [1, 1.08]})
    ax = axes[0]
    for det, ls, mk in (("GapCNN-WHITE", "-", "o"), ("BAND", "--", "D")):
        for N, col in zip([128, 512, 2048], ORD):
            key = f"N{N}_R{R}_{det}"
            if not all(key in d for d in data):
                continue
            deltas = [int(k) for k in data[0][key]["Pd"].keys()]
            v = np.array([[d[key]["Pd"][str(k)]["worst"] for k in deltas] for d in data])
            ax.plot(deltas, v.mean(0), color=col, ls=ls, marker=mk, ms=2.6, lw=1.1, label=(f"$N$={N}" if ls == "-" else None))
    ax.axhline(0.05, color=INK2, lw=0.6)
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_title("(a) second-order detectors", loc="left", pad=2, fontsize=6.6)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"SNR $-$ wall (dB)")
    ax.set_ylabel(r"worst-case $P_d$")
    h, l = ax.get_legend_handles_labels()
    h += [Line2D([], [], color=INK2, ls="-"), Line2D([], [], color=INK2, ls="--")]
    l += ["CNN", "band ratio"]
    leg = ax.legend(h, l, loc="upper left", handlelength=1.3, labelspacing=0.12, fontsize=5.4, borderaxespad=0.15, handletextpad=0.4,
                    frameon=True, framealpha=0.92, edgecolor="none", facecolor="white", borderpad=0.25)
    leg.set_zorder(5)
    # (b) at the wall versus N
    ax = axes[1]
    series = [("GapCNN-WHITE", "CNN, white neg.", BLUE, "o", "-"), (struct_net, "CNN, surrogate neg.", ORANGE, "s", "-"),
              ("BAND", "band ratio", AQUA, "D", "--"), ("CYC", "cyclostationary", VIOLET, "^", "--")]
    for det, lab, col, mk, ls in series:
        Ns, m = [], []
        for N in [128, 512, 2048]:
            key = f"N{N}_R{R}_{det}"
            vals = [d[key]["Pd"]["0"]["worst"] for d in data if key in d]
            if vals:
                Ns.append(N); m.append(np.mean(vals))
        for N in [8192, 32768]:
            vals = [d[f"N{N}_R{R}"]["Pd"]["0"][det]["worst"] for d in longd if f"N{N}_R{R}" in d and det in d[f"N{N}_R{R}"]["Pd"]["0"]]
            if vals:
                Ns.append(N); m.append(np.mean(vals))
        if Ns:
            ax.plot(Ns, m, color=col, ls=ls, marker=mk, ms=3.0, lw=1.1, label=lab)
    ax.axhline(0.05, color=INK2, lw=0.6)
    ax.set_xscale("log", base=2)
    ax.set_xticks([128, 512, 2048, 8192, 32768])
    ax.set_xticklabels(["128", "512", "2k", "8k", "32k"])
    ax.minorticks_off()
    ax.set_ylim(0, None)
    ax.set_title("(b) at the wall", loc="left", pad=2, fontsize=6.6)
    ax.set_xlabel(r"observation length $N$")
    ax.legend(loc="upper left", handlelength=1.5, labelspacing=0.12, fontsize=5.6, borderaxespad=0.2)
    fig.tight_layout(pad=0.3, w_pad=0.5)
    save(fig, "fig_wall")


# ----------------------------------------------------------------------------- Fig. (real noise): dose-response
def fig_real_noise():
    import re
    f3 = os.path.join(ROOT, "RESULTS", "real_noise_v3_N128.json")
    f2 = os.path.join(ROOT, "RESULTS", "real_noise_v2_N128.json")
    f1 = os.path.join(ROOT, "RESULTS", "real_noise_N128.json")
    src = f3 if os.path.exists(f3) else (f2 if os.path.exists(f2) else f1)
    R = json.load(open(src))["recordings"]
    white = [a + "-WHITE" for a in ["CLDNN", "ResNet1D", "CovCNN", "STFTCNN", "GapCNN"]]

    def det_mean(rec, field, prefixes):
        ks = [k for k in rec[field] if re.sub(r"-s\d+$", "", k) in prefixes]
        return float(np.mean([rec[field][k] for k in ks])) if ks else np.nan
    import collections
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    for name, rec in R.items():
        if rec["shape"].get("narrowband"):
            continue
        g = "control" if rec["group"] == "unlabelled" else rec["meta"].get("center_freq", "?")
        groups[g]["x"].append(rec["shape"]["edge_rolloff_db"])
        groups[g]["white"].append(det_mean(rec, "Pf", white))
        whf = "Pf_wbig" if "Pf_wbig" in rec else "Pf_whitened"
        groups[g]["white_wh"].append(det_mean(rec, whf, white))
        groups[g]["BAND"].append(det_mean(rec, "Pf", ["BAND"]))
        groups[g]["SURR"].append(det_mean(rec, "Pf", ["GapCNN-SURR"]))
        groups[g]["CYC"].append(det_mean(rec, "Pf", ["CYC"]))
    fig, ax = new_fig(WC, 2.15)
    style = [("white", "Nets, white neg. (5 archs)", BLUE, "o", "full", 0.0),
             ("white_wh", "Same, whitened", BLUE, "o", "none", 0.06),
             ("BAND", "Band ratio (oracle)", AQUA, "D", "full", -0.06),
             ("SURR", "CNN, surrogate neg.", ORANGE, "s", "full", 0.12),
             ("CYC", "Cyclostationary", VIOLET, "^", "full", -0.12)]
    for key, lab, col, mk, fill, dx in style:
        xm = [np.mean(v["x"]) + dx for v in groups.values()]
        ym = [100 * np.mean(v[key]) for v in groups.values()]
        ys = [100 * np.std(v[key]) for v in groups.values()]
        ax.errorbar(xm, ym, yerr=ys, ls="none", marker=mk, ms=3.8, mfc=(col if fill == "full" else "white"), mec=col, mew=0.9,
                    ecolor=col, elinewidth=0.7, capsize=1.5, label=lab)
    ax.axhline(5, color=INK2, lw=0.6)
    ax.set_xlabel("Measured band-edge roll-off (dB)")
    ax.set_ylabel(r"$P_f$ (%)")
    ax.set_ylim(0, None)
    ax.set_xlim(-0.6, 6.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, handlelength=1.0, labelspacing=0.15,
              columnspacing=0.6, borderaxespad=0.0, handletextpad=0.3)   # below the axes: does not cover high roll-off points
    save_exact(fig, "fig_real_noise")
    return src


# ============================================================================= journal-only figures
def fig_wall_journal(R="1.0"):
    """As fig_wall, but panel (b) adds the GAP CNNs trained at N = 2048 applied to N = 8192 and 32768
    (open markers), from wall_test_long_nets_s*.json. 2-column wide layout for IEEE journal."""
    files = sorted(glob.glob(os.path.join(ROOT, "RESULTS", "wall_test_s*.json")))
    data = [json.load(open(f)) for f in files]
    longd = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(ROOT, "RESULTS", "wall_test_long_s*.json")))]
    lnets = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(ROOT, "RESULTS", "wall_test_long_nets_s*.json")))]
    from matplotlib.lines import Line2D
    # 2026-10-05: legends sit below the panels (they overlapped the curves and ran into the next panel when inside)
    fig, axes = plt.subplots(1, 3, figsize=(WT, 2.1), layout="constrained", gridspec_kw={"width_ratios": [1, 1, 0.78]})
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.06)
    below = dict(loc="upper center", bbox_to_anchor=(0.5, -0.3), fontsize=FS, handlelength=1.5, labelspacing=0.15,
                 columnspacing=0.9, borderaxespad=0.0, handletextpad=0.4)
    fig_escape_panel(axes[2], below)
    ax = axes[0]
    for det, ls, mk in (("GapCNN-WHITE", "-", "o"), ("BAND", "--", "D")):
        for N, col in zip([128, 512, 2048], ORD):
            key = f"N{N}_R{R}_{det}"
            deltas = [int(k) for k in data[0][key]["Pd"].keys()]
            v = np.array([[d[key]["Pd"][str(k)]["worst"] for k in deltas] for d in data])
            ax.plot(deltas, v.mean(0), color=col, ls=ls, marker=mk, ms=3.0, lw=1.1, label=(f"$N$={N}" if ls == "-" else None))
    ax.axhline(0.05, color=INK2, lw=0.6); ax.axvline(0, color=INK2, lw=0.6)
    ax.set_title("(a) second-order detectors", loc="left", pad=2, fontsize=FS)
    ax.set_ylim(0, 1.05); ax.set_yticks([0, 0.5, 1.0]); ax.set_xlabel(r"SNR $-$ wall (dB)"); ax.set_ylabel(r"worst-case $P_d$")
    h, l = ax.get_legend_handles_labels()
    h += [Line2D([], [], color=INK2, ls="-"), Line2D([], [], color=INK2, ls="--")]
    l += ["CNN", "band ratio"]
    ax.legend(h, l, ncol=3, **below)
    ax = axes[1]
    series = [("GapCNN-WHITE", "CNN, white neg.", BLUE, "o", "-"), ("GapCNN-SURRXM", r"CNN, Surr$^{\ast}_{\mathrm{m}}$", ORANGE, "s", "-"),
              ("BAND", "band ratio", AQUA, "D", "--"), ("CYC", "cyclostationary", VIOLET, "^", "--")]
    for det, lab, col, mk, ls in series:
        Ns, m = [], []
        for N in [128, 512, 2048]:
            vals = [d[f"N{N}_R{R}_{det}"]["Pd"]["0"]["worst"] for d in data if f"N{N}_R{R}_{det}" in d]
            if vals:
                Ns.append(N); m.append(np.mean(vals))
        src = longd if det in ("BAND", "CYC") else lnets
        Nl, ml = [], []
        for N in [8192, 32768]:
            vals = [d[f"N{N}_R{R}"]["Pd"]["0"][det]["worst"] for d in src if f"N{N}_R{R}" in d and det in d[f"N{N}_R{R}"]["Pd"]["0"]]
            if vals:
                Nl.append(N); ml.append(np.mean(vals))
        if det in ("BAND", "CYC"):
            ax.plot(Ns + Nl, m + ml, color=col, ls=ls, marker=mk, ms=3.0, lw=1.1, label=lab)
        else:
            ax.plot(Ns, m, color=col, ls=ls, marker=mk, ms=3.0, lw=1.1, label=lab)
            if Nl:
                ax.plot([Ns[-1]] + Nl, [m[-1]] + ml, color=col, ls=":", lw=1.0, zorder=1)
                ax.plot(Nl, ml, color=col, ls="none", marker=mk, ms=3.0, mfc="white", mew=0.9, zorder=3)
    ax.axhline(0.05, color=INK2, lw=0.6)
    ax.set_xscale("log", base=2); ax.set_xticks([128, 512, 2048, 8192, 32768]); ax.set_xticklabels(["128", "512", "2k", "8k", "32k"])
    ax.minorticks_off(); ax.set_ylim(0, None)
    ax.set_title("(b) at the wall", loc="left", pad=2, fontsize=FS); ax.set_xlabel(r"observation length $N$")
    ax.set_ylabel(r"worst-case $P_d$")
    ax.legend(ncol=2, **below)
    save_exact(fig, "fig_wall_journal")

def fig_templates():
    """2026-10-05 (U1). (a) Implicit spectral templates of the white-trained networks (ridge fit of their logits on the
    normalised periodogram, mean over 5 seeds, standardised) against the LMPI template (signal PSD p) and the band-ratio
    indicator, N = 128. (b) False alarms under the LP .8 roll-off at the nominal threshold versus N: LMPI test replayed on
    the networks' test windows and the white-trained GAP CNN (mean over seeds)."""
    L = json.load(open(os.path.join(ROOT, "RESULTS", "lmpi_equivalence.json")))
    M, prof = L["models"], L["lmpi_profile"]
    f = np.sort(np.fft.fftfreq(128))
    z = lambda v: (np.asarray(v) - np.mean(v)) / np.std(v)
    fig, axes = plt.subplots(1, 2, figsize=(WC, 2.12), layout="constrained", gridspec_kw={"width_ratios": [1.45, 1]})
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.04)
    ax = axes[0]
    p0 = prof["N128_sps8_s0"]
    ax.plot(f, z(p0["template_band"]), color=AXIS, ls="--", lw=1.0, label="band")
    ax.plot(f, z(p0["template_p"]), color=INK, lw=1.4, label=r"LMPI ($p$)")
    for arch, col, ls in (("GapCNN", BLUE, "-"), ("CLDNN", ORANGE, (0, (4, 1.5))), ("CovCNN", AQUA, (0, (1.5, 1.2)))):
        T = [z(M[f"{arch}_WHITE_N128_sps8_s{s}"]["template_vec"]) for s in range(5)]
        ax.plot(f, np.mean(T, 0), color=col, ls=ls, lw=1.1, label={"GapCNN": "GAP CNN", "CLDNN": "CNN-LSTM", "CovCNN": "Cov CNN"}[arch])
    ax.set_xlim(-0.5, 0.5); ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5]); ax.set_xticklabels(["-.5", "-.25", "0", ".25", ".5"])
    ax.set_xlabel("frequency (cycles/sample)"); ax.set_ylabel("weight (std.)")
    ax.set_title("(a) spectral templates", loc="left", pad=2, fontsize=FS)
    # legend below the panel (inside, it covered the network curves; 2026-10-05)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), fontsize=FS, handlelength=1.5, labelspacing=0.15,
              columnspacing=0.8, borderaxespad=0.0, handletextpad=0.4, ncol=2)
    ax.set_ylim(-1.3, 3.8)
    ax = axes[1]
    Ns = [128, 512, 2048, 8192]
    lm = [np.mean([p["Pf"]["lpf0.8"] for g, p in prof.items() if g.startswith(f"N{N}_sps8_s") and "QAM16" not in g]) for N in Ns]
    gm = [np.mean([m["h0_lpf0.8"]["pf_model"] for k, m in M.items() if k.startswith(f"GapCNN_WHITE_N{N}_sps8_s")
                   and k.split("_s")[-1].isdigit()]) for N in Ns]
    ax.plot(Ns, gm, color=BLUE, marker="o", ms=3.0, lw=1.1, label="GAP CNN")
    ax.plot(Ns, lm, color=INK, marker="s", ms=3.0, mfc="white", mew=0.9, lw=1.0, ls="--", label="LMPI")
    ax.axhline(0.05, color=INK2, lw=0.6)
    ax.set_xscale("log", base=2); ax.set_xticks(Ns); ax.set_xticklabels(["128", "512", "2k", "8k"]); ax.minorticks_off()
    ax.set_ylim(0, 1.05); ax.set_xlabel(r"$N$"); ax.set_ylabel(r"$P_f$, LP .8")
    ax.set_title("(b) false alarms", loc="left", pad=2, fontsize=FS)
    ax.legend(loc="lower right", fontsize=FS, handlelength=1.4, labelspacing=0.1, borderaxespad=0.1)
    save_exact(fig, "fig_templates")


def fig_offband():
    """2026-10-05 (U6, supplement). Detection of shifted signals (8 samples/symbol, -6 dB) versus carrier offset:
    measured (markers) and predicted from the template overlap (lines) for the LMPI test, the band ratio and the
    white-trained GAP CNN (mean of 3 seeds); right axis: overlap of the training template."""
    O = json.load(open(os.path.join(ROOT, "RESULTS", "offband_sweep.json")))
    rows = [r for r in O["rows"] if r["sps"] == 8 and r["snr_db"] == -6]
    f0 = [r["f0"] for r in rows]
    fig, ax = new_fig(4.6, 2.3)
    for key, lab, col, mk in (("LMPI", "LMPI", INK, "s"), ("BAND", "band ratio", AQUA, "D"), ("GapCNN", "GAP CNN", BLUE, "o"),
                              ("STFTCNN", "STFT CNN", VIOLET, "^")):
        ks = [k for k in rows[0]["pd"] if k == key or k.startswith(key + "_WHITE")]
        meas = [np.mean([r["pd"][k] for k in ks]) for r in rows]
        pred = [np.mean([r["pd_pred"][k] for k in ks]) for r in rows]
        ax.plot(f0, pred, color=col, lw=1.0, ls="-" if key != "STFTCNN" else ":")
        ax.plot(f0, meas, color=col, ls="none", marker=mk, ms=3.6, mfc="white", mew=1.0, label=lab)
    ax.axhline(0.05, color=INK2, lw=0.6)
    ax.set_xlabel("carrier offset (cycles/sample)"); ax.set_ylabel(r"$P_d$ at $-6$ dB"); ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right", fontsize=FS, handlelength=1.0, labelspacing=0.1)
    save_exact(fig, "fig_offband")


def fig_dynamics():
    """2026-10-05 (U7, supplement). GAP CNN training dynamics (mean of 3 seeds) for the WHITE, SURRXM and PAIRED
    recipes: (a) Spearman correlation with the LMPI statistic on -10 dB signal windows; (b) surrogate gap at 0 dB."""
    D_ = json.load(open(os.path.join(ROOT, "RESULTS", "dynamics.json")))["runs"]
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.2), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.05)
    for rec, lab, col, mk in (("WHITE", "White", BLUE, "o"), ("SURRXM", r"Surr$^{\ast}_{\rm m}$", ORANGE, "s"), ("PAIRED", "Paired", VIOLET, "^")):
        H = [D_[f"GapCNN_{rec}_s{s}"] for s in range(3) if f"GapCNN_{rec}_s{s}" in D_]
        if not H:
            continue
        st = [max(h["step"], 1) for h in H[0]]
        for ax, key, scale in ((axes[0], "rho_lmpi", 1.0), (axes[1], "gap_0", 100.0)):
            v = np.mean([[scale * h[key] for h in hh] for hh in H], 0)
            ax.plot(st, v, color=col, marker=mk, ms=2.8, lw=1.1, label=lab)
    for ax, yl, ttl in ((axes[0], r"$\rho$ with LMPI", "(a) agreement with the LMPI test"), (axes[1], "gap at 0 dB (points)", "(b) surrogate gap")):
        ax.set_xscale("log"); ax.set_xlabel("training step"); ax.set_ylabel(yl); ax.set_title(ttl, loc="left", pad=2, fontsize=FS)
    axes[0].legend(loc="lower right", fontsize=FS, handlelength=1.3, labelspacing=0.1)
    save_exact(fig, "fig_dynamics")


def fig_escape_panel(ax, legend_kw=None):
    """Panel (c) of the journal wall figure (U3): worst-case P_d of the cyclostationary test predicted by the escape law
    (no fitted parameter) against the measured rate: earlier wall tests (u = 1, 2 dB) and the new roll-off sweep."""
    EL = json.load(open(os.path.join(ROOT, "RESULTS", "escape_law.json")))
    B = [r for r in EL["B"] if not (r["source"] == "wall_test_escape" and r["u_db"] == 1.0)]
    ax.plot([0, 1], [0, 1], color=AXIS, lw=0.8)
    ax.plot([r["pd_pred"] for r in B], [r["pd_white"] for r in B], ls="none", marker="o", ms=2.6, color=ORD[1], label="wall tests")
    for beta, mk, col in ((0.15, "v", ORANGE), (0.35, "s", VIOLET), (0.6, "^", AQUA)):
        C = [r for r in EL.get("C", []) if r["beta"] == beta]
        ax.plot([r["pd_pred"] for r in C], [r["pd_white"] for r in C], ls="none", marker=mk, ms=2.8, mfc="white", mew=0.9,
                color=col, label=rf"$\beta$={beta}")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02); ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel(r"predicted $P_d$"); ax.set_ylabel(r"measured $P_d$")
    ax.set_title("(c) escape law", loc="left", pad=2, fontsize=FS)
    if legend_kw:
        ax.legend(ncol=2, **{**legend_kw, "handlelength": 0.8})
    else:
        ax.legend(loc="upper left", fontsize=FS, handlelength=0.8, labelspacing=0.05, borderaxespad=0.1, handletextpad=0.2)


def _load_df():
    from analysis import load_runs
    d = load_runs()
    # OFDM runs share sps = 8 and an empty tag; keep only the single-carrier set the text reports, at the N on the axis.
    return d[(d.tag == "") & (d.sps == 8) & (d.signal == "modulated") & d.N.isin([128, 512, 2048])]


def fig_nscaling():
    """(a) nominal-threshold Pf under the 0.8-Nyquist roll-off versus N; (b) worst-case Pd(-10 dB) over the receiver
    set with P_f <= 5 % enforced, versus N. Mean +- s.d. over seeds (3 for N > 128 and SURRX, 5 otherwise)."""
    df = _load_df()
    series = [("GapCNN", "WHITE", "CNN, white neg.", BLUE, "o", "-"), ("GapCNN", "SURRX", r"CNN, Surr$^{\ast}$", ORANGE, "s", "-"),
              ("GapCNN", "SURRXM", r"CNN, Surr$^{\ast}_{\rm m}$", ORANGE, "s", "--"), ("BAND", "-", "band ratio", AQUA, "D", ":")]
    fig, axes = plt.subplots(1, 2, figsize=(WC, 2.05), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.03)
    for ax, col, ttl in [(axes[0], "Pf_lpf0.8", r"(a) $P_f$, LP .8"), (axes[1], "wcPd_U_receiver_-10", r"(b) worst $P_d$, $-10$ dB")]:
        for model, rec, lab, c, mk, ls in series:
            d = df[(df.model == model) & (df.recipe == rec)]
            Ns = sorted(d.N.unique())
            m = [d[d.N == n][col].mean() for n in Ns]
            s = [d[d.N == n][col].std(ddof=1) if (d.N == n).sum() > 1 else 0 for n in Ns]
            mfc = "white" if ls == "--" else c
            ax.errorbar(Ns, m, yerr=s, color=c, ls=ls, marker=mk, ms=3.0, mfc=mfc, mew=0.9, lw=1.1, capsize=1.5, elinewidth=0.6, label=lab)
        ax.set_xscale("log", base=2); ax.set_xticks([128, 512, 2048]); ax.set_xticklabels(["128", "512", "2048"]); ax.minorticks_off()
        ax.set_ylim(0, 1.02); ax.set_title(ttl, loc="left", pad=2, fontsize=FS); ax.set_xlabel(r"$N$")
        ax.axhline(0.05, color=INK2, lw=0.6)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=2, fontsize=FS, handlelength=1.6, columnspacing=0.9,
               handletextpad=0.4, labelspacing=0.15, frameon=False)
    save_exact(fig, "fig_nscaling")


def fig_masking():
    """Pd at -10 dB with the signal in LP .8 noise, per training recipe (GAP CNN, N = 128): at the nominal threshold
    (filled) and with a threshold calibrated on LP .8 noise itself (hollow); white-noise Pd for reference (grey bar)."""
    import pandas as pd
    A = pd.DataFrame(json.load(open(os.path.join(ROOT, "RESULTS", "alpha_sensitivity.json"))))
    A = A[(A.model == "GapCNN") & np.isclose(A.alpha, 0.05)]
    order = [("WHITE", r"White"), ("PINK", r"Pink"), ("PINKM", r"Pink$_{\rm m}$"), ("RAND", r"Rand"), ("RANDM", r"Rand$_{\rm m}$"),
             ("SURR", r"Surr"), ("SURRX", r"Surr$^{\ast}$"), ("SURRXM", r"Surr$^{\ast}_{\rm m}$")]
    fig, ax = new_fig(4.6, 2.6)      # supplement (one column, included at natural size)
    for i, (rec, lab) in enumerate(order):
        r = A[A.recipe == rec]
        if not len(r):
            continue
        r = r.iloc[0]
        y = len(order) - 1 - i
        ax.plot([0, r["Pd_-10"]], [y, y], color=GRID, lw=2.2, solid_capstyle="butt", zorder=1)
        ax.plot(r["PdIn_lpf0.8_-10"], y, "o", color=BLUE, ms=4.2, zorder=3, label="nominal threshold" if i == 0 else None)
        ax.plot(r["PdCal_lpf0.8_-10"], y, "o", color=ORANGE, mfc="white", mew=1.1, ms=4.2, zorder=3,
                label="threshold calibrated on LP .8 noise" if i == 0 else None)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=FS, handlelength=1.0, borderaxespad=0.2)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([lab for _, lab in order][::-1])
    ax.set_xlim(0, 1.0); ax.set_xlabel(r"$P_d$ at $-10$ dB, signal in LP .8 noise (grey bar: in white noise)")
    ax.grid(axis="y", visible=False)
    save_exact(fig, "fig_masking")


def fig_wall_theory():
    """J2. (a) worst-case Pd of the band ratio versus SNR - wall (u = 1 dB, 8 samples/symbol) for N = 2048 (long test),
    8192 and 32768: the transition sharpens onto the predicted wall. (b) constructed wall versus u for three
    occupancies (lines), energy-detector wall (grey dashed), and the measured 50 % crossing at N = 32768 (markers)."""
    J = json.load(open(os.path.join(ROOT, "RESULTS", "wall_vs_theory.json")))
    import pandas as pd
    S = pd.read_csv(os.path.join(ROOT, "TABLES", "wall_vs_theory.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.4), layout="constrained", gridspec_kw={"width_ratios": [1, 1.1]})   # supplement
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.04)
    ax = axes[0]
    for N, col in zip([8192, 32768], [ORD[1], ORD[2]]):
        v = J[f"sps8_u1.0_N{N}_s0"]["Pd"]
        ds = sorted(float(d) for d in v)
        ax.plot(ds, [v[str(d)]["worst"] for d in ds], color=col, marker="o", ms=2.8, lw=1.1, label=f"$N$={N}")
    longd = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(ROOT, "RESULTS", "wall_test_long_s*.json")))]
    if longd:
        ds = [-2, 0, 2]
        m = [np.mean([d_["N2048_R1.0"]["Pd"][str(d)]["BAND"]["worst"] for d_ in longd]) for d in ds]
        ax.plot(ds, m, color=ORD[0], marker="o", ms=2.8, lw=1.1, label="$N$=2048")
    ax.axhline(0.05, color=INK2, lw=0.6); ax.axvline(0, color=INK2, lw=0.6)
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"SNR $-$ wall (dB)"); ax.set_ylabel(r"worst-case $P_d$ (band ratio)")
    ax.set_title(r"(a) $u=1$ dB, 8 samples/symbol", loc="left", pad=2, fontsize=FS)
    h, l = ax.get_legend_handles_labels()
    order_ = sorted(range(len(l)), key=lambda i: int(l[i].split("=")[1].strip("$")))
    ax.legend([h[i] for i in order_], [l[i] for i in order_], loc="upper left", fontsize=FS, handlelength=1.3, labelspacing=0.12)
    ax = axes[1]
    for sps, col, mk in [(4, BLUE, "o"), (8, ORANGE, "s"), (16, VIOLET, "^")]:
        d = S[(S.sps == sps) & (S.N == 32768)].sort_values("u_db")
        rho = d.rho.iloc[0]
        ax.plot(d.u_db, d.wall_db, color=col, lw=1.1, label=rf"$\rho$={rho:.2f}")
        ax.plot(d.u_db, d.wall_db + d.delta50_db, ls="none", marker=mk, ms=3.2, mfc="white", mec=col, mew=0.9)
    d = S[(S.sps == 8) & (S.N == 32768)].sort_values("u_db")
    ax.plot(d.u_db, d.ed_wall_db, color=INK2, ls="--", lw=1.0, label="energy det.")
    ax.set_xlabel(r"uncertainty $u$ (dB)"); ax.set_ylabel("SNR (dB)")
    ax.set_title("(b) wall (lines), measured 50 % (markers)", loc="left", pad=2, fontsize=FS)
    ax.legend(loc="lower right", fontsize=FS, handlelength=1.3, labelspacing=0.12)
    save_exact(fig, "fig_wall_theory")


if __name__ == "__main__":
    fig_theory()
    fig_surrogate()
    fig_wall()
    fig_wall_journal()
    fig_real_noise()
    fig_nscaling()
    fig_masking()
    fig_wall_theory()
    fig_templates()
    fig_offband()
    if os.path.exists(os.path.join(ROOT, "RESULTS", "dynamics.json")):
        fig_dynamics()
    print("figures written to", FIG)
