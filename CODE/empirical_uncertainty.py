"""How large is the noise-shape uncertainty of a real receiver, and where would its wall be? (Task T7 / W2 of the internal review)

From recorded RTL-SDR receiver noise across two hardware tuners:
  Receiver 1: Nooelec NESDR SMArt v5 (R820T2 tuner, DATA/real_noise_hf, 131 clean live recordings across 7 frequencies)
  Receiver 2: RTL-SDR Blog V4 (R828D tuner, DATA/trevtron/noise, 74 clean live recordings at 145 MHz)

We estimate smooth PSD shapes (128-point DC-blocked periodograms, 16-term cosine/sine log-PSD fit, unit mean)
and quantify peak-to-peak shape uncertainty u (dB) across five operational scenarios:
  1. Split-half estimation floor (disjoint halves of the same stationary block)
  2. Steady-state in-session drift (consecutive 1024-window blocks post-transient: windows 1024..2048 vs 2048..3072)
  3. Start-up power transient (initial 32 ms / first 256 windows vs steady-state reference windows 256..1024)
  4. In-session / same-frequency variation (between independent recordings at the same center frequency)
  5. Retuning cross-frequency variation (between recordings at different tuning frequencies)
  6. Cross-receiver transfer variation (Receiver 1 vs Receiver 2 at the shared 145 MHz frequency)

For every scenario, we report median, 90th percentile, max, and non-parametric bootstrap 95% CIs (B=1000).
We also compute the empirical SNR wall: least-squares fit of (c0 + g(c0 - p) ≈ c1) over g >= 0.
A shape pair fakes an SNR g only if the relative residual <= estimation noise floor epsilon.

Outputs:
  - RESULTS/empirical_uncertainty.json
  - FIGURES/fig_empirical_uncertainty.pdf (+ .png)
"""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_noise_eval as RN, wall as W

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 128
f = np.fft.fftfreq(N)
A = np.stack([np.cos(2 * np.pi * k * f) for k in range(16)] + [np.sin(2 * np.pi * k * f) for k in range(1, 16)], 1)
use = np.abs(f) > 0.5 / N

# Visualization constants matching the project palette
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#9a998f"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.0, "axes.labelsize": 8.0, "axes.titlesize": 8.0,
    "xtick.labelsize": 8.0, "ytick.labelsize": 8.0, "legend.fontsize": 8.0, "axes.edgecolor": AXIS,
    "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-", "legend.frameon": False,
    "lines.linewidth": 1.3, "lines.markersize": 3.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300, "figure.dpi": 150,
})


def fit_shape(windows):
    P = np.mean(np.abs(np.fft.fft(windows, axis=1)) ** 2, axis=0)
    coef, *_ = np.linalg.lstsq(A[use], np.log(np.maximum(P[use], 1e-12)), rcond=None)
    s = np.exp(A @ coef)
    return s / s.mean()


def u_db(a, b):
    r = a / b
    return float(10 * np.log10(r[use].max() / r[use].min()))


def bootstrap_ci(arr, stat_fn, B=1000, seed=42):
    arr = np.asarray(arr)
    if len(arr) == 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    boot = [float(stat_fn(rng.choice(arr, size=len(arr), replace=True))) for _ in range(B)]
    return [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]


def summarise_with_ci(x, p_signal, B=1000):
    x = np.asarray(x, dtype=float)
    med = float(np.median(x))
    p90 = float(np.quantile(x, 0.9))
    mx = float(np.max(x))
    med_ci = bootstrap_ci(x, np.median, B=B)
    p90_ci = bootstrap_ci(x, lambda a: np.quantile(a, 0.9), B=B)
    wall_p90 = float(10 * np.log10(W.wall(p_signal, p90))) if p90 > 0 else None
    wall_max = float(10 * np.log10(W.wall(p_signal, mx))) if mx > 0 else None
    return {
        "median": med, "median_ci": med_ci,
        "p90": p90, "p90_ci": p90_ci,
        "max": mx, "n": int(len(x)),
        "wall_db_at_p90": wall_p90,
        "wall_db_at_max": wall_max,
    }


def main():
    p_signal = W.signal_psd(N, n=8000)

    # -------------------------------------------------------------------------
    # 1. Receiver 1 (HF composite, R820T2 tuner)
    # -------------------------------------------------------------------------
    v1_file = os.path.join(ROOT, "RESULTS", "real_noise_v3_N128.json")
    if not os.path.exists(v1_file):
        v1_file = os.path.join(ROOT, "RESULTS", "real_noise_v2_N128.json")
    V1 = json.load(open(v1_file))["recordings"]
    recs1 = {k: v for k, v in V1.items() if v["group"] == "live_rtl" and not v["shape"]["narrowband"]}

    S1, drift1, split1, steady1, head1 = {}, [], [], [], []
    for name, v in recs1.items():
        fp = os.path.join(ROOT, "DATA", "real_noise_hf", name)
        iq, _ = RN.load(fp)
        w = RN.windows(iq, N)
        kbig = 1024 if len(w) >= 2048 else len(w) // 2
        ref, test = w[:kbig], w[kbig: kbig + 2000]
        drift1.append(u_db(fit_shape(test), fit_shape(ref)))
        split1.append(u_db(fit_shape(test[0::2]), fit_shape(test[1::2])))
        if len(w) >= 3072:
            steady1.append(u_db(fit_shape(w[2048:3072]), fit_shape(w[1024:2048])))
            head1.append(u_db(fit_shape(w[256:1024]), fit_shape(w[:256])))
        S1[name] = (v["meta"]["center_freq"], fit_shape(w[1024:3024] if len(w) >= 3024 else w))

    freqs1 = sorted({cf for cf, _ in S1.values()})
    session1, retune1 = [], []
    names1 = list(S1)
    for i in range(len(names1)):
        for j in range(i + 1, len(names1)):
            (fa, sa), (fb, sb) = S1[names1[i]], S1[names1[j]]
            (session1 if fa == fb else retune1).append(u_db(sa, sb))

    # Estimation noise residual floor for Receiver 1
    noise_rel1 = []
    for name in list(recs1.keys())[:40]:
        fp = os.path.join(ROOT, "DATA", "real_noise_hf", name)
        iq, _ = RN.load(fp)
        w = RN.windows(iq, N)[1024:3024]
        a, b = fit_shape(w[0::2]), fit_shape(w[1::2])
        noise_rel1.append(float(np.linalg.norm((a - b)[use]) / np.linalg.norm(b[use])))
    floor1 = float(np.quantile(noise_rel1, 0.9))

    # Empirical wall least-squares fit for Receiver 1
    shapes1 = [(cf, s) for cf, s in S1.values()]
    best1 = {"same_freq": [], "retune": []}
    for i in range(len(shapes1)):
        for j in range(len(shapes1)):
            if i == j:
                continue
            (fa, c0), (fb, c1) = shapes1[i], shapes1[j]
            d = (c0 - p_signal)[use]
            g = max(0.0, float(np.dot(d, (c1 - c0)[use]) / np.dot(d, d)))
            rel = float(np.linalg.norm((c0 + g * (c0 - p_signal) - c1)[use]) / np.linalg.norm(c1[use]))
            best1["same_freq" if fa == fb else "retune"].append((g, rel))

    emp1 = {}
    for k, lst in best1.items():
        ok = [g for g, rel in lst if rel <= floor1]
        emp1[k] = {
            "pairs": len(lst),
            "pairs_within_noise": len(ok),
            "max_g_within_noise_db": float(10 * np.log10(max(ok))) if ok and max(ok) > 0 else None,
            "median_rel_residual": float(np.median([rel for _, rel in lst])),
            "min_rel_residual": float(np.min([rel for _, rel in lst])),
        }

    # -------------------------------------------------------------------------
    # 2. Receiver 2 (TrevTron rtl-ml-dataset v2, R828D tuner)
    # -------------------------------------------------------------------------
    v2_file = os.path.join(ROOT, "RESULTS", "real_noise_trevtron_N128.json")
    recs2, S2, split2, steady2, session2 = {}, {}, [], [], []
    floor2 = 0.025
    emp2 = {}
    if os.path.exists(v2_file):
        V2 = json.load(open(v2_file))["recordings"]
        recs2 = {k: v for k, v in V2.items() if v["group"] == "live_rtl" and not v["shape"]["narrowband"]}
        noise_rel2 = []
        for name in list(recs2.keys()):
            fp = os.path.join(ROOT, "DATA", "trevtron", "noise", name)
            iq, _ = RN.load(fp)
            w = RN.windows(iq, N)
            split2.append(u_db(fit_shape(w[1024:3024:2]), fit_shape(w[1025:3024:2])))
            steady2.append(u_db(fit_shape(w[2048:3072]), fit_shape(w[1024:2048])))
            s = fit_shape(w[1024:3024])
            S2[name] = s
            a, b = fit_shape(w[1024:3024:2]), fit_shape(w[1025:3024:2])
            noise_rel2.append(float(np.linalg.norm((a - b)[use]) / np.linalg.norm(b[use])))

        floor2 = float(np.quantile(noise_rel2, 0.9))
        names2 = list(S2)
        for i in range(len(names2)):
            for j in range(i + 1, len(names2)):
                session2.append(u_db(S2[names2[i]], S2[names2[j]]))

        best2_same = []
        for i in range(len(names2)):
            for j in range(len(names2)):
                if i == j:
                    continue
                c0, c1 = S2[names2[i]], S2[names2[j]]
                d = (c0 - p_signal)[use]
                g = max(0.0, float(np.dot(d, (c1 - c0)[use]) / np.dot(d, d)))
                rel = float(np.linalg.norm((c0 + g * (c0 - p_signal) - c1)[use]) / np.linalg.norm(c1[use]))
                best2_same.append((g, rel))

        ok2 = [g for g, rel in best2_same if rel <= floor2]
        emp2["same_freq"] = {
            "pairs": len(best2_same),
            "pairs_within_noise": len(ok2),
            "max_g_within_noise_db": float(10 * np.log10(max(ok2))) if ok2 and max(ok2) > 0 else None,
            "median_rel_residual": float(np.median([rel for _, rel in best2_same])),
            "min_rel_residual": float(np.min([rel for _, rel in best2_same])),
        }

    # -------------------------------------------------------------------------
    # 3. Cross-Receiver comparison at shared center frequency 145 MHz
    # -------------------------------------------------------------------------
    cross_u = []
    cross_best = []
    if S2:
        recs1_145 = [k for k, v in recs1.items() if v["meta"]["center_freq"] == "145000000.0"]
        for k1 in recs1_145:
            s1 = S1[k1][1]
            for k2, s2 in S2.items():
                cross_u.append(u_db(s1, s2))
                d = (s1 - p_signal)[use]
                g = max(0.0, float(np.dot(d, (s2 - s1)[use]) / np.dot(d, d)))
                rel = float(np.linalg.norm((s1 + g * (s1 - p_signal) - s2)[use]) / np.linalg.norm(s2[use]))
                cross_best.append((g, rel))

    emp_cross = {
        "pairs": len(cross_best),
        "median_rel_residual": float(np.median([rel for _, rel in cross_best])) if cross_best else None,
        "min_rel_residual": float(np.min([rel for _, rel in cross_best])) if cross_best else None,
    }

    # -------------------------------------------------------------------------
    # Assemble comprehensive results dictionary
    # -------------------------------------------------------------------------
    out = {
        "receiver_1_r820t2": {
            "n_recordings": len(recs1),
            "n_frequencies": len(freqs1),
            "split_half_noise": summarise_with_ci(split1, p_signal),
            "steady_drift": summarise_with_ci(steady1, p_signal),
            "startup_transient": summarise_with_ci(head1, p_signal),
            "session": summarise_with_ci(session1, p_signal),
            "retune": summarise_with_ci(retune1, p_signal),
            "residual_floor_rel": floor1,
            "empirical_wall": emp1,
        },
        "receiver_2_r828d": {
            "n_recordings": len(recs2),
            "split_half_noise": summarise_with_ci(split2, p_signal) if split2 else None,
            "steady_drift": summarise_with_ci(steady2, p_signal) if steady2 else None,
            "session": summarise_with_ci(session2, p_signal) if session2 else None,
            "residual_floor_rel": floor2,
            "empirical_wall": emp2,
        },
        "cross_receiver_145mhz": {
            "n_pairs": len(cross_u),
            "uncertainty": summarise_with_ci(cross_u, p_signal) if cross_u else None,
            "empirical_wall": emp_cross,
        },
        # Flat legacy keys for backward compatibility
        "empirical_wall": {"residual_floor_rel": floor1, **emp1},
        "split_half_noise": summarise_with_ci(split1, p_signal),
        "drift": summarise_with_ci(drift1, p_signal),
        "drift_steady": summarise_with_ci(steady1, p_signal),
        "head_vs_rest": summarise_with_ci(head1, p_signal),
        "session": summarise_with_ci(session1, p_signal),
        "retune": summarise_with_ci(retune1, p_signal),
        "n_recordings": len(recs1),
        "n_frequencies": len(freqs1),
    }

    out_path = os.path.join(ROOT, "RESULTS", "empirical_uncertainty.json")
    json.dump(out, open(out_path, "w"), indent=1)
    print("Saved results to", out_path)

    # -------------------------------------------------------------------------
    # Generate publication figure: fig_empirical_uncertainty.pdf
    # -------------------------------------------------------------------------
    # 2026-10-05: drawn at its printed size (supplement, natural size) with all text >= 8 pt; saved without cropping
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.1), layout="constrained", gridspec_kw={"width_ratios": [1.3, 1.0]})
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.04)

    # Subplot (a): Uncertainty distributions by scenario
    data_groups = [
        ("Split-half\n(floor)", split1, BLUE),
        ("Steady drift\n(in-session)", steady1, AQUA),
        ("Start-up\n(32 ms)", head1, ORANGE),
        ("Same freq.\n(Rec 1)", session1, BLUE),
        ("Same freq.\n(Rec 2)", session2, VIOLET),
        ("Retune\n(Rec 1)", retune1, ORANGE),
        ("Cross-rec.\n(145 MHz)", cross_u, INK2),
    ]

    positions = np.arange(len(data_groups))
    for i, (label, vals, col) in enumerate(data_groups):
        if not vals:
            continue
        vals = np.asarray(vals)
        q25, med, q75 = np.percentile(vals, [25, 50, 75])
        p10, p90 = np.percentile(vals, [10, 90])
        ax1.plot([i, i], [p10, p90], color=col, lw=1.2, alpha=0.7)
        ax1.plot([i, i], [q25, q75], color=col, lw=3.0, solid_capstyle="round")
        ax1.plot(i, med, "o", color="white", markeredgecolor=col, markeredgewidth=1.2, markersize=4.5)

    ax1.set_xticks(positions)
    ax1.set_xticklabels([d[0].replace("\n", " ") for d in data_groups], fontsize=8.0, rotation=30, ha="right",
                        rotation_mode="anchor")   # one line, rotated: two-line labels overlap at 8 pt
    ax1.set_ylabel("Shape uncertainty $u$ (dB)")
    ax1.set_title("(a) Empirical shape uncertainty by scenario", fontsize=8.0, pad=4)
    ax1.set_ylim(-0.2, 8.2)

    # Subplot (b): Relative residual vs SNR faked
    # Plot sample scatter of best fits
    rng = np.random.default_rng(123)
    same_sample = [best1["same_freq"][k] for k in rng.choice(len(best1["same_freq"]), size=min(400, len(best1["same_freq"])), replace=False)]
    retune_sample = [best1["retune"][k] for k in rng.choice(len(best1["retune"]), size=min(400, len(best1["retune"])), replace=False)]

    g_same = [10 * np.log10(max(1e-4, g)) for g, rel in same_sample]
    r_same = [100 * rel for g, rel in same_sample]
    g_ret = [10 * np.log10(max(1e-4, g)) for g, rel in retune_sample]
    r_ret = [100 * rel for g, rel in retune_sample]

    ax2.scatter(g_same, r_same, color=BLUE, alpha=0.45, s=10, label="Same freq. pairs (Rec 1)", edgecolors="none")
    ax2.scatter(g_ret, r_ret, color=ORANGE, alpha=0.45, s=10, label="Retuning pairs (Rec 1)", edgecolors="none")
    if cross_best:
        cross_sample = [cross_best[k] for k in rng.choice(len(cross_best), size=min(300, len(cross_best)), replace=False)]
        g_cr = [10 * np.log10(max(1e-4, g)) for g, rel in cross_sample]
        r_cr = [100 * rel for g, rel in cross_sample]
        ax2.scatter(g_cr, r_cr, color=VIOLET, alpha=0.45, s=10, label="Cross-receiver (145 MHz)", edgecolors="none")

    ax2.axhline(100 * floor1, color=INK2, ls="--", lw=1.0, label=f"Estimation noise floor ({100*floor1:.1f}%)")
    wall_same = emp1["same_freq"]["max_g_within_noise_db"]
    if wall_same is not None:
        ax2.axvline(wall_same, color=BLUE, ls=":", lw=1.1, label=f"Empirical wall ({wall_same:.1f} dB)")

    ax2.set_xlabel(r"Fitted signal SNR $\gamma$ (dB)")
    ax2.set_ylabel("Relative residual (%)")
    ax2.set_title("(b) Can real shape shifts fake a signal?", fontsize=8.0, pad=4)
    ax2.set_xlim(-40, 5)
    ax2.set_ylim(0, 30)
    # Legend placed below the axes; inside the panel it covers the retuning cluster and the wall marker.
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, fontsize=8.0, handlelength=1.5,
               labelspacing=0.2, borderaxespad=0.0, markerscale=1.4)

    fig_dir = os.path.join(ROOT, "FIGURES")
    os.makedirs(fig_dir, exist_ok=True)
    pdf_path = os.path.join(fig_dir, "fig_empirical_uncertainty.pdf")
    png_path = os.path.join(fig_dir, "fig_empirical_uncertainty.png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    print("Saved figure to", pdf_path, "and", png_path)

    # Print summary
    print("\n=== Summary of Empirical Shape Uncertainty ===")
    print(f"Receiver 1 (R820T2, n={len(recs1)}):")
    print(f"  Split-half floor: median {out['receiver_1_r820t2']['split_half_noise']['median']:.2f} dB, p90 {out['receiver_1_r820t2']['split_half_noise']['p90']:.2f} dB")
    print(f"  Steady drift:     median {out['receiver_1_r820t2']['steady_drift']['median']:.2f} dB, p90 {out['receiver_1_r820t2']['steady_drift']['p90']:.2f} dB")
    print(f"  Startup transient:median {out['receiver_1_r820t2']['startup_transient']['median']:.2f} dB, max {out['receiver_1_r820t2']['startup_transient']['max']:.2f} dB")
    print(f"  Same-freq session:median {out['receiver_1_r820t2']['session']['median']:.2f} dB, p90 {out['receiver_1_r820t2']['session']['p90']:.2f} dB")
    print(f"  Retuning:         median {out['receiver_1_r820t2']['retune']['median']:.2f} dB, p90 {out['receiver_1_r820t2']['retune']['p90']:.2f} dB")
    print(f"  Empirical wall (same-freq): {emp1['same_freq']['max_g_within_noise_db']:.2f} dB")
    print(f"  Retuning pairs within noise: {emp1['retune']['pairs_within_noise']} (min residual {100*emp1['retune']['min_rel_residual']:.1f}%)")

    if recs2:
        print(f"\nReceiver 2 (R828D, n={len(recs2)}):")
        print(f"  Split-half floor: median {out['receiver_2_r828d']['split_half_noise']['median']:.2f} dB, p90 {out['receiver_2_r828d']['split_half_noise']['p90']:.2f} dB")
        print(f"  Steady drift:     median {out['receiver_2_r828d']['steady_drift']['median']:.2f} dB, p90 {out['receiver_2_r828d']['steady_drift']['p90']:.2f} dB")
        print(f"  Same-freq session:median {out['receiver_2_r828d']['session']['median']:.2f} dB, p90 {out['receiver_2_r828d']['session']['p90']:.2f} dB")

    if cross_u:
        print(f"\nCross-Receiver at 145 MHz (n={len(cross_u)} pairs):")
        print(f"  Uncertainty u:    median {out['cross_receiver_145mhz']['uncertainty']['median']:.2f} dB, p90 {out['cross_receiver_145mhz']['uncertainty']['p90']:.2f} dB")
        print(f"  Residual:         median {100*emp_cross['median_rel_residual']:.1f}%, min {100*emp_cross['min_rel_residual']:.1f}%")


if __name__ == "__main__":
    main()
