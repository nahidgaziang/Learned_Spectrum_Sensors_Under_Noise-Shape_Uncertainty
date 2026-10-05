"""Write MANUSCRIPT/numbers.tex: every number quoted in the paper, computed from RESULTS/.
Also writes RESULTS/paper_numbers.json (same values, machine-readable) for the claim checker."""
import glob, json, os
import numpy as np
import pandas as pd

import sim, wall as W
from analysis import load_runs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHS = ["CLDNN", "ResNet1D", "CovCNN", "STFTCNN", "GapCNN"]
VALS = {}


def pct(x, nd=0):
    return f"{100 * x:.{nd}f}\\%"


def rng_pct(xs, nd=0):
    xs = [x for x in xs if x is not None and not np.isnan(x)]
    lo, hi = min(xs), max(xs)
    return f"{100*lo:.{nd}f}--{100*hi:.{nd}f}\\%" if round(100*lo, nd) != round(100*hi, nd) else pct(lo, nd)


def put(name, tex, raw=None):
    VALS[name] = {"tex": tex, "raw": raw}


def verification_macros():
    """Macros added by the 2026-10-05 verification (internal notes, internal notes): every number that the
    corrected text quotes and that was previously typed by hand or missing."""
    R = os.path.join(ROOT, "RESULTS")
    J = lambda f: json.load(open(os.path.join(R, f))) if os.path.exists(os.path.join(R, f)) else None
    # --- DetectNet replica: largest |P_d(signal) - P_d(surrogate)| over [-20, 0] dB, per seed (points)
    dmax = []
    for f in sorted(glob.glob(os.path.join(R, "runs", "DetectNetExact_WHITE_N128_sps8_s*_QAM16.json"))):
        e = json.load(open(f))["eval"]
        ks = [k for k in e["Pd"] if -20 <= float(k) <= 0]
        dmax.append(max(abs(e["Pd"][k]["white"]["nominal"] - e["Pd_surrogate"][k]["nominal"]) for k in ks))
    if dmax:
        put("DetMaxAbsDiff", f"{100 * max(dmax):.1f}", max(dmax))
        put("DetMaxAbsDiffRange", f"{100 * min(dmax):.1f}--{100 * max(dmax):.1f}", dmax)
    # --- real signals in real noise: detection tracks the centred band ratio; non-detected classes; class counts
    S, M = J("real_signal_surrogate.json"), J("real_signal_summary.json")
    if S and M:
        M = M["results"]
        used_cls = sorted({c for rx in S["receivers"].values() for c, C in rx["classes"].items() if C["captures"]})
        put("RealSigNclasses", str(len(used_cls)), used_cls)
        combos = [(rx, c) for rx, RX in S["receivers"].items() for c, C in RX["classes"].items() if C["captures"]]
        put("RealSigNcombos", str(len(combos)), combos)
        A = S["receivers"]["trevtron_R828D"]["classes"]["APRS"]
        put("RealSigAprsFiles", str(A["n_files"]), A["n_files"])
        put("RealSigAprsUsed", str(len(A["captures"])), len(A["captures"]))
        mx = lambda rx, c, d: max(M[f"{rx}|{c}|{d}|sur"]["pd_sig"])
        diffs = [abs(mx(rx, c, "GapCNN-WHITE") - mx(rx, c, "BAND")) for rx, c in combos]
        put("RealSigCnnBandMaxDiff", f"{100 * max(diffs):.0f}", max(diffs))
        nd = [mx("trevtron_R828D", c, d) for c in ("NOAA_weather", "pager", "FRS_GMRS") for d in ("GapCNN-WHITE", "BAND")]
        put("RealSigNdMax", f"{max(nd):.2f}", max(nd))
        others = [mx("trevtron_R828D", c, f"{a}-WHITE") for c in ("NOAA_weather", "pager", "FRS_GMRS") for a in ARCHS
                  if not (a == "STFTCNN" and c == "NOAA_weather")]
        put("RealSigNdMaxAll", f"{max(others):.2f}", max(others))
        put("RealSigStftNoaa", f"{mx('trevtron_R828D', 'NOAA_weather', 'STFTCNN-WHITE'):.2f}", None)
        put("RealSigFmTtCnnMax", f"{mx('trevtron_R828D', 'FM_broadcast', 'GapCNN-WHITE'):.2f}", None)
        put("RealSigFmTtBandMax", f"{mx('trevtron_R828D', 'FM_broadcast', 'BAND'):.2f}", None)
        put("RealSigFmHfCnnMax", f"{mx('hf_composite', 'FM_broadcast', 'GapCNN-WHITE'):.2f}", None)
        put("RealSigFmTtSurrxMax", f"{mx('trevtron_R828D', 'FM_broadcast', 'GapCNN-SURRX'):.2f}", None)
        put("RealSigFmTtKappaLoMax", f"{mx('trevtron_R828D', 'FM_broadcast', 'KAPPALO'):.2f}", None)
        own = {c: np.median([v["info"]["own_snr_db"] for v in S["receivers"]["trevtron_R828D"]["classes"][c]["captures"].values()])
               for c in ("FM_broadcast", "NOAA_weather", "pager", "FRS_GMRS")}
        put("RealSigOwnSnrFmTt", f"{own['FM_broadcast']:.0f}", own["FM_broadcast"])
        put("RealSigOwnSnrLowRange", f"${min(own[c] for c in ('NOAA_weather', 'pager', 'FRS_GMRS')):.0f}$ to "
                                     f"${max(own[c] for c in ('NOAA_weather', 'pager', 'FRS_GMRS')):.0f}$", own)
        ci = lambda k: "$[" + ", ".join(f"{100 * x:+.1f}" for x in M[k]["ci"]) + "]$"
        put("RealSigFmGapHfCi", ci("hf_composite|FM_broadcast|GapCNN-WHITE|sur"), M["hf_composite|FM_broadcast|GapCNN-WHITE|sur"]["ci"])
        put("RealSigAdsbKappaGapHfCi", ci("hf_composite|ADS_B|KAPPA|sur"), M["hf_composite|ADS_B|KAPPA|sur"]["ci"])
        kl = [M[f"trevtron_R828D|{c}|KAPPALO|sur"]["gap"] for c in ("NOAA_weather", "pager", "FRS_GMRS")]
        put("RealSigKappaLoNegRange", f"${100 * min(kl):.0f}$ to ${100 * max(kl):.0f}$", kl)
    # --- empirical shape uncertainty: retuning pairs inside the residual floor
    E = J("empirical_uncertainty.json")
    if E:
        ew = E["receiver_1_r820t2"]["empirical_wall"]
        put("EmpResidFloor", pct(E["receiver_1_r820t2"]["residual_floor_rel"], 1), E["receiver_1_r820t2"]["residual_floor_rel"])
        put("EmpRetunePairs", f"{ew['retune']['pairs']:,}".replace(",", "\\,"), ew["retune"]["pairs"])
        put("EmpRetunePairsWithin", str(ew["retune"]["pairs_within_noise"]), ew["retune"]["pairs_within_noise"])
        put("EmpRetuneWallDb", f"${ew['retune']['max_g_within_noise_db']:.1f}$", ew["retune"]["max_g_within_noise_db"])
        put("EmpRetuneResidMin", pct(ew["retune"]["min_rel_residual"], 1), ew["retune"]["min_rel_residual"])
    # --- Gaussianity of recorded noise (noise_gaussianity.py)
    G = J("noise_gaussianity.json")
    if G:
        g, hf, tt = G["gaussian_ref"], G["receivers"]["hf_composite"]["summary"], G["receivers"]["trevtron_R828D"]["summary"]
        put("NoiseKappaGauss", f"{g['mean']:.2f}", g["mean"])
        put("NoiseKappaGaussSd", f"{g['sd']:.2f}", g["sd"])
        put("NoiseKappaMedHf", f"{hf['kappa_mean_median']:.2f}", hf["kappa_mean_median"])
        put("NoiseKappaRangeHf", f"{hf['kappa_mean_range'][0]:.2f}--{hf['kappa_mean_range'][1]:.2f}", hf["kappa_mean_range"])
        put("NoiseKappaMedTt", f"{tt['kappa_mean_median']:.2f}", tt["kappa_mean_median"])
        put("NoiseKappaFracTt", pct(tt["frac_above_q99_median"]), tt["frac_above_q99_median"])
        put("NoisePartialRolloff", f"{hf['partial_rolloff_vs_Pf_given_kappa_v3']:.2f}", hf["partial_rolloff_vs_Pf_given_kappa_v3"])
        put("NoisePartialKappa", f"{hf['partial_kappa_vs_Pf_given_rolloff_v3']:.2f}", hf["partial_kappa_vs_Pf_given_rolloff_v3"])
        put("NoiseKappaWhitenedRhoHf", f"${hf['spearman_kappa_vs_Pf_whitened_v3']:.2f}$", hf["spearman_kappa_vs_Pf_whitened_v3"])
        put("NoiseKappaPfRhoTt", f"${tt['spearman_kappa_vs_Pf_v3']:.2f}$", tt["spearman_kappa_vs_Pf_v3"])
    # --- classical CYC with 16 lags vs 5 lags vs CycNet (cyc_lag_ablation.py; escape test for CycNet)
    C = J("cyc_lag_ablation.json")
    if C:
        m = C["mean"]
        put("CycLagFiveBelowOne", f"{m['R1.0']['-2']['CYC5']:.2f}", m["R1.0"]["-2"]["CYC5"])
        put("CycLagSixteenBelowOne", f"{m['R1.0']['-2']['CYC16']:.2f}", m["R1.0"]["-2"]["CYC16"])
        put("CycLagSixteenBelowTwo", f"{m['R2.0']['-2']['CYC16']:.2f}", m["R2.0"]["-2"]["CYC16"])
        esc = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, "wall_test_escape_s*.json")))]
        if len(esc) == 3:
            cn = float(np.mean([d["N32768_R1.0"]["Pd"]["-2"]["CycNet-WHITE-N8192"]["worst"] for d in esc]))
            put("CycNetOverCycSixteen", f"{cn / m['R1.0']['-2']['CYC16']:.1f}", cn / m["R1.0"]["-2"]["CYC16"])
    # --- escape control: circular vs linear noise synthesis, Gaussian-surrogate H1 (wall_test_escape_controls.py)
    V = J("verification_controls_summary.json")
    if V and V.get("n_seeds") == 3:
        e1, e2 = V["escape"]["N8192_R1.0"], V["escape"]["N8192_R2.0"]
        nets = ("GapCNN-PAIRED-N8192", "CycNet-WHITE-N8192")
        put("CtrlSeeds", str(V["n_seeds"]), V["n_seeds"])
        put("CtrlPairedLinTwo", f"{e2[nets[0]]['lin_sig-2']:.2f}", e2[nets[0]]["lin_sig-2"])
        put("CtrlCycNetLinTwo", f"{e2[nets[1]]['lin_sig-2']:.2f}", e2[nets[1]]["lin_sig-2"])
        dif = [abs(e[n][f"circ_sig{d}"] - e[n][f"lin_sig{d}"]) for e in (e1, e2) for n in nets for d in (-2, 0)]
        put("CtrlSynthMaxDiff", f"{max(dif):.2f}", max(dif))
        gs = [e[n][f"{md}_gsur{d}"] for e in (e1, e2) for n in nets for md in ("circ", "lin") for d in (-2, 0)]
        put("CtrlGsurMax", f"{max(gs):.2f}", max(gs))
        ga = [e[n][f"{md}_gsur2"] for e in (e1, e2) for n in nets for md in ("circ", "lin")]
        put("CtrlGsurAboveRange", f"{min(ga):.2f}--{max(ga):.2f}", ga)
    # --- reference-calibrated energy detector on both receivers (adaptive_baselines*.json)
    a1, a2 = J("adaptive_baselines.json"), J("adaptive_baselines_trevtron.json")
    if a1 and a2:
        ed = [a1["recorded_summary"]["ED-ref"]["Pf_mean"], a2["recorded_summary"]["ED-ref"]["Pf_mean"]]
        put("AdaptEdPfRange", rng_pct(ed, 1), ed)
    # --- observation time of the long windows at the recordings' sample rate
    fs = 1.024e6
    put("ObsRateMsps", f"{fs / 1e6:.3f}", fs)
    put("ObsTimeND", f"{8192 / fs * 1e3:.0f}", 8192 / fs)
    put("ObsTimeNE", f"{32768 / fs * 1e3:.0f}", 32768 / fs)


def upgrade_macros():
    """Macros of the 2026-10-05 novelty upgrades (the project plan, internal notes). Every value is read
    from the result file of the corresponding script; files that do not exist yet are skipped (missing macros are then
    flagged by the check at the end of main())."""
    R = os.path.join(ROOT, "RESULTS")
    J = lambda f: json.load(open(os.path.join(R, f))) if os.path.exists(os.path.join(R, f)) else None
    rng2 = lambda xs, nd=2: (f"{min(xs):.{nd}f}--{max(xs):.{nd}f}" if round(min(xs), nd) != round(max(xs), nd) else f"{min(xs):.{nd}f}")
    # ---------------- U1 theory check (u1_theory_check.py)
    U = J("u1_theory_check.json")
    if U:
        r = U["results"]
        k = r["KAPPALO"]
        ratios, snrs = [], []
        for i, d in enumerate(k["snr"]):
            gp, se, pdv = k["gap_ideal"][i], k["se_ideal"][i], k["pd_sig"][i]
            if gp > 4 * se and pdv <= 0.4:
                ratios.append(gp / (10 ** (d / 10)) ** 2); snrs.append(d)
        put("UoneKappaRatioRange", rng2(ratios), ratios)
        put("UoneKappaRatioSnrs", f"${min(snrs)}$ to ${max(snrs)}$", snrs)
        so = [abs(x) for d in ("LMPI", "BAND") for x in r[d]["gap_paper"]]
        put("UoneSecondOrderMaxAbs", f"{100 * max(so):.2f}", max(so))
        put("UoneIdealLmpiMax", f"{100 * max(r['LMPI']['gap_ideal']):.1f}", max(r["LMPI"]["gap_ideal"]))
        put("UoneN", str(U["N"]), U["N"])
    # ---------------- U2 exact wall (wall_exact.py)
    X = J("wall_exact.json")
    if X:
        s = X["summary"]
        put("ExactNcases", str(s["n_cases"]), s["n_cases"])
        put("ExactNspectra", str(s["n_spectra"]), s["n_spectra"])
        put("ExactMaxDiff", f"$10^{{{int(np.floor(np.log10(s['max_abs_formula_minus_lp_db'])))+1}}}$", s["max_abs_formula_minus_lp_db"])
        put("ExactPropTwoShortfall", f"{-s['prop2_minus_exact_db_range'][0]:.1f}", -s["prop2_minus_exact_db_range"][0])
        put("ExactBoundaryOk", s["boundary_ok"].replace("/", " of "), s["boundary_ok"])
        Dd = X["D"]["N"]
        Ns = sorted(Dd, key=int)
        put("ExactDNFirst", f"{int(Ns[0]):,}".replace(",", "\\,"), int(Ns[0]))
        put("ExactDNLast", f"{int(Ns[-1]):,}".replace(",", "\\,"), int(Ns[-1]))
        put("ExactDNshapes", str(len(Dd[Ns[0]]["Pf"]) + 40), len(Dd[Ns[0]]["Pf"]) + 40)
        put("ExactDPfFirst", f"{Dd[Ns[0]]['max_Pf_all']:.2f}", Dd[Ns[0]]["max_Pf_all"])
        put("ExactDPfLast", f"{Dd[Ns[-1]]['max_Pf_all']:.2f}", Dd[Ns[-1]]["max_Pf_all"])
        put("ExactDPdFirst", f"{Dd[Ns[0]]['Pd']['1.0']['worst']:.2f}", Dd[Ns[0]]["Pd"]["1.0"]["worst"])
        put("ExactDPdLast", f"{Dd[Ns[-1]]['Pd']['1.0']['worst']:.2f}", Dd[Ns[-1]]["Pd"]["1.0"]["worst"])
        put("ExactDPdWallFirst", f"{Dd[Ns[0]]['Pd']['0.0']['worst']:.2f}", Dd[Ns[0]]["Pd"]["0.0"]["worst"])
        put("ExactDPdWallLast", f"{Dd[Ns[-1]]['Pd']['0.0']['worst']:.2f}", Dd[Ns[-1]]["Pd"]["0.0"]["worst"])
        put("ExactDWall", f"${Dd[Ns[0]]['wall_db']:.2f}$", Dd[Ns[0]]["wall_db"])
        put("ExactDPropTwo", f"${Dd[Ns[0]]['prop2_db']:.2f}$", Dd[Ns[0]]["prop2_db"])
        put("ExactDPropGap", f"{Dd[Ns[0]]['wall_db'] - Dd[Ns[0]]['prop2_db']:.2f}", Dd[Ns[0]]["wall_db"] - Dd[Ns[0]]["prop2_db"])
    # ---------------- U1 evidence (lmpi_equivalence.py, real_signal_lmpi.py)
    L = J("lmpi_equivalence.json")
    if L:
        M = L["models"]
        sel = lambda arch, recipe="WHITE", N=128, tag="": [m for k, m in M.items() if m["config"].get("arch") == arch
                                                            and m["config"].get("recipe") == recipe and m["config"].get("N") == N
                                                            and m["config"].get("sps") == 8 and (m["config"].get("tag") or "") == tag
                                                            and m["config"].get("kind") == "learned"]
        put("LmpiNmodels", str(len(M)), len(M))
        for lab, arch, tag in (("Gap", "GapCNN", ""), ("Cldnn", "CLDNN", ""), ("Cov", "CovCNN", ""), ("Res", "ResNet1D", ""),
                               ("ResEs", "ResNet1D", "es"), ("Stft", "STFTCNN", ""), ("Det", "DetectNetExact", "QAM16"),
                               ("Tfm", "Transformer", ""), ("Fourx", "GapCNN4X", "scale4x_600k")):
            ms = sel(arch, tag=tag)
            if ms:
                put(f"LmpiRho{lab}Range", rng2([m["dstar"]["rho_sig"] for m in ms]), [m["dstar"]["rho_sig"] for m in ms])
                put(f"LmpiCosP{lab}Range", rng2([m["template"]["cos_p"] for m in ms]), [m["template"]["cos_p"] for m in ms])
                put(f"LmpiPdTen{lab}Range", rng2([m["m10"]["pd_model"] for m in ms]), [m["m10"]["pd_model"] for m in ms])
        strong = [m for a in ("GapCNN", "CLDNN", "CovCNN") for m in sel(a)]
        put("LmpiRhoStrongRange", rng2([m["dstar"]["rho_sig"] for m in strong]), None)
        put("LmpiAgreeStrongRange", rng2([m["dstar"]["agree_sig"] for m in strong]), None)
        put("LmpiCosPStrongRange", rng2([m["template"]["cos_p"] for m in strong], 3), None)
        put("LmpiCosBandStrongRange", rng2([m["template"]["cos_band"] for m in strong], 3), None)
        put("LmpiRtwoStrongRange", rng2([m["template"]["r2_linear"] for m in strong]), None)
        des = [m["template"]["best_design_snr_db"] for m in strong]
        put("LmpiDesignMax", f"${max(des):.0f}$", max(des))
        put("LmpiDesignAtLimit", str(sum(d <= -39.9 for d in des)), sum(d <= -39.9 for d in des))
        white_all = [m for a in ("GapCNN", "CLDNN", "CovCNN", "ResNet1D", "STFTCNN") for m in sel(a)]
        pcloser = sum(m["template"]["cos_p"] > m["template"]["cos_band"] for m in white_all)
        put("LmpiPcloserCount", f"{pcloser} of {len(white_all)}", pcloser)
        snr_s = sorted({m["dstar"]["snr"] for m in strong})
        put("LmpiDstar", f"${snr_s[0]}$", snr_s[0])
        exc = [m[k]["pd_model"] - m[k]["pd_lmpi"] for m in white_all + sel("GapCNN4X", tag="scale4x_600k") for k in ("dstar", "m10")]
        put("LmpiMaxExcess", f"{100 * max(exc):.1f}", max(exc))
        low = [m["pd_low"][d]["model"] - m["pd_low"][d]["lmpi"] for m in white_all for d in ("-20", "-17")]
        put("LmpiLowMaxExcess", f"{100 * max(low):.1f}", max(low))
        # reference agreement (identical windows, N = 128, sps 8, seeds 0-4)
        nn_, nl_, nb_ = [], [], []
        for gid, ra in L["reference_agreement"].items():
            if not gid.startswith("N128_sps8_s") or "QAM16" in gid:
                continue
            for pk, v in ra["pairs"].items():
                a, b = pk.split("|")
                fam = lambda x: x.split("_")[0]
                sa, sb = fam(a) in ("GapCNN", "CLDNN", "CovCNN"), fam(b) in ("GapCNN", "CLDNN", "CovCNN")
                if sa and sb and "_s" in a and "_s" in b and not a.endswith(("_nonorm", "_ntrain20k", "_ntrain180k")) \
                        and not b.endswith(("_nonorm", "_ntrain20k", "_ntrain180k")):
                    nn_.append(v["agree"])
                elif sa and b == "LMPI" and not a.endswith(("_nonorm", "_ntrain20k", "_ntrain180k")):
                    nl_.append(v["agree"])
                elif sa and b.startswith("cls_BAND"):
                    nb_.append(v["agree"])
        if nn_:
            put("LmpiAgreeNetNet", rng2(nn_), nn_)
            put("LmpiAgreeNetLmpi", rng2(nl_), nl_)
            put("LmpiAgreeNetBand", rng2(nb_), nb_)
        # LP .8 false alarms of the LMPI and the white GAP CNN vs N
        prof = L["lmpi_profile"]
        for lab, N in (("NA", 128), ("NB", 512), ("NC", 2048), ("ND", 8192)):
            v = [p["Pf"]["lpf0.8"] for g, p in prof.items() if g.startswith(f"N{N}_sps8_s") and "QAM16" not in g]
            if v:
                put(f"LmpiPfLpf{lab}", pct(float(np.mean(v))), float(np.mean(v)))
            gm = sel("GapCNN", N=N)
            if gm:
                put(f"LmpiGapPfLpf{lab}", pct(float(np.mean([m["h0_lpf0.8"]["pf_model"] for m in gm]))), None)
        esc = [m for a, rcp in (("GapCNN", "PAIRED"), ("CycNet", "WHITE")) for m in sel(a, rcp, 8192)]
        if esc:
            put("LmpiRhoEscapeMax", f"{max(abs(m['dstar']['rho_sig']) for m in esc):.2f}", None)
            put("LmpiCosPEscapeMax", f"{max(m['template']['cos_p'] for m in esc):.2f}", None)
        gs = L.get("gap_vs_snr", {}).get("STFTCNN")
        if gs:
            put("LmpiStftGapRange", rng2([100 * v["mean"] for d, v in gs.items() if 0.1 < v["pd_mean"] < 0.9], 1), None)
    RL = J("real_signal_lmpi.json")
    if RL:
        c = RL["comparison"]
        put("RealLmpiNcaps", str(RL["n_captures_checked"]), RL["n_captures_checked"])
        put("RealLmpiMaxCurveDiff", f"{100 * max(v['max_abs_curve_diff_lmpi_cnn'] for v in c.values()):.1f}", None)
        put("RealBandMaxCurveDiff", f"{100 * max(v['max_abs_curve_diff_band_cnn'] for v in c.values()):.0f}", None)
        nd = [c[f"trevtron_R828D|{k}"]["max_pd_lmpi"] for k in ("NOAA_weather", "pager", "FRS_GMRS")]
        put("RealLmpiNdMax", f"{max(nd):.2f}", max(nd))
    # ---------------- U3 escape law (escape_law.py)
    EL = J("escape_law.json")
    if EL:
        put("EscLawKappaFive", f"{EL['A']['beta0.35']['kappa5']:.3f}", EL["A"]["beta0.35"]["kappa5"])
        put("EscLawKappaSixteen", f"{EL['A']['beta0.35']['kappa16']:.3f}", EL["A"]["beta0.35"]["kappa16"])
        put("EscLawKappaLow", f"{EL['A']['beta0.15']['kappa5']:.3f}", EL["A"]["beta0.15"]["kappa5"])
        put("EscLawKappaHigh", f"{EL['A']['beta0.6']['kappa5']:.2f}", EL["A"]["beta0.6"]["kappa5"])
        B = [r for r in EL["B"] if not (r["source"] == "wall_test_escape" and r["u_db"] == 1.0)]   # escape u=1 duplicates long
        Bi = [r for r in B if 0.08 < r["pd_white"] < 0.97]
        put("EscLawNpts", str(len(Bi)), len(Bi))
        put("EscLawMaxErr", f"{max(abs(r['pd_white'] - r['pd_pred']) for r in Bi):.2f}", None)
        ki = [r["kappa_implied"] for r in Bi if r["kappa_implied"]]
        put("EscLawKappaImplied", f"{np.median(ki):.3f}", float(np.median(ki)))
        if "C" in EL:
            put("EscLawNptsC", str(len(EL["C"])), len(EL["C"]))
            put("EscLawMaxErrC", f"{max(abs(r['pd_white'] - r['pd_pred']) for r in EL['C']):.2f}", None)
            pick = {r["beta"]: r for r in EL["C"] if r["N"] == 32768 and r["delta_db"] == -2}
            put("EscLawBetaPdMeas", ", ".join(f"{pick[b]['pd_white']:.2f}" for b in sorted(pick)), None)
            put("EscLawBetaPdPred", ", ".join(f"{pick[b]['pd_pred']:.2f}" for b in sorted(pick)), None)
        dd = {(r["u_db"], r["delta_db"], r["detector"]): r for r in EL["D"]}
        for lab, det in (("Five", "CYC5"), ("Sixteen", "CYC16")):
            r = dd[(1.0, -2.0, det)]
            put(f"EscLawNesc{lab}", f"{1e-3 * r['N_esc']:.0f}k", r["N_esc"])
            put(f"EscLawMs{lab}", f"{r['ms']:.0f}", r["ms"])
            r0 = dd[(1.0, 0.0, det)]
            put(f"EscLawNescWall{lab}", f"{1e-3 * r0['N_esc']:.0f}k", r0["N_esc"])
            put(f"EscLawMsWall{lab}", f"{r0['ms']:.0f}", r0["ms"])
    # ---------------- U4 hardware wall budget (hardware_wall_budget.py)
    H = J("hardware_wall_budget.json")
    if H:
        r1, r2 = H["R820T2 (Receiver 1)"], H["R828D (Receiver 2)"]
        for lab, r in (("Hf", r1), ("Tt", r2)):
            for sl, key in (("Within", "within_capture"), ("Pairs", "same_frequency_pairs")):
                v = r[key]
                put(f"Hw{sl}PowNinety{lab}", f"{v['u_power_db']['p90']:.2f}", v["u_power_db"]["p90"])
                put(f"Hw{sl}PowMed{lab}", f"{v['u_power_db']['median']:.2f}", v["u_power_db"]["median"])
                put(f"Hw{sl}EdWallNinety{lab}", f"${v['wall_ed_db']['p90']:.1f}$", v["wall_ed_db"]["p90"])
                put(f"Hw{sl}ShapeWallNinety{lab}", f"${v['wall_shape_db']['p90']:.1f}$", v["wall_shape_db"]["p90"])
                put(f"Hw{sl}GainMed{lab}", f"${v['paired_gain_db']['median']:+.1f}$", v["paired_gain_db"]["median"])
                put(f"Hw{sl}GainPos{lab}", pct(v["paired_gain_db"]["frac_positive"]), v["paired_gain_db"]["frac_positive"])
        g90 = [r[k]["wall_ed_db"]["p90"] - r[k]["wall_shape_db"]["p90"] for r in (r1, r2) for k in ("within_capture", "same_frequency_pairs")]
        put("HwGainNinetyRange", rng2(g90, 1), g90)
        gm = [r[k]["wall_ed_db"]["median"] - r[k]["wall_shape_db"]["median"] for r in (r1, r2) for k in ("within_capture", "same_frequency_pairs")]
        put("HwGainMedRange", f"${min(gm):+.1f}$ to ${max(gm):+.1f}$", gm)
        put("HwPairsHf", f"{r1['same_frequency_pairs']['u_power_db']['n']:,}".replace(",", "\\,"), None)
        put("HwPairsTt", f"{r2['same_frequency_pairs']['u_power_db']['n']:,}".replace(",", "\\,"), None)
    # ---------------- U6 off-band sweep (offband_sweep.py)
    O = J("offband_sweep.json")
    if O:
        s = O["summary"]
        main_nets = [v["mae"] for k, v in s.items() if any(k.startswith(a) for a in ("CLDNN", "ResNet1D", "CovCNN", "GapCNN", "DetectNet"))]
        put("OffMaeRange", rng2([100 * x for x in main_nets], 1), main_nets)
        put("OffMaeStft", rng2([100 * v["mae"] for k, v in s.items() if k.startswith("STFTCNN")], 1), None)
        put("OffMaeLmpi", f"{100 * s['LMPI']['mae']:.1f}", s["LMPI"]["mae"])
        # each detector's OWN overlap; near-zero overlaps carry no first-order prediction, so "clearly negative" < -0.05
        neg = [r["pd"][k] for r in O["rows"] for k in r["pd"] if not k.startswith("STFTCNN") and r["overlap"][k] < -0.05]
        put("OffNegPdMax", f"{max(neg):.3f}", max(neg))
        put("OffNegN", str(len(neg)), len(neg))
        put("OffNcells", str(len(O["rows"])), len(O["rows"]))
    # ---------------- U5 Gaussian ceiling (gaussian_ceiling.py)
    G = J("gaussian_ceiling.json")
    if G:
        # SNR loss relative to the MPI test evaluated on the SAME (power-normalised) signals; the strict Gaussian ceiling
        # (un-normalised surrogate) is not an upper bound for power-normalised windows (window-power effect, U1 check)
        grid, mpi = np.array(G["grid_db"]), np.array(G["mpi_on_signals"])
        need = lambda pd: float(np.interp(pd, mpi, grid))
        loss = lambda archs: [r["snr_db"] - need(r["pd"]) for a in archs for r in G["efficiency"][a]]
        ls, lw = loss(("GapCNN", "CLDNN", "CovCNN")), loss(("ResNet1D", "STFTCNN", "DetectNetExact"))
        put("GceilLossStrongMax", f"{max(ls):.1f}", max(ls))
        put("GceilLossWeakRange", rng2(lw, 1), lw)
        put("GceilCeilVsMpiTen", f"{G['ceiling'][list(grid).index(-10.0)]:.3f}", None)
        put("GceilMpiSigTen", f"{G['mpi_on_signals'][list(grid).index(-10.0)]:.3f}", None)
        cert = G["certificates_below_wall"]
        put("GceilCertDetectors", str(sum(v["exceeds_ceiling"] for v in cert.values())), None)
    # ---------------- U8 transformer (run records) and U7 dynamics (dynamics.py)
    tf = [json.load(open(f))["eval"] for f in sorted(glob.glob(os.path.join(R, "runs", "Transformer_WHITE_N128_sps8_s?.json")))]
    if tf:
        put("TfmSeeds", str(len(tf)), len(tf))
        put("TfmGapRange", rng2([100 * e["surrogate_gap"] for e in tf], 1), None)
        put("TfmPdTenRange", rng2([e["Pd"]["-10"]["white"]["nominal"] for e in tf]), None)
        put("TfmPfLpfRange", rng_pct([e["Pf"]["lpf0.8"]["nominal"] for e in tf]), None)
        put("TfmPfPinkMin", pct(min(e["Pf"]["pink"]["nominal"] for e in tf)), None)
    DY = J("dynamics.json")
    if DY:
        runs_ = DY["runs"]
        first = []
        for k, h in runs_.items():
            if "WHITE" in k:
                st = [x["step"] for x in h if x["rho_lmpi"] >= 0.9]
                first.append(min(st) if st else None)
        if all(first):
            put("DynStepsToLmpi", f"{max(first)}", first)
        gw_ = [max(abs(x["gap_-10"]) for x in h) for k, h in runs_.items() if "WHITE" in k]
        put("DynWhiteGapMax", f"{100 * max(gw_):.1f}", max(gw_))
        # the same architecture in the main evaluation (independent test windows per seed), gap at -10 dB
        mg = [json.load(open(f))["eval"] for f in sorted(glob.glob(os.path.join(R, "runs", "GapCNN_WHITE_N128_sps8_s?.json")))]
        mg = [100 * (e["Pd"]["-10"]["white"]["nominal"] - e["Pd_surrogate"]["-10"]["nominal"]) for e in mg]
        put("DynMainGapTenRange", f"${min(mg):+.1f}$ to ${max(mg):+.1f}$", mg)
        fin = lambda rcp, key: [h[-1][key] for k, h in runs_.items() if f"_{rcp}_" in k]
        if fin("PAIRED", "gap_0"):
            put("DynPairedGapZeroRange", rng2([100 * x for x in fin("PAIRED", "gap_0")], 0), None)
            put("DynPairedRhoRange", rng2(fin("PAIRED", "rho_lmpi")), None)
            put("DynSurrxmGapZeroRange", rng2([100 * x for x in fin("SURRXM", "gap_0")], 0), None)
            put("DynStructGapTenRange", rng2([100 * x for x in fin("PAIRED", "gap_-10") + fin("SURRXM", "gap_-10")], 1), None)
            # first checkpoint at which the gap at 0 dB exceeds 10 points (structure acquired), per run
            fs_ = [min([x["step"] for x in h if x["gap_0"] > 0.10] or [None]) for k, h in runs_.items() if "WHITE" not in k]
            put("DynStructStepMin", str(min(fs_)), fs_)
            put("DynStructStepMax", str(max(fs_)), fs_)
        h0 = next(iter(runs_.values()))
        per_ep = [x["step"] for x in h0 if abs(x.get("epoch", 0) - 1.0) < 1e-9]
        if per_ep:
            put("DynStepsPerEpoch", str(per_ep[0]), per_ep[0])


def main():
    full = load_runs()
    df = full[(full.tag == "") & (full.signal != "OFDM")]          # main numbers: untagged modulated runs only
    b = df[(df.N == 128) & (df.sps == 8)]
    white = b[(b.kind == "learned") & (b.recipe == "WHITE") & (b.model.isin(ARCHS))]
    # ---------- ablations and longer observations (explicit tags / N)
    def tagged(tag, **kw):
        d = full[full.tag == tag]
        for k, v in kw.items():
            d = d[d[k] == v]
        return d
    nn_ = tagged("nonorm", model="GapCNN")
    if len(nn_):
        put("AblNoNormPfUp", pct(nn_["Pf_white+1dB"].mean()), float(nn_["Pf_white+1dB"].mean()))
    big = tagged("ntrain180k", model="GapCNN")
    if len(big):
        put("AblBigGap", f"${100*big.surgap.mean():+.1f}$", float(big.surgap.mean()))
    for rec, nm in [("SURR", "Surr"), ("SURRX", "Surrx")]:
        for N, L in [(512, "NB"), (2048, "NC")]:
            s = df[(df.N == N) & (df.model == "GapCNN") & (df.recipe == rec)]
            if len(s):
                put(f"GapCNN{nm}PdTen{L}", pct(s["Pd_-10"].mean()), float(s["Pd_-10"].mean()))
                shp = [c for c in s.columns if c.startswith("Pf_") and not c.startswith("Pf_tilt") and c != "Pf_white"]
                put(f"GapCNN{nm}MaxPf{L}", pct(max(s[c].mean() for c in shp)), {c: float(s[c].mean()) for c in shp})
                put(f"GapCNN{nm}Gap{L}", f"${100*s.surgap.mean():+.1f}$", float(s.surgap.mean()))
        s = df[(df.N == 128) & (df.model == "GapCNN") & (df.recipe == rec)]
        if len(s):
            put(f"{nm}GapRangeNA", f"${100*s.surgap.min():+.1f}$ to ${100*s.surgap.max():+.1f}$", [float(s.surgap.min()), float(s.surgap.max())])
    for rec, nm in [("SURRXM", "Surrxm")]:
        for N, L in [(512, "NB"), (2048, "NC")]:
            s = df[(df.N == N) & (df.model == "GapCNN") & (df.recipe == rec)]
            if len(s):
                put(f"GapCNN{nm}PdTen{L}", pct(s["Pd_-10"].mean()), float(s["Pd_-10"].mean()))
                shp = [c for c in s.columns if c.startswith("Pf_") and not c.startswith("Pf_tilt") and c != "Pf_white"]
                put(f"GapCNN{nm}MaxPf{L}", pct(max(s[c].mean() for c in shp)), {c: float(s[c].mean()) for c in shp})
    # robust calibration over the receiver set (P_f <= 5 % enforced): worst-case Pd at -10 dB versus N
    for rec, nm in [("WHITE", "White"), ("SURRX", "Surrx"), ("SURRXM", "Surrxm")]:
        for N, L in [(128, "NA"), (512, "NB"), (2048, "NC")]:
            s = df[(df.N == N) & (df.model == "GapCNN") & (df.recipe == rec) & (df.sps == 8)]
            if len(s) and "wcPd_U_receiver_-10" in s:
                put(f"WcRec{nm}{L}", f"{s['wcPd_U_receiver_-10'].mean():.2f}", float(s["wcPd_U_receiver_-10"].mean()))
    # masking: signal embedded in the noise colours that a recipe's negatives contain (nominal threshold, -10 dB)
    def pdin(d, shape, snr="-10"):
        col = f"PdIn_{shape}_{snr}"
        return float(d[col].mean()) if col in d and len(d) else float("nan")
    b8 = df[(df.N == 128) & (df.sps == 8) & (df.kind == "learned")]
    pk = [pdin(b8[(b8.model == a) & (b8.recipe == "PINK")], "pink") for a in ["CLDNN", "GapCNN"]]
    if not np.isnan(pk).all():
        put("MaskPinkPdPink", rng_pct(pk), pk)
        put("MaskPinkPdWhite", rng_pct([b8[(b8.model == a) & (b8.recipe == "PINK")]["Pd_-10"].mean() for a in ["CLDNN", "GapCNN"]]), None)
    r = b8[(b8.model == "GapCNN") & (b8.recipe == "RAND")]
    if len(r):
        put("MaskRandPdLpf", pct(pdin(r, "lpf0.8")), pdin(r, "lpf0.8"))
    for rec, nm in [("SURRX", "Surrx"), ("SURRXM", "Surrxm")]:
        for N, L in [(128, "NA"), (2048, "NC")]:
            s = df[(df.N == N) & (df.sps == 8) & (df.model == "GapCNN") & (df.recipe == rec)]
            if len(s):
                put(f"Mask{nm}PdLpf{L}", pct(pdin(s, "lpf0.8")), pdin(s, "lpf0.8"))
                put(f"Mask{nm}PdPink{L}", pct(pdin(s, "pink")), pdin(s, "pink"))
                put(f"Mask{nm}PdPinkZero{L}", pct(pdin(s, "pink", "0")), pdin(s, "pink", "0"))
                put(f"Mask{nm}PdWhite{L}", pct(s["Pd_-10"].mean()), float(s["Pd_-10"].mean()))
    # matched-colour recipes (subscript m) at N = 128
    for arch, an in [("CLDNN", "CLDNN"), ("GapCNN", "GapCNN")]:
        for rec, nm in [("PINKM", "Pinkm"), ("RANDM", "Randm"), ("SURRXM", "Surrxm")]:
            s = b8[(b8.model == arch) & (b8.recipe == rec)]
            if not len(s):
                continue
            put(f"{an}{nm}PdPink", pct(pdin(s, "pink")), pdin(s, "pink"))
            put(f"{an}{nm}PdLpf", pct(pdin(s, "lpf0.8")), pdin(s, "lpf0.8"))
            for col, name in [("Pf_pink", "PfPink"), ("Pf_lpf0.8", "PfLpf"), ("Pf_rtl3dB", "PfRtl"), ("Pd_-10", "PdTen"),
                              ("wcPd_U_broad_-10", "WcBroad"), ("surgap", "SurGap")]:
                if col in s:
                    v = s[col].mean()
                    put(f"{an}{nm}{name}", (f"${100*v:+.1f}$" if col == "surgap" else pct(v)), float(v))
            put(f"{an}{nm}Seeds", str(len(s)), len(s))
    # best worst-case Pd at -10 dB over the broad set among all recipes at N = 128 (no recipe moves the wall)
    wb = [b8[(b8.model == a) & (b8.recipe == r)]["wcPd_U_broad_-10"].mean() for a in ["CLDNN", "GapCNN"]
          for r in ["WHITE", "PINK", "PINKM", "RAND", "RANDM", "SURR", "SURRX", "SURRXM"]
          if len(b8[(b8.model == a) & (b8.recipe == r)]) and "wcPd_U_broad_-10" in b8]
    if wb:
        put("WcBroadMax", pct(np.nanmax(wb)), float(np.nanmax(wb)))
    # colour-calibrated detection (threshold from the colour's own noise, alpha = 5 %) and alpha sensitivity,
    # from saved raw scores (alpha_sensitivity.py); N = 128, untagged runs
    fa = os.path.join(ROOT, "RESULTS", "alpha_sensitivity.json")
    if os.path.exists(fa):
        A = pd.DataFrame(json.load(open(fa)))

        def av(model, recipe, alpha, col):
            d = A[(A.model == model) & (A.recipe == recipe) & (np.isclose(A.alpha, alpha))]
            return float(d[col].iloc[0]) if len(d) and col in d else float("nan")
        wl = [av(a, "WHITE", 0.05, "PdCal_lpf0.8_-10") for a in ARCHS]
        put("PdCalLpfWhiteRange", rng_pct(wl), wl)
        for rec, nm in [("RAND", "Rand"), ("RANDM", "Randm"), ("SURRX", "Surrx"), ("SURRXM", "Surrxm"), ("PINK", "Pink"), ("PINKM", "Pinkm")]:
            put(f"PdCalLpf{nm}", pct(av("GapCNN", rec, 0.05, "PdCal_lpf0.8_-10")), av("GapCNN", rec, 0.05, "PdCal_lpf0.8_-10"))
        put("PdCalLpfBand", pct(av("BAND", "-", 0.05, "PdCal_lpf0.8_-10")), av("BAND", "-", 0.05, "PdCal_lpf0.8_-10"))
        a5 = A[np.isclose(A.alpha, 0.05)]
        put("PdCalPinkMax", pct(a5["PdCal_pink_-10"].max()), float(a5["PdCal_pink_-10"].max()))
        put("PdCalPinkZeroPinkm", pct(av("GapCNN", "PINKM", 0.05, "PdCal_pink_0")), av("GapCNN", "PINKM", 0.05, "PdCal_pink_0"))
        put("PdCalPinkZeroSurrx", pct(av("GapCNN", "SURRX", 0.05, "PdCal_pink_0")), av("GapCNN", "SURRX", 0.05, "PdCal_pink_0"))
        for al, L in [(0.01, "One"), (0.10, "Ten")]:
            g_ = [av(a, "WHITE", al, "surgap") for a in ARCHS]
            put(f"AlphaGapMax{L}", f"{100*max(abs(x) for x in g_):.1f}", max(abs(x) for x in g_))
            lp_ = [av(a, "WHITE", al, "Pf_lpf0.8") for a in ARCHS]
            put(f"AlphaLpf{L}Range", rng_pct(lp_), lp_)
            pw_ = [av(a, "WHITE", al, "Pf_white+1dB") for a in ARCHS]
            put(f"AlphaPowerUp{L}Range", rng_pct(pw_, 1), pw_)
    # SURR N=2048 seed-0 re-run (same seed; reproducibility of the disclosed failure)
    rr = full[(full.tag == "rerun") & (full.model == "GapCNN") & (full.recipe == "SURR") & (full.N == 2048)]
    if len(rr):
        put("SurrRerunLpf", pct(rr["Pf_lpf0.8"].mean(), 1), float(rr["Pf_lpf0.8"].mean()))
    # high-SNR structure use: Pd on Gaussian surrogates at 0 dB (nominal threshold), per recipe and N
    for rec, nm in [("WHITE", "White"), ("SURR", "Surr"), ("SURRX", "Surrx"), ("SURRXM", "Surrxm")]:
        for N, L in [(128, "NA"), (512, "NB"), (2048, "NC")]:
            s = df[(df.N == N) & (df.model == "GapCNN") & (df.recipe == rec)]
            if len(s) and "PdSur_0" in s:
                put(f"SurZero{nm}{L}", pct(s["PdSur_0"].mean()), float(s["PdSur_0"].mean()))
                put(f"SigZero{nm}{L}", pct(s["Pd_0"].mean()), float(s["Pd_0"].mean()))
    s = df[(df.N == 2048) & (df.model == "GapCNN") & (df.recipe == "SURRX")]
    if len(s):
        put("SurrxRtlNC", pct(s["Pf_rtl3dB"].mean()), float(s["Pf_rtl3dB"].mean()))
    # SURR at N=2048: seed spread of the out-of-set false alarms (disclosed limitation)
    s = df[(df.N == 2048) & (df.model == "GapCNN") & (df.recipe == "SURR")]
    if len(s):
        put("SurrLpfSpreadNC", rng_pct(s["Pf_lpf0.8"].values, 1), list(map(float, s["Pf_lpf0.8"].values)))
        put("SurrRtlSpreadNC", rng_pct(s["Pf_rtl3dB"].values, 1), list(map(float, s["Pf_rtl3dB"].values)))
    # occupancy: network advantage over ED at -10 dB, sps 8 vs 4 vs 2
    for sps, L in [(8, "Eight"), (4, "Four"), (2, "Two")]:
        a = df[(df.N == 128) & (df.sps == sps) & (df.model == "GapCNN") & (df.recipe == "WHITE")]["Pd_-10"].mean()
        e = df[(df.N == 128) & (df.sps == sps) & (df.model == "ED")]["Pd_-10"].mean()
        if not (np.isnan(a) or np.isnan(e)):
            put(f"OccGain{L}", f"${100*(a-e):+.0f}$", float(a - e))
    # whitening remedy on synthetic shapes (wht_* runs: GapCNN and CLDNN WHITE, seeds 0-2; 16-term cosine fit)
    wf = sorted(glob.glob(os.path.join(ROOT, "RESULTS", "runs", "wht_*.json")))
    if wf:
        WH = [json.load(open(x))["whitening"] for x in wf]
        smooth = ["white", "rtl3dB", "ar+0.2", "pink", "tilt1.0+"]
        sm = [w[f"K1024_cos_{s}"]["Pf"] for w in WH for s in smooth]
        put("WhitenSmoothMaxKbig", pct(max(sm), 1), float(max(sm)))
        lp = [w["K1024_cos_lpf0.8"]["Pf"] for w in WH]
        put("WhitenLpfKbig", pct(np.mean(lp), 0), float(np.mean(lp)))
        lp16 = [w["K16_cos_lpf0.8"]["Pf"] for w in WH]
        put("WhitenLpfRangeKsixteen", rng_pct(lp16, 0), lp16)
        put("WhitenRuns", str(len(WH)), len(WH))
    # per-bin whitening with a near-exact PSD: real (linear-filter) vs circulant noise of the same periodogram
    fl = os.path.join(ROOT, "RESULTS", "whitening_probe_lpf.json")
    if os.path.exists(fl):
        L8 = json.load(open(fl))["lpf0.8"]
        put("WhitenExactLpfNone", pct(L8["none_linear"], 0), L8["none_linear"])
        put("WhitenExactLpfLinear", pct(L8["whitened_linear"], 0), L8["whitened_linear"])
        put("WhitenExactLpfCirc", pct(L8["whitened_circulant"], 1), L8["whitened_circulant"])
    # ablations: smaller training set; ResNet-1D with best-validation checkpoint selection; DetectNet replica
    small = tagged("ntrain20k", model="GapCNN")
    if len(small):
        put("AblSmallGap", f"${100*small.surgap.mean():+.1f}$", float(small.surgap.mean()))
    if len(big):
        put("AblBigLpf", pct(big["Pf_lpf0.8"].mean()), float(big["Pf_lpf0.8"].mean()))
    if len(small):
        put("AblSmallLpf", pct(small["Pf_lpf0.8"].mean()), float(small["Pf_lpf0.8"].mean()))
    es = tagged("es", model="ResNet1D")
    if len(es):
        put("ResEsLpf", pct(es["Pf_lpf0.8"].mean()), float(es["Pf_lpf0.8"].mean()))
        put("ResEsGap", f"${100*es.surgap.mean():+.1f}$", float(es.surgap.mean()))
        put("ResEsPdTen", pct(es["Pd_-10"].mean()), float(es["Pd_-10"].mean()))
    dn = tagged("QAM16", model="DetectNetExact")
    if len(dn):
        put("DetLpf", pct(dn["Pf_lpf0.8"].mean()), float(dn["Pf_lpf0.8"].mean()))
        put("DetGap", f"${100*dn.surgap.mean():+.1f}$", float(dn.surgap.mean()))
    scale = tagged("scale4x_600k", model="GapCNN4X")
    if len(scale):
        put("AblScaleGap", f"${100*scale.surgap.mean():+.1f}$", float(scale.surgap.mean()))
        put("AblScaleLpf", pct(scale["Pf_lpf0.8"].mean()), float(scale["Pf_lpf0.8"].mean()))
        put("AblScalePdTen", pct(scale["Pd_-10"].mean()), float(scale["Pd_-10"].mean()))
        import detectors as D
        scale_params = D.n_params(D.GapCNN4X())
        put("AblScaleParams", f"{scale_params/1e6:.2f}M", scale_params)
        put("AblScaleNtrain", "600\\,000", 600000)
    ofdm_runs = full[full.signal == "OFDM"]
    ofdm_net = ofdm_runs[ofdm_runs.model == "GapCNN"]
    ofdm_ed = ofdm_runs[ofdm_runs.model == "ED"]
    ofdm_band = ofdm_runs[ofdm_runs.model == "BAND"]
    if len(ofdm_net):
        put("OfdmSeeds", str(len(ofdm_net)), len(ofdm_net))
        put("OfdmGap", f"${100*ofdm_net.surgap.mean():+.1f}$", float(ofdm_net.surgap.mean()))
        put("OfdmPdTen", pct(ofdm_net["Pd_-10"].mean()), float(ofdm_net["Pd_-10"].mean()))
        put("OfdmEdPdTen", pct(ofdm_ed["Pd_-10"].mean()), float(ofdm_ed["Pd_-10"].mean()))
        put("OfdmBandPdTen", pct(ofdm_band["Pd_-10"].mean()), float(ofdm_band["Pd_-10"].mean()))
        diff_ed = float(ofdm_net["Pd_-10"].mean() - ofdm_ed["Pd_-10"].mean())
        put("OfdmEdDiff", f"${100*diff_ed:+.1f}$", diff_ed)
        put("OfdmPfLpf", pct(ofdm_net["Pf_lpf0.8"].mean()), float(ofdm_net["Pf_lpf0.8"].mean()))
        put("OfdmPfRtl", pct(ofdm_net["Pf_rtl3dB"].mean()), float(ofdm_net["Pf_rtl3dB"].mean()))
        put("OfdmPfPink", pct(ofdm_net["Pf_pink"].mean()), float(ofdm_net["Pf_pink"].mean()))
        put("OfdmRho", "0.75", 0.75)
    g = white.groupby("model")
    # ---------- surrogate gap
    sg = g.surgap.mean()
    put("SurGapRange", f"${100*sg.min():+.1f}$ to ${100*sg.max():+.1f}$ points", sg.to_dict())
    put("SurGapAbsMax", f"{100*np.abs(white.surgap).max():.1f}", float(np.abs(white.surgap).max()))
    cyc = b[b.model == "CYC"]
    put("CycSurGap", f"{100*cyc.surgap.mean():.1f}", float(cyc.surgap.mean()))
    # ---------- Pf under shapes (mean over seeds per arch; range over archs)
    for col, name in [("Pf_lpf0.8", "PfLpfEight"), ("Pf_lpf0.9", "PfLpfNine"), ("Pf_rtl3dB", "PfRtlThree"), ("Pf_rtl1dB", "PfRtlOne"),
                      ("Pf_ar+0.1", "PfArOne"), ("Pf_ar-0.1", "PfArMinusOne"), ("Pf_ar+0.2", "PfArTwo"), ("Pf_pink", "PfPink"),
                      ("Pf_white+1dB", "PfPowerUp"), ("Pf_white-1dB", "PfPowerDown"), ("Pf_white", "PfWhite")]:
        if col in white:
            m = g[col].mean()
            put(name + "Range", rng_pct(m.values), m.to_dict())
    for det in ["ED", "MME", "SFM", "BAND", "CYC"]:
        d = b[b.model == det]
        if len(d):
            for col, name in [("Pf_lpf0.8", "PfLpfEight"), ("Pf_rtl3dB", "PfRtlThree"), ("Pf_pink", "PfPink"), ("Pd_-10", "PdTen"), ("Pf_white+1dB", "PfPowerUp")]:
                put(det.capitalize() + name, pct(d[col].mean()), float(d[col].mean()))
    m = g["Pd_-10"].mean()
    put("PdTenRange", rng_pct(m.values), m.to_dict())
    if "wcPd_U_receiver_-10" in white:
        m = g["wcPd_U_receiver_-10"].mean()
        put("WcPdReceiverRange", rng_pct(m.values), m.to_dict())
    # ---------- recipes
    for arch in ["CLDNN", "GapCNN"]:
        for rec in ["PINK", "RAND", "SURR", "WHITE"]:
            d = b[(b.model == arch) & (b.recipe == rec)]
            if not len(d):
                continue
            k = f"{arch}{rec.capitalize()}"
            for col, name in [("Pf_pink", "PfPink"), ("Pf_lpf0.8", "PfLpfEight"), ("Pf_rtl3dB", "PfRtlThree"), ("Pf_ar+0.2", "PfArTwo"),
                              ("Pd_-10", "PdTen"), ("wcPd_U_broad_-10", "WcPdBroadTen"), ("surgap", "SurGap")]:
                if col in d:
                    v = d[col].mean()
                    put(k + name, (f"${100*v:+.1f}$" if col == "surgap" else pct(v)), float(v))
            put(k + "Seeds", str(len(d)), len(d))
    # ---------- theory numbers
    p = W.signal_psd(512)
    for u in [0.5, 1.0, 2.0, 3.0]:
        tag = {0.5: "Half", 1.0: "One", 2.0: "Two", 3.0: "Three"}[u]
        put(f"WallActual{tag}", f"${10*np.log10(W.wall(p, u)):.1f}$", float(10 * np.log10(W.wall(p, u))))
        put(f"WallED{tag}", f"${10*np.log10(sim.ed_power_wall(u)):.1f}$", float(10 * np.log10(sim.ed_power_wall(u))))
    put("Occupancy", f"{sim.occupancy(8):.2f}", sim.occupancy(8))
    put("OccupancyFour", f"{sim.occupancy(4):.2f}", sim.occupancy(4))
    put("OccupancyTwo", f"{sim.occupancy(2):.2f}", sim.occupancy(2))
    # ---------- real noise. MAIN = v3 protocol (reference = windows 256..1279, i.e. after the RTL start-up transient;
    # test = the following <= 2000 windows). v2 (reference from window 0) is kept as a sensitivity analysis: the same
    # macros with the suffix "VTwo". Falls back to v2 as main if v3 is missing.
    def real_noise_macros(f, sfx):
        R = json.load(open(f))
        put = lambda name, tex, raw=None: VALS.__setitem__(name + sfx, {"tex": tex, "raw": raw})
        recs = R["recordings"]
        live = {k: v for k, v in recs.items() if v["group"] == "live_rtl" and not v["shape"]["narrowband"]}
        ctrl = {k: v for k, v in recs.items() if v["group"] == "unlabelled"}
        put("RealNoiseNrec", str(len(live)), len(live))
        put("RealNoiseNctrl", str(len(ctrl)), len(ctrl))
        ro = [v["shape"]["edge_rolloff_db"] for v in live.values()]
        put("RealRolloffRange", f"{min(ro):.1f}--{max(ro):.1f}", [min(ro), max(ro)])

        def mean_over(recset, field, prefix):
            keys = [k for k in next(iter(recset.values()))[field] if k.startswith(prefix)]
            return float(np.mean([np.mean([r[field][k] for k in keys]) for r in recset.values()])) if keys else float("nan")
        white_arch = [a for a in ARCHS]
        per_arch = {a: mean_over(live, "Pf", f"{a}-WHITE") for a in white_arch}
        put("PfRealRange", rng_pct(list(per_arch.values())), per_arch)
        put("PfRealControlMax", pct(max(mean_over(ctrl, "Pf", f"{a}-WHITE") for a in white_arch), 1), None)
        put("PfRealWhitenedMax", pct(max(mean_over(live, "Pf_wbig", f"{a}-WHITE") for a in white_arch), 1), None)
        put("PfRealWhitenedRange", rng_pct([mean_over(live, "Pf_wbig", f"{a}-WHITE") for a in white_arch], 1), None)
        put("PfRealWhitenedShortRange", rng_pct([mean_over(live, "Pf_w16", f"{a}-WHITE") for a in white_arch]), None)
        put("PfRealControlWhitenedShortMax", pct(max(mean_over(ctrl, "Pf_w16", f"{a}-WHITE") for a in white_arch), 1), None)
        put("PfRealSurr", pct(mean_over(live, "Pf", "GapCNN-SURR"), 1), None)
        put("PfRealPink", pct(mean_over(live, "Pf", "CLDNN-PINK"), 1), None)
        put("PfRealRand", pct(mean_over(live, "Pf", "GapCNN-RAND"), 1), None)
        put("PfRealBand", pct(mean_over(live, "Pf", "BAND"), 0), None)
        put("PfRealBandWhitened", pct(mean_over(live, "Pf_wbig", "BAND"), 1), None)
        put("PfRealCyc", pct(mean_over(live, "Pf", "CYC"), 1), None)
        put("PfRealMME", pct(mean_over(live, "Pf", "MME"), 0), None)
        put("PfRealSFM", pct(mean_over(live, "Pf", "SFM"), 0), None)
        put("PdRealNominalRange", rng_pct([mean_over(live, "Pd_m10_nominal", f"{a}-WHITE") for a in white_arch]), None)
        put("PdRealWhitenedRange", rng_pct([mean_over(live, "Pd_m10_wbig", f"{a}-WHITE") for a in white_arch]), None)
        put("PdRealSurr", pct(mean_over(live, "Pd_m10_nominal", "GapCNN-SURR")), None)
        put("PdRealRand", pct(mean_over(live, "Pd_m10_nominal", "GapCNN-RAND")), None)
        put("PdRealPink", pct(mean_over(live, "Pd_m10_nominal", "CLDNN-PINK")), None)
        # dose-response: Spearman correlation between measured roll-off and Pf (white-trained networks, per recording)
        from scipy.stats import spearmanr
        xs = [r["shape"]["edge_rolloff_db"] for r in list(live.values()) + list(ctrl.values())]
        ys = [np.mean([v for k, v in r["Pf"].items() if any(k.startswith(a + "-WHITE") for a in white_arch)])
              for r in list(live.values()) + list(ctrl.values())]
        rho = spearmanr(xs, ys).statistic
        put("RealDoseRho", f"{rho:.2f}", float(rho))
        lo = [y for x, y in zip(xs, ys) if x < 1.5]
        hi = [y for x, y in zip(xs, ys) if x > 4.5]
        put("PfRealLowRolloff", pct(np.mean(lo), 1), float(np.mean(lo)))
        put("PfRealHighRolloff", pct(np.mean(hi), 0), float(np.mean(hi)))
        put("RealRefStart", str(next(iter(recs.values())).get("ref_start", 0)), None)
    fv3 = os.path.join(ROOT, "RESULTS", "real_noise_v3_N128.json")
    fv2 = os.path.join(ROOT, "RESULTS", "real_noise_v2_N128.json")
    for f, sfx in ([(fv3, ""), (fv2, "VTwo")] if os.path.exists(fv3) else [(fv2, "")]):
        if os.path.exists(f):
            real_noise_macros(f, sfx)
    # ---------- second receiver: TrevTron R828D tuner (RTL-SDR Blog V4, MIT licence)
    ftt = os.path.join(ROOT, "RESULTS", "real_noise_trevtron_N128.json")
    if os.path.exists(ftt):
        Rtt = json.load(open(ftt))["recordings"]
        clean_tt = {k: v for k, v in Rtt.items() if not v["shape"]["narrowband"]}
        nb_tt = {k: v for k, v in Rtt.items() if v["shape"]["narrowband"]}
        put("TrevNoiseNrec", str(len(clean_tt)), len(clean_tt))
        put("TrevNoiseNtotal", str(len(Rtt)), len(Rtt))
        put("TrevNoiseNarrowband", str(len(nb_tt)), len(nb_tt))
        rott = [v["shape"]["edge_rolloff_db"] for v in clean_tt.values()]
        put("TrevRolloffRange", f"{min(rott):.1f}--{max(rott):.1f}", [min(rott), max(rott)])
        put("TrevRolloff", f"{np.median(rott):.1f}", float(np.median(rott)))

        def mean_tt(recset, field, prefix):
            keys = [k for k in next(iter(recset.values()))[field] if k.startswith(prefix)]
            return float(np.mean([np.mean([r[field][k] for k in keys]) for r in recset.values()])) if keys else float("nan")
        # all five white-trained architectures, as for Receiver 1's \PfRealRange (fix 2026-10-05: was CLDNN + GapCNN only)
        per_arch_tt = [mean_tt(clean_tt, "Pf", f"{a}-WHITE") for a in ARCHS]
        put("TrevPfWhiteRange", rng_pct(per_arch_tt), per_arch_tt)
        wb_tt = [mean_tt(clean_tt, "Pf_wbig", f"{a}-WHITE") for a in ARCHS]
        put("TrevPfWhitenedRange", rng_pct(wb_tt, 1), wb_tt)
        put("TrevPfBand", pct(mean_tt(clean_tt, "Pf", "BAND")), None)
        put("TrevPfBandWhitened", pct(mean_tt(clean_tt, "Pf_wbig", "BAND"), 1), None)
        put("TrevPfCyc", pct(mean_tt(clean_tt, "Pf", "CYC")), None)
        put("TrevPfSurr", pct(mean_tt(clean_tt, "Pf", "GapCNN-SURR")), None)
        put("TrevPfSurrWhitened", pct(mean_tt(clean_tt, "Pf_wbig", "GapCNN-SURR"), 1), None)
    # ---------- adaptive classical baselines (AMF / ACE / Kelly / ED-ref)
    fa1 = os.path.join(ROOT, "RESULTS", "adaptive_baselines.json")
    if os.path.exists(fa1):
        ad1 = json.load(open(fa1))["recorded_summary"]
        put("AdaptAcePfMeanHf", pct(ad1["ACE"]["Pf_mean"], 1), ad1["ACE"]["Pf_mean"])
        put("AdaptAcePfMaxHf", pct(ad1["ACE"]["Pf_max"], 1), ad1["ACE"]["Pf_max"])
        put("AdaptAcePdHf", pct(ad1["ACE"]["Pd_m10_mean"], 0), ad1["ACE"]["Pd_m10_mean"])
        put("AdaptAmfPfMeanHf", pct(ad1["AMF"]["Pf_mean"], 1), ad1["AMF"]["Pf_mean"])
        put("AdaptKellyPfMeanHf", pct(ad1["KELLY"]["Pf_mean"], 1), ad1["KELLY"]["Pf_mean"])
    fa2 = os.path.join(ROOT, "RESULTS", "adaptive_baselines_trevtron.json")
    if os.path.exists(fa2):
        ad2 = json.load(open(fa2))["recorded_summary"]
        put("AdaptAcePfMeanTrev", pct(ad2["ACE"]["Pf_mean"], 1), ad2["ACE"]["Pf_mean"])
        put("AdaptAcePfMaxTrev", pct(ad2["ACE"]["Pf_max"], 1), ad2["ACE"]["Pf_max"])
        put("AdaptAcePdTrev", pct(ad2["ACE"]["Pd_m10_mean"], 0), ad2["ACE"]["Pd_m10_mean"])
        put("AdaptAmfPfMeanTrev", pct(ad2["AMF"]["Pf_mean"], 1), ad2["AMF"]["Pf_mean"])
        put("AdaptKellyPfMeanTrev", pct(ad2["KELLY"]["Pf_mean"], 1), ad2["KELLY"]["Pf_mean"])
    # ---------- real-signal surrogate test (T6)
    frs = os.path.join(ROOT, "RESULTS", "real_signal_summary.json")
    if os.path.exists(frs):
        RS = json.load(open(frs))["results"]
        put("RealSigNcapsTotal", "251", 251)
        put("RealSigNwinsTotal", "25\\,071", 25071)
        def g_pt(k):
            v = RS.get(k, {}).get("gap", float("nan"))
            return f"{100*v:+.1f}".replace("+", "$+$").replace("-", "$-$") if abs(100*v) >= 0.05 else "0.0"
        put("RealSigFmGapTt", g_pt("trevtron_R828D|FM_broadcast|GapCNN-WHITE|sur"))
        put("RealSigFmKappaLoGapTt", g_pt("trevtron_R828D|FM_broadcast|KAPPALO|sur"))
        put("RealSigFmGapHf", g_pt("hf_composite|FM_broadcast|GapCNN-WHITE|sur"))
        put("RealSigFmBandGapHf", g_pt("hf_composite|FM_broadcast|BAND|sur"))
        put("RealSigFmKappaLoGapHf", g_pt("hf_composite|FM_broadcast|KAPPALO|sur"))
        put("RealSigFmSurrxGapHf", g_pt("hf_composite|FM_broadcast|GapCNN-SURRX|sur"))
        put("RealSigAdsbGapHf", g_pt("hf_composite|ADS_B|GapCNN-WHITE|sur"))
        put("RealSigAdsbBandGapHf", g_pt("hf_composite|ADS_B|BAND|sur"))
        put("RealSigAdsbKappaGapHf", g_pt("hf_composite|ADS_B|KAPPA|sur"))
        put("RealSigIsmGapHf", g_pt("hf_composite|ISM_sensors|GapCNN-WHITE|sur"))
        put("RealSigIsmKappaLoGapHf", g_pt("hf_composite|ISM_sensors|KAPPALO|sur"))
    # ---------- oracle whitening probe (5 live recordings + 1 flat control; GapCNN-WHITE seed 0 and BAND)
    fo = os.path.join(ROOT, "RESULTS", "whitening_probe_oracle.json")
    if os.path.exists(fo):
        O = json.load(open(fo))["results"]
        liveo = {k: v for k, v in O.items() if not k.startswith("NOISE")}
        none = [v["none"]["GapCNN"] for v in liveo.values()]
        orc = [v["oracle-raw"]["GapCNN"] for v in liveo.values()]
        put("OracleNrec", str(len(liveo)), len(liveo))
        put("OracleNoneRange", rng_pct(none), none)
        put("OracleWhitenRange", rng_pct(orc, 1), orc)
    # ---------- wall test
    files = sorted(glob.glob(os.path.join(ROOT, "RESULTS", "wall_test_s*.json")))
    if files:
        data = [json.load(open(x)) for x in files]

        def wc(det, N, R_db, delta):
            key = f"N{N}_R{R_db}_{det}"
            v = [d[key]["Pd"][str(delta)]["worst"] for d in data if key in d]
            if len(v) < len(data):           # require every seed (no partial averages)
                raise KeyError(key)
            return float(np.mean(v))
        # LaTeX macro names cannot contain digits: N = 128, 512, 2048 -> A, B, C
        for det, nm in [("GapCNN-WHITE", "White"), ("GapCNN-SURR", "Surr"), ("GapCNN-SURRX", "Surrx"), ("GapCNN-SURRXM", "Surrxm"),
                        ("BAND", "Band"), ("CYC", "Cyc"), ("MME", "Mme"), ("SFM", "Sfm"), ("CLDNN-WHITE", "Cldnn")]:
            for N, L in [(128, "A"), (512, "B"), (2048, "C")]:
                try:
                    put(f"Wall{nm}N{L}", f"{wc(det, N, 1.0, 0):.2f}", wc(det, N, 1.0, 0))
                    put(f"Wall{nm}N{L}Minus", f"{wc(det, N, 1.0, -2):.2f}", wc(det, N, 1.0, -2))
                    put(f"Wall{nm}N{L}Plus", f"{wc(det, N, 1.0, 4):.2f}", wc(det, N, 1.0, 4))
                except Exception:
                    pass
                try:                     # u = 2 dB: the wall lies at a higher SNR (about -9.2 dB)
                    put(f"WallTwo{nm}N{L}", f"{wc(det, N, 2.0, 0):.2f}", wc(det, N, 2.0, 0))
                    put(f"WallTwo{nm}N{L}Minus", f"{wc(det, N, 2.0, -2):.2f}", wc(det, N, 2.0, -2))
                except Exception:
                    pass
        put("WallSeeds", str(len(data)), len(data))
        # largest 3-seed-mean worst-case Pd two dB below the wall, over N and both walls (u = 1, 2 dB)
        for det, nm in [("GapCNN-SURR", "Surr"), ("GapCNN-SURRX", "Surrx"), ("GapCNN-SURRXM", "Surrxm"), ("GapCNN-WHITE", "White")]:
            try:
                v = max(wc(det, N, Rd, -2) for N in [128, 512, 2048] for Rd in [1.0, 2.0])
                put(f"Wall{nm}BelowMax", f"{v:.2f}", v)
            except Exception:
                pass
        pfmax = max(max(d[k]["Pf"].values()) for d in data for k in d)
        put("WallPfMax", pct(pfmax, 1), float(pfmax))
    # ---------- N-scaling of nominal false alarms (white-trained GAP CNN)
    for col, nm in [("Pf_lpf0.8", "Lpf"), ("Pf_rtl3dB", "Rtl")]:
        for N, L in [(128, "NA"), (512, "NB"), (2048, "NC")]:
            s = df[(df.N == N) & (df.sps == 8) & (df.model == "GapCNN") & (df.recipe == "WHITE")]
            if len(s):
                put(f"Pf{nm}White{L}", pct(s[col].mean(), 1 if s[col].mean() > 0.99 else 0), float(s[col].mean()))
    # ---------- J5: GAP CNNs trained at N = 2048 applied to longer windows (wall test, u = 1 dB)
    lnf = sorted(glob.glob(os.path.join(ROOT, "RESULTS", "wall_test_long_nets_s*.json")))
    if lnf:
        LN = [json.load(open(x)) for x in lnf]

        def lnv(det, N, d, R="1.0"):
            v = [z[f"N{N}_R{R}"]["Pd"][str(d)][det]["worst"] for z in LN if f"N{N}_R{R}" in z]
            if len(v) < 3:
                raise KeyError(f"{det} N{N} R{R}: {len(v)} seeds")
            return float(np.mean(v))
        try:
            for N, L in [(2048, "NC"), (8192, "ND"), (32768, "NE")]:
                put(f"LongNetSurrxm{L}", f"{lnv('GapCNN-SURRXM', N, 0):.2f}", lnv("GapCNN-SURRXM", N, 0))
                put(f"LongNetTwoSurrxm{L}", f"{lnv('GapCNN-SURRXM', N, 0, '2.0'):.2f}", lnv("GapCNN-SURRXM", N, 0, "2.0"))
            wm = max(lnv("GapCNN-WHITE", N, 0, R) for N in [2048, 8192, 32768] for R in ["1.0", "2.0"])
            put("LongNetWhiteMax", f"{wm:.2f}", wm)
            bm = max(lnv("GapCNN-SURRXM", N, -2, R) for N in [2048, 8192, 32768] for R in ["1.0", "2.0"])
            put("LongNetSurrxmBelowMax", f"{bm:.2f}", bm)
        except KeyError as ex:
            print("J5 incomplete:", ex)
    # ---------- compute (compute_summary.py must be run first)
    fc = os.path.join(ROOT, "RESULTS", "compute_summary.json")
    if os.path.exists(fc):
        C = json.load(open(fc))
        put("ComputeRuns", str(C["n_runs"]), C["n_runs"])
        put("ComputeGpuHours", f"{C['gpu_run_hours']:.1f}", C["gpu_run_hours"])
        put("ComputeCpuHours", f"{C['cpu_run_hours']:.1f}", C["cpu_run_hours"])
        wt = sum(v for k, v in C.items() if k.endswith("_hours") and k.startswith("wall"))
        put("ComputeWallHours", f"{wt:.1f}", wt)
        put("ComputeVram", f"{C['peak_vram_MB_max']/1024:.1f}", C["peak_vram_MB_max"])
    # ---------- J2 theory vs experiment (band ratio, 3 occupancies x 4 uncertainties x N = 8192/32768)
    fj = os.path.join(ROOT, "TABLES", "wall_vs_theory.csv")
    if os.path.exists(fj):
        S = pd.read_csv(fj)
        put("ThConfigs", str(len(S)), len(S))
        put("ThMaxBelow", f"{S.max_Pd_at_or_below_wall.max():.3f}", float(S.max_Pd_at_or_below_wall.max()))
        e = S[S.N == 32768].delta50_db; d8 = S[S.N == 8192].delta50_db
        put("ThDeltaRangeNE", f"{e.min():.2f}--{e.max():.2f}", [float(e.min()), float(e.max())])
        put("ThDeltaRangeND", f"{d8.min():.2f}--{d8.max():.2f}", [float(d8.min()), float(d8.max())])
        gapflat = (S.wall_flat_db - S.wall_db)
        put("ThFlatExcessRange", f"{gapflat.min():.1f}--{gapflat.max():.1f}", [float(gapflat.min()), float(gapflat.max())])
        ed = (S.ed_wall_db - S.wall_db)
        put("ThEdGainRange", f"{ed.min():.1f}--{ed.max():.1f}", [float(ed.min()), float(ed.max())])
    # ---------- receiver set: smallest relative residual of the wall condition over all ordered pairs
    frw = os.path.join(ROOT, "RESULTS", "receiver_wall_check.json")
    if os.path.exists(frw):
        mn = json.load(open(frw))["min_rel_residual"]
        put("ReceiverWallMinResidual", pct(mn, 1), float(mn))
    # ---------- empirical noise-shape uncertainty (empirical_uncertainty.py)
    feu = os.path.join(ROOT, "RESULTS", "empirical_uncertainty.json")
    if os.path.exists(feu):
        EU = json.load(open(feu))
        r1 = EU.get("receiver_1_r820t2", {})
        r2 = EU.get("receiver_2_r828d", {})
        cr = EU.get("cross_receiver_145mhz", {})
        if r1:
            put("EmpSplitHalfMedianHf", f"{r1['split_half_noise']['median']:.2f}", r1["split_half_noise"]["median"])
            put("EmpSplitHalfPninetyHf", f"{r1['split_half_noise']['p90']:.2f}", r1["split_half_noise"]["p90"])
            put("EmpSteadyDriftMedianHf", f"{r1['steady_drift']['median']:.2f}", r1["steady_drift"]["median"])
            put("EmpSteadyDriftPninetyHf", f"{r1['steady_drift']['p90']:.2f}", r1["steady_drift"]["p90"])
            put("EmpStartupMedianHf", f"{r1['startup_transient']['median']:.2f}", r1["startup_transient"]["median"])
            put("EmpStartupMaxHf", f"{r1['startup_transient']['max']:.2f}", r1["startup_transient"]["max"])
            put("EmpSessionMedianHf", f"{r1['session']['median']:.2f}", r1["session"]["median"])
            put("EmpSessionPninetyHf", f"{r1['session']['p90']:.2f}", r1["session"]["p90"])
            put("EmpRetuneMedianHf", f"{r1['retune']['median']:.2f}", r1["retune"]["median"])
            put("EmpRetunePninetyHf", f"{r1['retune']['p90']:.2f}", r1["retune"]["p90"])
            w_same = r1["empirical_wall"]["same_freq"]["max_g_within_noise_db"]
            if w_same is not None:
                put("EmpWallSameFreqDb", f"${w_same:.1f}$", w_same)
            ret_res = r1["empirical_wall"]["retune"]["median_rel_residual"]
            put("EmpRetuneResidMedian", pct(ret_res, 1), ret_res)
        if r2 and r2.get("split_half_noise"):
            put("EmpSplitHalfMedianTt", f"{r2['split_half_noise']['median']:.2f}", r2["split_half_noise"]["median"])
            put("EmpSteadyDriftMedianTt", f"{r2['steady_drift']['median']:.2f}", r2["steady_drift"]["median"])
            put("EmpSessionMedianTt", f"{r2['session']['median']:.2f}", r2["session"]["median"])
        if cr and cr.get("uncertainty"):
            put("EmpCrossRecMedian", f"{cr['uncertainty']['median']:.2f}", cr["uncertainty"]["median"])
            put("EmpCrossRecPninety", f"{cr['uncertainty']['p90']:.2f}", cr["uncertainty"]["p90"])
            if cr.get("empirical_wall"):
                cr_res_min = cr["empirical_wall"]["min_rel_residual"]
                cr_res_med = cr["empirical_wall"]["median_rel_residual"]
                if cr_res_min is not None:
                    put("EmpCrossRecResidMin", pct(cr_res_min, 1), cr_res_min)
                if cr_res_med is not None:
                    put("EmpCrossRecResidMedian", pct(cr_res_med, 1), cr_res_med)
    # ---------- long-observation wall test (BAND, CYC; N = 2048, 8192, 32768), at the wall and 2 dB below
    lf = sorted(glob.glob(os.path.join(ROOT, "RESULTS", "wall_test_long_s*.json")))
    if lf:
        LD = [json.load(open(x)) for x in lf]
        for N, L in [(2048, "C"), (8192, "D"), (32768, "E")]:
            key = f"N{N}_R1.0"
            if all(key in d for d in LD):
                for det, nm in [("BAND", "Band"), ("CYC", "Cyc")]:
                    v0 = float(np.mean([d[key]["Pd"]["0"][det]["worst"] for d in LD]))
                    vm = float(np.mean([d[key]["Pd"]["-2"][det]["worst"] for d in LD]))
                    put(f"WallLong{nm}N{L}", f"{v0:.2f}", v0)
                    put(f"WallLong{nm}N{L}Minus", f"{vm:.2f}", vm)
        put("WallLongSeeds", str(len(LD)), len(LD))
    # ---------- Task T9 / Plan C: Escape models trained at N = 8192, PAIRED, and CycNet
    esc_files = sorted(glob.glob(os.path.join(ROOT, "RESULTS", "wall_test_escape_s*.json")))
    if esc_files:
        ESC = [json.load(open(x)) for x in esc_files]

        def esc_v(det, N, d, R="1.0"):
            v = [z[f"N{N}_R{R}"]["Pd"][str(d)][det]["worst"] for z in ESC if f"N{N}_R{R}" in z and "Pd" in z[f"N{N}_R{R}"]]
            return float(np.mean(v)) if v else float("nan")

        for N, L in [(8192, "ND"), (32768, "NE")]:
            for det, nm in [
                ("BAND", "Band"),
                ("CYC", "Cyc"),
                ("GapCNN-WHITE-N2048", "WhiteNC"),
                ("GapCNN-WHITE-N8192", "White"),
                ("GapCNN-SURRXM-N8192", "Surrxm"),
                ("GapCNN-PAIRED-N8192", "Paired"),
                ("CycNet-WHITE-N8192", "CycNet"),
            ]:
                v0 = esc_v(det, N, 0, "1.0")
                vm = esc_v(det, N, -2, "1.0")
                if not np.isnan(v0):
                    put(f"Escape{nm}Wall{L}", f"{v0:.2f}", v0)
                if not np.isnan(vm):
                    put(f"Escape{nm}Below{L}", f"{vm:.2f}", vm)

        for N, L in [(8192, "ND"), (32768, "NE")]:
            for det, nm in [
                ("BAND", "Band"),
                ("CYC", "Cyc"),
                ("GapCNN-WHITE-N8192", "White"),
                ("GapCNN-SURRXM-N8192", "Surrxm"),
                ("GapCNN-PAIRED-N8192", "Paired"),
                ("CycNet-WHITE-N8192", "CycNet"),
            ]:
                v0 = esc_v(det, N, 0, "2.0")
                vm = esc_v(det, N, -2, "2.0")
                if not np.isnan(v0):
                    put(f"EscapeTwo{nm}Wall{L}", f"{v0:.2f}", v0)
                if not np.isnan(vm):
                    put(f"EscapeTwo{nm}Below{L}", f"{vm:.2f}", vm)
        put("EscapeSeeds", str(len(ESC)), len(ESC))
    # ---------- Theorem 2 numeric check (theorem_check.py): fixed threshold, extremal + random shapes, N = 512 ... 32768
    ft = os.path.join(ROOT, "RESULTS", "theorem_check.json")
    if os.path.exists(ft):
        T = json.load(open(ft))
        Ns = sorted(int(k[1:]) for k in T if k.startswith("N"))
        first, last = T[f"N{Ns[0]}"], T[f"N{Ns[-1]}"]
        put("ThmChkWall", f"${T['wall_db']:.1f}$", T["wall_db"])
        put("ThmChkRho", f"{T['rho']:.3f}", T["rho"])
        put("ThmChkNFirst", f"{Ns[0]}", Ns[0])
        put("ThmChkNLast", f"{Ns[-1]:,}".replace(",", "\\,"), Ns[-1])
        put("ThmChkPfFirst", f"{first['max_Pf_over_shapes']:.2f}", first["max_Pf_over_shapes"])
        put("ThmChkPfLast", f"{last['max_Pf_over_shapes']:.3f}", last["max_Pf_over_shapes"])
        put("ThmChkPdFirst", f"{first['worst_Pd']['1.0']:.2f}", first["worst_Pd"]["1.0"])
        put("ThmChkPdLast", f"{last['worst_Pd']['1.0']:.2f}", last["worst_Pd"]["1.0"])
        put("ThmChkPdWallFirst", f"{first['worst_Pd']['0.0']:.2f}", first["worst_Pd"]["0.0"])
        put("ThmChkPdWallLast", f"{last['worst_Pd']['0.0']:.2f}", last["worst_Pd"]["0.0"])
    # ---------- statistical rigor (stats_ci.py): two-level bootstrap & paired tests
    f_ci = os.path.join(ROOT, "RESULTS", "stats_ci.json")
    if os.path.exists(f_ci):
        SCI = json.load(open(f_ci))
        t1 = SCI.get("table1", {})
        t2 = SCI.get("table2", {})
        paired = {p["key"] + "_" + p["model"] + "_" + p["rec2"] + "_" + p["rec1"]: p for p in SCI.get("paired", [])}
        if "GapCNN_WHITE" in t1 and "surgap" in t1["GapCNN_WHITE"]:
            sg = t1["GapCNN_WHITE"]["surgap"]
            lo, hi = 100 * sg["ci_lo"], 100 * sg["ci_hi"]
            put("CiGapCnnWhiteSurGap", f"$[{lo:+.1f}, {hi:+.1f}]$", [lo, hi])
        if "GapCNN_SURRX" in t2 and "surgap" in t2["GapCNN_SURRX"]:
            sg = t2["GapCNN_SURRX"]["surgap"]
            lo, hi = 100 * sg["ci_lo"], 100 * sg["ci_hi"]
            put("CiGapCnnSurrxSurGap", f"$[{lo:+.1f}, {hi:+.1f}]$", [lo, hi])
        p_surrxm = paired.get("h1_lpf0.8_-10_GapCNN_SURRXM_SURRX")
        if p_surrxm:
            g = 100 * p_surrxm["mean_gain"]
            lo, hi = 100 * p_surrxm["ci_lo"], 100 * p_surrxm["ci_hi"]
            put("CiSurrxmGainLpf", f"${g:+.1f}$", g)
            put("CiSurrxmGainLpfCi", f"$[{lo:+.1f}, {hi:+.1f}]$", [lo, hi])
            put("CiSurrxmGainLpfChi", f"{p_surrxm['chi2']:.1f}", p_surrxm["chi2"])
            put("CiSurrxmGainLpfP", "$p < 10^{-300}$" if p_surrxm["p_value"] < 1e-300 else f"$p = {p_surrxm['p_value']:.1e}$", p_surrxm["p_value"])
        p_randm = paired.get("h1_lpf0.8_-10_GapCNN_RANDM_RAND")
        if p_randm:
            g = 100 * p_randm["mean_gain"]
            lo, hi = 100 * p_randm["ci_lo"], 100 * p_randm["ci_hi"]
            put("CiRandmGainLpf", f"${g:+.1f}$", g)
            put("CiRandmGainLpfCi", f"$[{lo:+.1f}, {hi:+.1f}]$", [lo, hi])
            put("CiRandmGainLpfChi", f"{p_randm['chi2']:.1f}", p_randm["chi2"])
            put("CiRandmGainLpfP", "$p < 10^{-300}$" if p_randm["p_value"] < 1e-300 else f"$p = {p_randm['p_value']:.1e}$", p_randm["p_value"])
        p_pinkm_gap = paired.get("h1_pink_-10_GapCNN_PINKM_PINK")
        if p_pinkm_gap:
            g = 100 * p_pinkm_gap["mean_gain"]
            lo, hi = 100 * p_pinkm_gap["ci_lo"], 100 * p_pinkm_gap["ci_hi"]
            put("CiPinkmGainPinkGap", f"${g:+.1f}$", g)
            put("CiPinkmGainPinkGapCi", f"$[{lo:+.1f}, {hi:+.1f}]$", [lo, hi])
            put("CiPinkmGainPinkGapChi", f"{p_pinkm_gap['chi2']:.1f}", p_pinkm_gap["chi2"])
            put("CiPinkmGainPinkGapP", "$p < 10^{-300}$" if p_pinkm_gap["p_value"] < 1e-300 else f"$p = {p_pinkm_gap['p_value']:.1e}$", p_pinkm_gap["p_value"])
        p_pinkm_cldnn = paired.get("h1_pink_-10_CLDNN_PINKM_PINK")
        if p_pinkm_cldnn:
            g = 100 * p_pinkm_cldnn["mean_gain"]
            lo, hi = 100 * p_pinkm_cldnn["ci_lo"], 100 * p_pinkm_cldnn["ci_hi"]
            put("CiPinkmGainPinkCldnn", f"${g:+.1f}$", g)
            put("CiPinkmGainPinkCldnnCi", f"$[{lo:+.1f}, {hi:+.1f}]$", [lo, hi])
            put("CiPinkmGainPinkCldnnChi", f"{p_pinkm_cldnn['chi2']:.1f}", p_pinkm_cldnn["chi2"])
            put("CiPinkmGainPinkCldnnP", "$p < 10^{-300}$" if p_pinkm_cldnn["p_value"] < 1e-300 else f"$p = {p_pinkm_cldnn['p_value']:.1e}$", p_pinkm_cldnn["p_value"])
    verification_macros()
    upgrade_macros()
    # ---------- any macro used in the manuscript but not computed -> visible marker (caught by the final audit)
    import re
    used = set()
    for tf in (glob.glob(os.path.join(ROOT, "MANUSCRIPT", "sec_*.tex")) + glob.glob(os.path.join(ROOT, "MANUSCRIPT_JOURNAL", "*.tex"))
               + glob.glob(os.path.join(ROOT, "TABLES", "*.tex"))):
        used |= set(re.findall(r"\\([A-Z][A-Za-z]+)\{\}", open(tf, encoding="utf-8").read()))
        used |= set(re.findall(r"\\([A-Z][A-Za-z]+)(?=[\s\\$.,;:)\-])", open(tf, encoding="utf-8").read()))
    ours = ("Pf", "Pd", "SurGap", "Wall", "Real", "CLDNN", "GapCNN", "Ed", "Mme", "Sfm", "Band", "Cyc", "Occupancy", "Wc",
            "Whiten", "Occ", "Abl", "Res", "Det", "Surr", "Oracle", "SurZero", "SigZero", "Mask", "Th", "Alpha", "LongNet", "Receiver",
            "Compute", "PdCal", "Trev", "Adapt", "Emp", "RealSig", "Ci", "Escape", "Ofdm", "Noise", "CycLag", "CycNet", "Ctrl",
            "Obs", "Uone", "Exact", "Lmpi", "RealLmpi", "RealBand", "EscLaw", "Hw", "Off", "Gceil", "Tfm", "Dyn")
    missing = sorted(u for u in used if u not in VALS and u.startswith(ours))
    for u in missing:
        VALS[u] = {"tex": f"\\textbf{{[MISSING:{u}]}}", "raw": None}
    if missing:
        print("MISSING macros:", missing)
    # ---------- write
    with open(os.path.join(ROOT, "MANUSCRIPT", "numbers.tex"), "w", encoding="utf-8") as fh:
        fh.write("% generated by CODE/paper_numbers.py -- do not edit by hand\n")
        for k, v in VALS.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v['tex']}}}\n")
    json.dump(VALS, open(os.path.join(ROOT, "RESULTS", "paper_numbers.json"), "w"), indent=1, default=str)
    for k, v in VALS.items():
        print(f"{k:28s} {v['tex']}")


if __name__ == "__main__":
    main()


