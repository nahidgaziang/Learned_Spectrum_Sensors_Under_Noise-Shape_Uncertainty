"""Automated checks of every qualitative claim in the manuscript against RESULTS/.
Prints PASS/FAIL per claim and a summary line; exit code 1 if any claim fails.
Each claim names the manuscript sentence it protects."""
import glob, json, os, sys
import numpy as np
from analysis import load_runs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHS = ["CLDNN", "ResNet1D", "CovCNN", "STFTCNN", "GapCNN"]
results = []


def claim(text, ok, detail=""):
    results.append((bool(ok), text, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {text}  {detail}")


def verification_fix_claims():
    """Checks for the corrections of 2026-10-05 (internal notes, internal notes): the corrected statements are
    supported by the data, and the removed statements do not come back."""
    import re
    P = json.load(open(os.path.join(ROOT, "RESULTS", "paper_numbers.json")))
    raw = lambda k: P[k]["raw"] if k in P else float("nan")
    claim("R828D white-net P_f range uses all five architectures", len(raw("TrevPfWhiteRange")) == 5, P["TrevPfWhiteRange"]["tex"])
    claim("Real signals: GAP CNN max P_d equals band-ratio max P_d within 3 points in every combination",
          raw("RealSigCnnBandMaxDiff") <= 0.03, f"{raw('RealSigCnnBandMaxDiff'):.3f}")
    claim("Real signals: NOAA/pager/FRS on R828D not detected by GAP CNN or band ratio (max P_d <= 0.1)",
          raw("RealSigNdMax") <= 0.10, f"{raw('RealSigNdMax'):.3f}")
    tab = open(os.path.join(ROOT, "TABLES", "tab_real_signals.tex"), encoding="utf-8").read()
    body = tab.split("\\midrule")[1].split("\\bottomrule")[0]
    rows = [l for l in body.strip().splitlines() if "&" in l]
    claim("Real-signal table lists all 9 evaluated class/receiver combinations incl. NOAA, FRS/GMRS, APRS",
          len(rows) == 9 and all(s in tab for s in ("NOAA", "FRS/GMRS", "APRS")), f"{len(rows)} rows")
    claim("Signal-like retuning pairs lie below the same-frequency empirical wall",
          raw("EmpRetuneWallDb") < raw("EmpWallSameFreqDb") and raw("EmpRetunePairsWithin") > 0,
          f"{raw('EmpRetunePairsWithin')} pairs, max {raw('EmpRetuneWallDb'):.1f} dB vs {raw('EmpWallSameFreqDb'):.1f} dB")
    claim("CycNet beats 16-lag classical CYC at u=1 dB (>=1.3x) but not at u=2 dB (within 0.02)",
          raw("CycNetOverCycSixteen") >= 1.3 and abs(raw("CycLagSixteenBelowTwo") - raw("EscapeTwoCycNetBelowNE")) <= 0.02,
          f"ratio {raw('CycNetOverCycSixteen'):.2f}; u=2: {raw('CycLagSixteenBelowTwo'):.3f} vs {raw('EscapeTwoCycNetBelowNE'):.3f}")
    claim("Escape is independent of noise synthesis (circ vs lin <= 0.05) and Gaussian surrogates <= 0.07 at/below the wall",
          raw("CtrlSynthMaxDiff") <= 0.05 and raw("CtrlGsurMax") <= 0.07, f"{raw('CtrlSynthMaxDiff'):.3f}, {raw('CtrlGsurMax'):.3f}")
    claim("Roll-off is the stronger factor for real-noise P_f (partial rho roll-off > partial rho kappa)",
          raw("NoisePartialRolloff") > raw("NoisePartialKappa"), f"{raw('NoisePartialRolloff'):.2f} vs {raw('NoisePartialKappa'):.2f}")
    claim("DetectNet replica: largest |Pd(signal)-Pd(surrogate)| over [-20,0] dB <= 3 points", raw("DetMaxAbsDiff") <= 0.03,
          f"{raw('DetMaxAbsDiff'):.3f}")
    J = os.path.join(ROOT, "MANUSCRIPT_JOURNAL")
    if not os.path.isdir(J):          # e.g. the public code repository, which ships no LaTeX sources
        print("[SKIP] manuscript text checks: MANUSCRIPT_JOURNAL/ not present")
        return
    src = "".join(open(os.path.join(J, f), encoding="utf-8").read() for f in os.listdir(J)
                  if f.endswith(".tex") and (f.startswith("j_") or f in ("main.tex", "supplement.tex")) and f != "j_appendix.tex")
    banned = ["published 16QAM detection to within", "throughput collapse", "interference temperature", "1--10",
              "our RTL-SDR hardware", "16--34", "monotonically", "does not sacrifice", "robust sub-wall", "single-core"]
    found = [b for b in banned if b in src]
    claim("Removed unsupported statements are absent from the journal sources", not found, str(found))
    claim("AI-use acknowledgment present (IEEE policy)", "\\section*{Acknowledgment}" in src and "Claude" in src, "")
    kw = re.search(r"\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}", src, re.S)
    nkw = len([k for k in kw.group(1).strip().rstrip(".").split(",") if k.strip()]) if kw else 99
    claim("TCCN: at most five keywords", nkw <= 5, str(nkw))
    pdf = os.path.join(J, "main.pdf")
    if os.path.exists(pdf):
        import pymupdf
        d = pymupdf.open(pdf)
        p0 = d[0].get_text()
        nw = len(p0.split("Abstract")[1].split("Index Terms")[0].split()) if "Index Terms" in p0 else 999
        claim("TCCN: abstract 75-200 words (rendered PDF)", 75 <= nw <= 200, str(nw))
        # page budget: 12 pages after the novelty upgrades, then 10 pages on 2026-10-05 at the author's request
        # (no mandatory overlength charge; TCCN hard limit 13 incl. references)
        claim("TCCN: at most 10 pages (no overlength charge; hard limit 13)", d.page_count <= 10, f"{d.page_count} pages")
        full = "".join(pg.get_text() for pg in d)
        claim("Manuscript is self-contained: no mention of supplementary material", "upplement" not in full, "")
        claim("Manuscript contains the proofs (Appendix)", "APPENDIX" in full.upper() and "Proposition" in full, "")


def upgrade_claims():
    """Checks for the statements added with the 2026-10-05 novelty upgrades (internal notes): theory checks,
    LMPI identification, exact wall, escape law, hardware budget, off-band mechanism, dynamics, transformer."""
    R = os.path.join(ROOT, "RESULTS")
    J = lambda f: json.load(open(os.path.join(R, f))) if os.path.exists(os.path.join(R, f)) else None
    P = json.load(open(os.path.join(R, "paper_numbers.json")))
    raw = lambda k: P[k]["raw"] if k in P else float("nan")
    U = J("u1_theory_check.json")
    if U:
        r = U["results"]
        low = [abs(r[d]["gap_ideal"][i]) / r[d]["se_ideal"][i] for d in ("CYC5", "CYC16", "KAPPALO")
               for i, s in enumerate(r[d]["snr"]) if s <= -16]
        claim("Prop. 3 check: structure detectors' gap not detectable at <= -16 dB (|gap| <= 3 SE)", max(low) <= 3, f"max {max(low):.2f} SE")
        k = raw("UoneKappaRatioRange")
        claim("Prop. 3 check: gap/gamma^2 of kappa_lo roughly constant (max/min <= 1.5)", max(k) / min(k) <= 1.5, f"{min(k):.3f}-{max(k):.3f}")
        so = [abs(r[d]["gap_paper"][i]) / max(r[d]["se_paper"][i], 1e-12) for d in ("LMPI", "BAND") for i in range(len(r[d]["snr"]))
              if r[d]["se_paper"][i] > 0]
        claim("Prop. 3 check: second-order gaps vs power-normalised surrogates <= 0.5 points and <= 2.5 SE",
              raw("UoneSecondOrderMaxAbs") <= 0.005 and max(so) <= 2.5, f"{100 * raw('UoneSecondOrderMaxAbs'):.2f} pts, {max(so):.2f} SE")
        i = int(np.argmax(r["LMPI"]["gap_ideal"]))
        claim("Prop. 3 check: LMPI gap vs un-normalised surrogate peaks near P_d = 0.9", 0.8 <= r["LMPI"]["pd_sig"][i] <= 0.97,
              f"P_d {r['LMPI']['pd_sig'][i]:.3f}")
    X = J("wall_exact.json")
    if X:
        s = X["summary"]
        claim("Thm 3: closed form equals the LP in every case", s["formula_eq_lp"].split("/")[0] == s["formula_eq_lp"].split("/")[1],
              s["formula_eq_lp"])
        claim("Thm 3: infinite wall exactly when p_max/p_min <= R (boundary cases)", s["boundary_ok"].split("/")[0] == s["boundary_ok"].split("/")[1],
              s["boundary_ok"])
        claim("Thm 3: dual margin zero below and positive above the wall", s["margin_sign_ok"], "")
        claim("Thm 3: s = R^2 - 1 is exact for spectra with nulls (RRC)", s["prop2_exact_when_qmin_tiny"] and s["thm1_agree"], "")
        D = X["D"]["N"]; Ns = sorted(D, key=int)
        pf = [D[n]["max_Pf_all"] for n in Ns]; pd1 = [D[n]["Pd"]["1.0"]["worst"] for n in Ns]; pdw = [D[n]["Pd"]["0.0"]["worst"] for n in Ns]
        claim("Thm 3(iii) check: max P_f falls, P_d(+1 dB) rises and P_d at the wall falls with N",
              all(np.diff(pf) < 0) and all(np.diff(pd1) > 0) and all(np.diff(pdw) < 0), f"Pf {pf}, Pd+1 {pd1}, Pd0 {pdw}")
    L = J("lmpi_equivalence.json")
    if L:
        M = L["models"]
        pick = lambda arch, tag="": [m for m in M.values() if m["config"].get("arch") == arch and m["config"].get("recipe") == "WHITE"
                                     and m["config"].get("N") == 128 and m["config"].get("sps") == 8 and (m["config"].get("tag") or "") == tag
                                     and m["config"].get("kind") == "learned"]
        strong = [m for a in ("GapCNN", "CLDNN", "CovCNN") for m in pick(a)]
        claim("LMPI: strong networks rank like the LMPI test (rho >= 0.93, 15 models)",
              len(strong) == 15 and min(m["dstar"]["rho_sig"] for m in strong) >= 0.93, f"min {min(m['dstar']['rho_sig'] for m in strong):.3f}")
        nl, nn, nb = raw("LmpiAgreeNetLmpi"), raw("LmpiAgreeNetNet"), raw("LmpiAgreeNetBand")
        claim("LMPI: net-LMPI agreement within the net-net range and above every net-BAND agreement",
              min(nl) >= min(nn) - 0.01 and max(nl) <= max(nn) + 0.01 and min(nl) > max(nb), f"{min(nl):.3f}-{max(nl):.3f} vs {min(nn):.3f}-{max(nn):.3f}, band max {max(nb):.3f}")
        claim("LMPI: implicit templates closer to the signal PSD than to the band indicator in >= 24 of 25 white models",
              raw("LmpiPcloserCount") >= 24, P["LmpiPcloserCount"]["tex"])
        dev = [abs(raw(f"LmpiPfLpf{x}") - P[f"LmpiGapPfLpf{x}"]["raw"]) if P[f"LmpiGapPfLpf{x}"]["raw"] is not None else 0 for x in ("NA", "NB", "NC")]
        gp = [np.mean([m["h0_lpf0.8"]["pf_model"] for m in M.values() if m["config"].get("kind") == "learned" and m["config"].get("arch") == "GapCNN"
                       and m["config"].get("recipe") == "WHITE" and m["config"].get("N") == n and m["config"].get("sps") == 8
                       and not m["config"].get("tag")]) for n in (128, 512, 2048)]
        lp = [raw(f"LmpiPfLpf{x}") for x in ("NA", "NB", "NC")]
        claim("LMPI: LP .8 false alarms grow with N like the GAP CNN's (|diff| <= 0.06 at each N)",
              all(abs(a - b) <= 0.06 for a, b in zip(lp, gp)) and all(np.diff(lp) > 0), f"LMPI {np.round(lp, 3)} GAP {np.round(gp, 3)}")
        claim("LMPI: no white network exceeds the LMPI P_d by more than 1.5 points", raw("LmpiMaxExcess") <= 0.015, f"{100 * raw('LmpiMaxExcess'):.1f}")
        claim("LMPI: escape networks are not spectral (|rho| <= 0.1, template cos <= 0.4)",
              float(P["LmpiRhoEscapeMax"]["tex"]) <= 0.1 and float(P["LmpiCosPEscapeMax"]["tex"]) <= 0.4, "")
        weak = [m["dstar"]["rho_sig"] for a, t in (("ResNet1D", ""), ("STFTCNN", ""), ("DetectNetExact", "QAM16"), ("Transformer", "")) for m in pick(a, t)]
        claim("LMPI: weaker architectures less LMPI-like than the strong ones", max(weak) < min(m["dstar"]["rho_sig"] for m in strong), f"max weak {max(weak):.3f}")
    tf = [json.load(open(f))["eval"] for f in sorted(glob.glob(os.path.join(R, "runs", "Transformer_WHITE_N128_sps8_s?.json")))]
    if tf:
        claim("Transformer: 5 seeds, |surrogate gap| <= 2 points, pink P_f >= 0.99", len(tf) == 5 and all(abs(e["surrogate_gap"]) <= 0.02 for e in tf)
              and all(e["Pf"]["pink"]["nominal"] >= 0.99 for e in tf), "")
    RL = J("real_signal_lmpi.json")
    if RL:
        c = RL["comparison"]
        claim("Real signals: LMPI replay exact (band counts) and tracks the GAP CNN within 3 points, closer than the band ratio",
              all(v["band_matches_published"] for v in c.values()) and RL["n_captures_checked"] == raw("RealLmpiNcaps")
              and max(v["max_abs_curve_diff_lmpi_cnn"] for v in c.values()) <= 0.03
              and max(v["max_abs_curve_diff_lmpi_cnn"] for v in c.values()) < max(v["max_abs_curve_diff_band_cnn"] for v in c.values()), "")
        claim("Real signals: LMPI misses NOAA/pager/FRS on R828D (max P_d <= 0.1)", raw("RealLmpiNdMax") <= 0.1, f"{raw('RealLmpiNdMax'):.3f}")
    O = J("offband_sweep.json")
    if O:
        s = O["summary"]
        mains = [v["mae"] for k, v in s.items() if k.startswith(("CLDNN", "ResNet1D", "CovCNN", "GapCNN", "DetectNet"))]
        stft = [v["mae"] for k, v in s.items() if k.startswith("STFTCNN")]
        claim("Off-band: overlap rule predicts the networks within MAE <= 3 points; STFT CNN is the worst", max(mains) <= 0.03 and min(stft) > max(mains),
              f"max {max(mains):.3f}, STFT min {min(stft):.3f}")
        claim("Off-band: clearly negative own overlap (< -0.05) keeps P_d at or below the false-alarm level (<= 0.06)",
              raw("OffNegPdMax") <= 0.06 and raw("OffNegN") >= 50, f"{raw('OffNegPdMax'):.3f} over {raw('OffNegN')} detector-cells")
    EL = J("escape_law.json")
    if EL:
        claim("Escape law: existing wall tests predicted within 0.07 and new roll-off tests within 0.06, no fitted parameter",
              float(P["EscLawMaxErr"]["tex"]) <= 0.07 and float(P["EscLawMaxErrC"]["tex"]) <= 0.06, "")
        claim("Escape law: implied kappa within 10 % of the pulse-derived kappa", abs(raw("EscLawKappaImplied") / raw("EscLawKappaFive") - 1) <= 0.10,
              f"{raw('EscLawKappaImplied'):.4f} vs {raw('EscLawKappaFive'):.4f}")
        claim("Escape law: 16 lags need fewer samples than 5 lags", raw("EscLawNescSixteen") < raw("EscLawNescFive"), "")
    H = J("hardware_wall_budget.json")
    if H:
        g90, gm = raw("HwGainNinetyRange"), raw("HwGainMedRange")
        claim("Hardware: at p90 the shape wall lies below the ED wall on both receivers; at the median the sign varies",
              min(g90) > 0 and min(gm) < 0 < max(gm), f"p90 {np.round(g90, 1)}, median {np.round(gm, 1)}")
        claim("Hardware: within-capture power drift small (median <= 0.2 dB) on both receivers",
              raw("HwWithinPowMedHf") <= 0.2 and raw("HwWithinPowMedTt") <= 0.2, "")
    DY = J("dynamics.json")
    if DY:
        runs_ = DY["runs"]
        w = [h for k, h in runs_.items() if "WHITE" in k]
        s_ = [h for k, h in runs_.items() if "WHITE" not in k]
        claim("Dynamics: WHITE reaches rho >= 0.9 with the LMPI test within the first epoch; probe gap <= 3.5 points while the "
              "main evaluation (independent windows) gives |gap(-10 dB)| <= 1.5 points",
              max(raw("DynStepsToLmpi")) <= raw("DynStepsPerEpoch") and raw("DynWhiteGapMax") <= 0.035
              and max(abs(x) for x in raw("DynMainGapTenRange")) <= 1.5,
              f"steps {raw('DynStepsToLmpi')}, probe max {100 * raw('DynWhiteGapMax'):.1f}, main {np.round(raw('DynMainGapTenRange'), 2)}")
        claim("Dynamics: structure recipes learn second-order first (rho >= 0.5 by step 20) and their gap is large at 0 dB, small at -10 dB",
              all(max(x["rho_lmpi"] for x in h if x["step"] <= 20) >= 0.5 for h in s_) and all(h[-1]["gap_0"] > 0.3 for h in s_)
              and all(h[-1]["gap_-10"] < 0.1 for h in s_), "")
    G = J("gaussian_ceiling.json")
    if G:
        claim("MPI efficiency: strong networks lose <= 0.5 dB, weaker ones more", raw("GceilLossStrongMax") <= 0.5 and min(raw("GceilLossWeakRange")) > raw("GceilLossStrongMax"),
              f"{raw('GceilLossStrongMax'):.2f} vs {min(raw('GceilLossWeakRange')):.2f}")


def main():
    df = load_runs()
    df = df[(df.tag == "") & (df.signal == "modulated")]          # claims about the main experiments use untagged modulated runs
    b = df[(df.N == 128) & (df.sps == 8)]
    W = b[(b.kind == "learned") & (b.recipe == "WHITE") & (b.model.isin(ARCHS))]
    g = W.groupby("model")
    # --- A: surrogate test
    claim("All five architectures have 5 WHITE seeds", (g.size() == 5).all() and len(g) == 5, str(g.size().to_dict()))
    claim("Per-seed |surrogate gap| <= 2 points for every WHITE network", (W.surgap.abs() <= 0.02).all(), f"max {W.surgap.abs().max():.4f}")
    # per-SNR: surrogate tracks signal within 4 points at every SNR, every arch (mean over seeds)
    snr_cols = [c for c in W.columns if c.startswith("Pd_-") or c == "Pd_0"]
    worst = 0.0
    for c in snr_cols:
        s = c.replace("Pd_", "PdSur_")
        if s in W:
            worst = max(worst, (g[c].mean() - g[s].mean()).abs().max())
    claim("Surrogate Pd tracks signal Pd within 4 points at every SNR (arch means)", worst <= 0.04, f"worst {worst:.3f}")
    band = b[b.model == "BAND"]; cyc = b[b.model == "CYC"]
    claim("Oracle band ratio is second-order (|gap| <= 1 point)", abs(band.surgap.mean()) <= 0.01, f"{band.surgap.mean():.4f}")
    claim("Cyclostationary detector separates signals from surrogates (gap >= 5 points)", cyc.surgap.mean() >= 0.05, f"{cyc.surgap.mean():.3f}")
    claim("CYC detects little at -10 dB with N=128 (Pd <= 10%)", cyc["Pd_-10"].mean() <= 0.10, f"{cyc['Pd_-10'].mean():.3f}")
    ed = b[b.model == "ED"]
    claim("Networks beat ED at -10 dB (min arch Pd > ED Pd)", g["Pd_-10"].mean().min() > ed["Pd_-10"].mean(), f"{g['Pd_-10'].mean().min():.3f} vs {ed['Pd_-10'].mean():.3f}")
    claim("Band ratio Pd(-10) within the networks' range +-5 points", g["Pd_-10"].mean().min() - 0.05 <= band["Pd_-10"].mean() <= g["Pd_-10"].mean().max() + 0.05,
          f"{band['Pd_-10'].mean():.3f}")
    # --- B: shape not power
    claim("+1 dB power keeps every network's Pf in [3%,7%]", W["Pf_white+1dB"].between(0.03, 0.07).all(), f"[{W['Pf_white+1dB'].min():.3f},{W['Pf_white+1dB'].max():.3f}]")
    claim("-1 dB power keeps every network's Pf in [3%,7%]", W["Pf_white-1dB"].between(0.03, 0.07).all(), f"[{W['Pf_white-1dB'].min():.3f},{W['Pf_white-1dB'].max():.3f}]")
    claim("ED Pf at +1 dB >= 50%", ed["Pf_white+1dB"].mean() >= 0.5, f"{ed['Pf_white+1dB'].mean():.3f}")
    claim("LP 0.8 roll-off raises every network's Pf above 10% (arch means)", (g["Pf_lpf0.8"].mean() > 0.10).all(), str(g["Pf_lpf0.8"].mean().round(3).to_dict()))
    claim("RTL-like 3 dB roll-off raises every network's Pf above 8% (arch means)", (g["Pf_rtl3dB"].mean() > 0.08).all(), str(g["Pf_rtl3dB"].mean().round(3).to_dict()))
    claim("Pink noise gives Pf >= 99% for every WHITE network seed", (W["Pf_pink"] >= 0.99).all(), f"min {W['Pf_pink'].min():.3f}")
    claim("Sign asymmetry: AR+0.1 Pf > 10% and AR-0.1 Pf < 3% (arch means)", (g["Pf_ar+0.1"].mean() > 0.10).all() and (g["Pf_ar-0.1"].mean() < 0.03).all(),
          f"{g['Pf_ar+0.1'].mean().min():.3f} / {g['Pf_ar-0.1'].mean().max():.3f}")
    claim("Worst-case Pd(-10) over receiver set < nominal Pd(-10) for every arch", (g["wcPd_U_receiver_-10"].mean() < g["Pd_-10"].mean()).all())
    claim("Robust thresholds keep realised max Pf <= 7% on the test stream (receiver set)", (W["maxPf_U_receiver"] <= 0.07).all(), f"max {W['maxPf_U_receiver'].max():.3f}")
    mme = b[b.model == "MME"]; sfm = b[b.model == "SFM"]
    claim("MME and SFM are more roll-off-sensitive than the networks (LP 0.8)", min(mme["Pf_lpf0.8"].mean(), sfm["Pf_lpf0.8"].mean()) > g["Pf_lpf0.8"].mean().max())
    claim("CYC Pf under LP 0.8 and RTL stays <= 10%", cyc["Pf_lpf0.8"].mean() <= 0.10 and cyc["Pf_rtl3dB"].mean() <= 0.10)
    # --- D: recipes
    def R(arch, rec, col):
        d = b[(b.model == arch) & (b.recipe == rec)]
        return d[col].mean(), len(d)
    pp, n1 = R("CLDNN", "PINK", "Pf_pink"); pl, _ = R("CLDNN", "PINK", "Pf_lpf0.8"); pr, _ = R("CLDNN", "PINK", "Pf_rtl3dB")
    claim("PINK recipe removes pink false alarms (CLDNN, <= 5%)", pp <= 0.05, f"{pp:.3f} (n={n1})")
    claim("PINK recipe leaves roll-off false alarms above nominal (CLDNN LP.8 or RTL > 7%)", max(pl, pr) > 0.07, f"LP.8 {pl:.3f} RTL {pr:.3f}")
    rp, n2 = R("CLDNN", "RAND", "Pf_pink")
    claim("RAND recipe does not remove pink false alarms (CLDNN >= 50%)", rp >= 0.5, f"{rp:.3f} (n={n2})")
    for rec in ["PINK", "RAND"]:
        a, _ = R("CLDNN", rec, "Pd_-10"); w, _ = R("CLDNN", "WHITE", "Pd_-10")
        claim(f"{rec} recipe costs nominal sensitivity (CLDNN Pd(-10) lower than WHITE)", a < w, f"{a:.3f} vs {w:.3f}")
    sg, n3 = R("GapCNN", "SURR", "surgap")
    claim("SURR recipe gives a positive surrogate gap (GapCNN >= 2 points)", sg >= 0.02, f"{sg:.3f} (n={n3})")
    # --- D: masking (signal in the colours the negatives contain) and matched recipes
    for arch in ["CLDNN", "GapCNN"]:
        v, n = R(arch, "PINK", "PdIn_pink_-10")
        claim(f"PINK masks: {arch} Pd(-10) with the signal in pink noise <= 5%", v <= 0.05, f"{v:.3f} (n={n})")
    v, n = R("GapCNN", "RAND", "PdIn_lpf0.8_-10")
    claim("RAND masks: GapCNN Pd(-10) in LP .8 noise <= 5%", v <= 0.05, f"{v:.3f} (n={n})")
    for N in [128, 2048]:
        d = df[(df.N == N) & (df.sps == 8) & (df.model == "GapCNN") & (df.recipe == "SURRX")]
        if len(d):
            claim(f"SURRX masks (N={N}): Pd(-10) in LP .8 < half of Pd(-10) in white", d["PdIn_lpf0.8_-10"].mean() < 0.5 * d["Pd_-10"].mean(),
                  f"{d['PdIn_lpf0.8_-10'].mean():.3f} vs {d['Pd_-10'].mean():.3f}")
    sx, _ = R("GapCNN", "SURRX", "PdIn_lpf0.8_-10"); sxm, nm_ = R("GapCNN", "SURRXM", "PdIn_lpf0.8_-10")
    if nm_:
        claim("Matching removes most masking: SURRXM Pd(-10) in LP .8 >= 2x SURRX (N=128)", sxm >= 2 * sx, f"{sxm:.3f} vs {sx:.3f} (n={nm_})")
        pfp, _ = R("GapCNN", "SURRXM", "Pf_pink")
        claim("SURRXM false-alarms on pink noise (outside its mixture) >= 10%", pfp >= 0.10, f"{pfp:.3f}")
    # colour-calibrated detection (threshold from the colour's own noise; separates lost information from threshold effects)
    fa = os.path.join(ROOT, "RESULTS", "alpha_sensitivity.json")
    if os.path.exists(fa):
        import pandas as pd
        A = pd.DataFrame(json.load(open(fa)))

        def av(model, recipe, alpha, col):
            d = A[(A.model == model) & (A.recipe == recipe) & (np.isclose(A.alpha, alpha))]
            return float(d[col].iloc[0]) if len(d) else float("nan")
        w = av("GapCNN", "WHITE", 0.05, "PdCal_lpf0.8_-10"); r_ = av("GapCNN", "RAND", 0.05, "PdCal_lpf0.8_-10")
        rm = av("GapCNN", "RANDM", 0.05, "PdCal_lpf0.8_-10"); sx_ = av("GapCNN", "SURRX", 0.05, "PdCal_lpf0.8_-10")
        sxm_ = av("GapCNN", "SURRXM", 0.05, "PdCal_lpf0.8_-10")
        claim("Lost information: colour-calibrated Pd(-10) in LP .8, RAND <= 0.5 x WHITE (GapCNN)", r_ <= 0.5 * w, f"{r_:.3f} vs {w:.3f}")
        claim("Matching restores it: RANDM >= 3 x RAND and SURRXM >= 2 x SURRX (colour-calibrated, LP .8)", rm >= 3 * r_ and sxm_ >= 2 * sx_,
              f"RANDM {rm:.3f} vs {r_:.3f}; SURRXM {sxm_:.3f} vs {sx_:.3f}")
        pmax = A[np.isclose(A.alpha, 0.05)]["PdCal_pink_-10"].max()
        claim("Pink at -10 dB is intrinsically hard: no detector exceeds 25% with a pink-calibrated threshold", pmax <= 0.25, f"max {pmax:.3f}")
        for al in (0.01, 0.10):
            gaps = [abs(av(a, "WHITE", al, "surgap")) for a in ARCHS]
            lps = [av(a, "WHITE", al, "Pf_lpf0.8") for a in ARCHS]
            claim(f"alpha={al}: every WHITE network |surrogate gap| <= 2 points and LP .8 Pf >= 3 alpha", max(gaps) <= 0.02 and min(lps) >= 3 * al,
                  f"max gap {max(gaps):.3f}, min LP.8 {min(lps):.3f}")
    wbs = {(a, r): R(a, r, "wcPd_U_broad_-10")[0] for a in ["CLDNN", "GapCNN"]
           for r in ["WHITE", "PINK", "PINKM", "RAND", "RANDM", "SURR", "SURRX", "SURRXM"] if R(a, r, "wcPd_U_broad_-10")[1]}
    claim("No recipe exceeds 10% worst-case Pd(-10) over the broad set (N=128)", max(wbs.values()) <= 0.10, f"max {max(wbs.values()):.3f}")
    # --- E: robust calibration over the receiver set scales with N for the white-trained network
    wcN = [df[(df.N == N) & (df.sps == 8) & (df.model == "GapCNN") & (df.recipe == "WHITE")]["wcPd_U_receiver_-10"].mean() for N in [128, 512, 2048]]
    claim("Receiver-set calibration: WHITE GapCNN worst-case Pd(-10) grows with N and >= 0.9 at N=2048",
          wcN[0] < wcN[1] < wcN[2] and wcN[2] >= 0.9, f"{np.round(wcN,3)}")
    frw = os.path.join(ROOT, "RESULTS", "receiver_wall_check.json")
    if os.path.exists(frw):
        mn = json.load(open(frw))["min_rel_residual"]
        claim("Receiver set admits no exact wall pair (min relative residual of k c0 - g p - c1 >= 1%)", mn >= 0.01, f"{mn:.4f}")
    # --- E: high-SNR structure use of the surrogate-trained network decreases surrogate detection with N
    sz = [df[(df.N == N) & (df.sps == 8) & (df.model == "GapCNN") & (df.recipe == "SURRX")]["PdSur_0"].mean() for N in [128, 512, 2048]]
    wz = df[(df.sps == 8) & (df.model == "GapCNN") & (df.recipe == "WHITE")]["PdSur_0"].min()
    claim("SURRX surrogate detection at 0 dB falls with N (<= 10% at N=2048); WHITE stays >= 99%", sz[0] > sz[1] > sz[2] and sz[2] <= 0.10 and wz >= 0.99,
          f"{np.round(sz,3)}; WHITE min {wz:.3f}")
    # --- E: mechanism checks (tagged runs)
    full = load_runs()
    nn_ = full[(full.tag == "nonorm") & (full.model == "GapCNN")]
    if len(nn_):
        claim("Without power normalisation, Pf at +1 dB >= 15%", nn_["Pf_white+1dB"].mean() >= 0.15, f"{nn_['Pf_white+1dB'].mean():.3f} (n={len(nn_)})")
    for tg in ["ntrain20k", "ntrain180k"]:
        t = full[(full.tag == tg) & (full.model == "GapCNN")]
        if len(t):
            claim(f"Training-set size ({tg}) leaves |surrogate gap| <= 2 points", abs(t.surgap.mean()) <= 0.02, f"{t.surgap.mean():+.4f} (n={len(t)})")
    es = full[(full.tag == "es") & (full.model == "ResNet1D")]
    if len(es):
        claim("ResNet-1D with best-validation selection: LP .8 Pf >= 20% and |gap| <= 2 points",
              es["Pf_lpf0.8"].mean() >= 0.20 and abs(es.surgap.mean()) <= 0.02, f"LP.8 {es['Pf_lpf0.8'].mean():.3f}, gap {es.surgap.mean():+.4f} (n={len(es)})")
    dn = full[(full.tag == "QAM16") & (full.model == "DetectNetExact")]
    if len(dn):
        claim("DetectNet replica: |surrogate gap| <= 2 points and LP .8 Pf >= 15%", abs(dn.surgap.mean()) <= 0.02 and dn["Pf_lpf0.8"].mean() >= 0.15,
              f"gap {dn.surgap.mean():+.4f}, LP.8 {dn['Pf_lpf0.8'].mean():.3f} (n={len(dn)})")
    scale = full[(full.tag == "scale4x_600k") & (full.model == "GapCNN4X")]
    if len(scale):
        claim("T10: 4x capacity (1.84M params) and 10x dataset (600k) GapCNN: |surrogate gap| <= 1.0 points and LP .8 Pf >= 25%",
              abs(scale.surgap.mean()) <= 0.01 and scale["Pf_lpf0.8"].mean() >= 0.25,
              f"gap {scale.surgap.mean():+.4f}, LP.8 {scale['Pf_lpf0.8'].mean():.3f} (n={len(scale)})")
    # --- integrity re-runs after the 2026-10-02 power cut / driver events
    for rid, orig_id, col, tol in [("GapCNN_SURR_N2048_sps8_s0_rerun", "GapCNN_SURR_N2048_sps8_s0", "Pf_lpf0.8", 0.10),
                                   ("GapCNN_SURRX_N2048_sps8_s1_rerun", "GapCNN_SURRX_N2048_sps8_s1", "Pd_-10", 0.02)]:
        a = full[full.id == rid]; o = full[full.id == orig_id]
        if len(a) and len(o):
            claim(f"Re-run {rid}: {col} within {tol:.2f} of the original", abs(float(a[col].iloc[0]) - float(o[col].iloc[0])) <= tol,
                  f"{float(a[col].iloc[0]):.3f} vs {float(o[col].iloc[0]):.3f}")
    fr = os.path.join(ROOT, "RESULTS", "real_noise_v2_repro_check.json")
    if os.path.exists(fr):
        rc = json.load(open(fr))
        claim("Real-noise v2 replay (first 40 recordings) reproduces every stored value exactly", rc["max_abs_diff_all"] == 0.0 and rc["max_threshold_abs_diff"] == 0.0,
              f"max diff {rc['max_abs_diff_all']}, thr diff {rc['max_threshold_abs_diff']}")
    # --- whitening (synthetic) and the per-bin whitening limit
    wf = sorted(glob.glob(os.path.join(ROOT, "RESULTS", "runs", "wht_*.json")))
    if wf:
        WH = [json.load(open(x))["whitening"] for x in wf]
        sm = max(w[f"K1024_cos_{s}"]["Pf"] for w in WH for s in ["white", "rtl3dB", "ar+0.2", "pink", "tilt1.0+"])
        lp = float(np.mean([w["K1024_cos_lpf0.8"]["Pf"] for w in WH]))
        claim("Long-reference whitening (K=1024): Pf <= 8% on smooth shapes, LP .8 stays >= 10%", sm <= 0.08 and lp >= 0.10, f"smooth max {sm:.3f}, LP.8 mean {lp:.3f}")
    fl = os.path.join(ROOT, "RESULTS", "whitening_probe_lpf.json")
    if os.path.exists(fl):
        L8 = json.load(open(fl))["lpf0.8"]
        claim("Exact-PSD per-bin whitening: LP .8 linear-filter noise >= 10%, circulant twin <= 6%",
              L8["whitened_linear"] >= 0.10 and L8["whitened_circulant"] <= 0.06, f"{L8['whitened_linear']:.3f} vs {L8['whitened_circulant']:.3f}")
    # --- real noise (v3 main, v2 sensitivity)
    fv3 = os.path.join(ROOT, "RESULTS", "real_noise_v3_N128.json")
    fv2 = os.path.join(ROOT, "RESULTS", "real_noise_v2_N128.json")
    f = fv3 if os.path.exists(fv3) else fv2
    if os.path.exists(f):
        R0 = json.load(open(f))["recordings"]
        live = [v for v in R0.values() if v["group"] == "live_rtl" and not v["shape"]["narrowband"]]
        ctrl = [v for v in R0.values() if v["group"] == "unlabelled"]

        def m(rs, field, pre):
            ks = [k for k in rs[0][field] if k.startswith(pre)]
            return float(np.mean([np.mean([r[field][k] for k in ks]) for r in rs]))
        lv = [m(live, "Pf", f"{a}-WHITE") for a in ARCHS]
        ct = [m(ctrl, "Pf", f"{a}-WHITE") for a in ARCHS]
        wb = [m(live, "Pf_wbig", f"{a}-WHITE") for a in ARCHS]
        w16 = [m(live, "Pf_w16", f"{a}-WHITE") for a in ARCHS]
        claim("Real RTL noise: every WHITE network Pf > 10% (nominal threshold)", min(lv) > 0.10, f"{np.round(lv,3)}")
        claim("Flat control files: every WHITE network Pf <= 8%", max(ct) <= 0.08, f"{np.round(ct,3)}")
        claim("Long-reference whitening (K_big, cosine fit) brings every WHITE network to Pf <= 10% on real noise", max(wb) <= 0.10, f"{np.round(wb,3)}")
        claim("Short-reference whitening (K=16) in v3 achieves Pf <= 8% (transient-free)", max(w16) <= 0.08, f"{np.round(w16,3)}")
        claim("Cyclostationary detector stays <= 8% on real noise", m(live, "Pf", "CYC") <= 0.08, f"{m(live, 'Pf', 'CYC'):.3f}")
        claim("Surrogate-trained network masks on real noise: Pd(-10) <= 50% at its nominal threshold",
              m(live, "Pd_m10_nominal", "GapCNN-SURR") <= 0.50, f"{m(live, 'Pd_m10_nominal', 'GapCNN-SURR'):.3f}")
        from scipy.stats import spearmanr
        xs = [r["shape"]["edge_rolloff_db"] for r in live + ctrl]
        ys = [np.mean([v for k, v in r["Pf"].items() if any(k.startswith(a + "-WHITE") for a in ARCHS)]) for r in live + ctrl]
        rho = spearmanr(xs, ys).statistic
        claim("Dose-response: Spearman(roll-off, Pf) >= 0.6 across recordings", rho >= 0.6, f"rho={rho:.3f}")
        if os.path.exists(fv3) and os.path.exists(fv2):
            R2 = json.load(open(fv2))["recordings"]
            live2 = [v for v in R2.values() if v["group"] == "live_rtl" and not v["shape"]["narrowband"]]
            wb2 = [m(live2, "Pf_wbig", f"{a}-WHITE") for a in ARCHS]
            w16_v2 = [m(live2, "Pf_w16", f"{a}-WHITE") for a in ARCHS]
            claim("Short-reference whitening (K=16) in v2 leaves Pf above 8% (transient effect)", max(w16_v2) > 0.08, f"{np.round(w16_v2,3)}")
            claim("Transient-free reference improves whitening: v3 whitened Pf <= v2 whitened Pf", max(wb) <= max(wb2), f"{max(wb):.3f} vs {max(wb2):.3f}")
            claim("Real noise v3 evaluated for 131 clean recordings and 10 controls", len(live) == 131 and len(ctrl) == 10, f"{len(live)} live, {len(ctrl)} ctrl")
    else:
        claim("real-noise results present", False, "real-noise JSON missing")
    fo = os.path.join(ROOT, "RESULTS", "whitening_probe_oracle.json")
    if os.path.exists(fo):
        O = json.load(open(fo))["results"]
        liveo = [v for k, v in O.items() if not k.startswith("NOISE")]
        orc = [v["oracle-raw"]["GapCNN"] for v in liveo]
        none = [v["none"]["GapCNN"] for v in liveo]
        claim("Oracle whitening (test windows' own PSD) brings real-noise Pf to <= 7% (second-order explanation)", max(orc) <= 0.07,
              f"none {np.round(none,3)} -> oracle {np.round(orc,3)}")
    # --- second receiver: TrevTron noise (R828D tuner)
    ftt = os.path.join(ROOT, "RESULTS", "real_noise_trevtron_N128.json")
    if os.path.exists(ftt):
        Rtt = json.load(open(ftt))["recordings"]
        cl_tt = [v for v in Rtt.values() if not v["shape"]["narrowband"]]
        ro_tt = [v["shape"]["edge_rolloff_db"] for v in cl_tt]
        claim("TrevTron noise: 74 clean recordings (26 narrowband screened), roll-off within 2.3--2.6 dB",
              len(cl_tt) == 74 and min(ro_tt) >= 2.3 and max(ro_tt) <= 2.6,
              f"{len(cl_tt)} clean, roll-off [{min(ro_tt):.2f}, {max(ro_tt):.2f}]")
        pf_tt = [np.mean([r["Pf"][k] for k in r["Pf"] if k.startswith("GapCNN-WHITE")]) for r in cl_tt]
        pfw_tt = [np.mean([r["Pf_wbig"][k] for k in r["Pf_wbig"] if k.startswith("GapCNN-WHITE")]) for r in cl_tt]
        claim("TrevTron noise: GapCNN-WHITE nominal Pf in [15%,20%], whitened Pf <= 5.5%",
              0.15 <= np.mean(pf_tt) <= 0.20 and np.mean(pfw_tt) <= 0.055,
              f"nominal {np.mean(pf_tt):.3f}, whitened {np.mean(pfw_tt):.3f}")
    f_ov = os.path.join(ROOT, "RESULTS", "real_noise_overlap.json")
    if os.path.exists(f_ov):
        ov = json.load(open(f_ov))
        claim("TrevTron noise has zero SHA-256 overlap with HF composite dataset",
              ov["sha256_overlap_count"] == 0, f"overlap {ov['sha256_overlap_count']}")
    # --- adaptive classical baselines (AMF / ACE / Kelly / ED-ref)
    fa1 = os.path.join(ROOT, "RESULTS", "adaptive_baselines.json")
    fa2 = os.path.join(ROOT, "RESULTS", "adaptive_baselines_trevtron.json")
    if os.path.exists(fa1) and os.path.exists(fa2):
        ad1 = json.load(open(fa1))["recorded_summary"]
        ad2 = json.load(open(fa2))["recorded_summary"]
        claim("Adaptive baseline ACE achieves Pf <= 6% on average on both receivers",
              ad1["ACE"]["Pf_mean"] <= 0.06 and ad2["ACE"]["Pf_mean"] <= 0.06,
              f"HF {ad1['ACE']['Pf_mean']:.3f}, Trev {ad2['ACE']['Pf_mean']:.3f}")
        claim("Adaptive baseline ACE max Pf <= 15% on HF and <= 8% on TrevTron",
              ad1["ACE"]["Pf_max"] <= 0.15 and ad2["ACE"]["Pf_max"] <= 0.08,
              f"HF max {ad1['ACE']['Pf_max']:.3f}, Trev max {ad2['ACE']['Pf_max']:.3f}")
        claim("Adaptive baseline AMF exceeds 10% average Pf on HF composite noise",
              ad1["AMF"]["Pf_mean"] > 0.10, f"HF AMF {ad1['AMF']['Pf_mean']:.3f}")
    # --- wall test
    files = sorted(glob.glob(os.path.join(ROOT, "RESULTS", "wall_test_s*.json")))
    if files:
        data = [json.load(open(x)) for x in files]

        def wc(det, N, d):
            v = [z[f"N{N}_R1.0_{det}"]["Pd"][str(d)]["worst"] for z in data if f"N{N}_R1.0_{det}" in z]
            return float(np.mean(v)) if len(v) == len(data) else float("nan")      # every seed required
        claim("Wall test has 3 seeds", len(data) == 3, str(len(data)))

        def allle(vals, lim):            # NaN (missing seed or N) fails explicitly; max() would silently skip it
            return (not np.isnan(vals).any()) and max(vals) <= lim
        for det in ["GapCNN-WHITE", "CLDNN-WHITE", "BAND", "MME", "SFM"]:
            Ns = [128] if det == "CLDNN-WHITE" else [128, 512, 2048]
            below = [wc(det, N, -2) for N in Ns]
            at = [wc(det, N, 0) for N in Ns]
            claim(f"{det}: worst-case Pd <= 8% at 2 dB below the wall and <= 8% at the wall, N={Ns}",
                  allle(below, 0.08) and allle(at, 0.08), f"-2 dB {np.round(below,3)}, 0 dB {np.round(at,3)}")
        up = [wc("GapCNN-WHITE", N, 2) for N in [128, 512, 2048]]
        claim("Above the wall (+2 dB) the white-trained CNN improves with N", (not np.isnan(up).any()) and up[0] < up[1] < up[2], f"{np.round(up,3)}")
        for det in ["GapCNN-SURR", "GapCNN-SURRX", "GapCNN-SURRXM"]:
            below = [wc(det, N, -2) for N in [128, 512, 2048]]
            at = [wc(det, N, 0) for N in [128, 512, 2048]]
            claim(f"{det}: worst-case Pd <= 8% at 2 dB below the wall for every N (does not escape)", allle(below, 0.08),
                  f"-2 dB {np.round(below,3)}, 0 dB {np.round(at,3)}")
            claim(f"{det}: worst-case Pd at the wall <= 12% for every N", allle(at, 0.12), f"0 dB {np.round(at,3)}")
        def wc2(det, N, d):
            v = [z[f"N{N}_R2.0_{det}"]["Pd"][str(d)]["worst"] for z in data if f"N{N}_R2.0_{det}" in z]
            return float(np.mean(v)) if len(v) == len(data) else float("nan")
        s2 = [wc2("GapCNN-SURRXM", N, 0) for N in [128, 512, 2048]]
        w2 = [wc2("GapCNN-WHITE", N, 0) for N in [128, 512, 2048]]
        claim("u=2 dB wall: SURRXM worst-case Pd rises with N (>= 0.3 at N=2048) while WHITE stays <= 8%",
              (not np.isnan(s2).any()) and s2[0] < s2[1] < s2[2] and s2[2] >= 0.3 and allle(w2, 0.08), f"SURRXM {np.round(s2,3)}, WHITE {np.round(w2,3)}")
        bm = max(wc(det, N, -2) if Rd == 1 else wc2(det, N, -2) for det in ["GapCNN-SURRXM"] for N in [128, 512, 2048] for Rd in [1, 2])
        claim("SURRXM: worst-case Pd two dB below either wall <= 3%", bm <= 0.03, f"{bm:.3f}")
        pfm = max(max(z[k]["Pf"].values()) for z in data for k in z)
        claim("Wall test: realised max Pf on the test stream <= 7%", pfm <= 0.07, f"{pfm:.3f}")
    else:
        claim("wall-test results present", False, "RESULTS/wall_test_s*.json missing")
    fj = os.path.join(ROOT, "RESULTS", "wall_vs_theory_summary.json")
    if os.path.exists(fj):
        J = json.load(open(fj))
        claim("J2: worst-case Pd at or below the constructed wall <= 7% in all 24 (sps, u, N) configurations",
              J["n_configs"] == 24 and J["max_Pd_at_or_below_wall"] <= 0.07, f"n={J['n_configs']}, max {J['max_Pd_at_or_below_wall']:.3f}")
        claim("J2: the 50% crossing approaches the wall with N (all cases) and is <= 1 dB above it at N=32768",
              J["delta50_shrinks_with_N_all"] and J["delta50_max_N32768"] <= 1.0, f"max {J['delta50_max_N32768']:.2f} dB")
    lnf = sorted(glob.glob(os.path.join(ROOT, "RESULTS", "wall_test_long_nets_s*.json")))
    if lnf:
        LN = [json.load(open(x)) for x in lnf]

        def lnv(det, N, d, R="1.0"):
            v = [z[f"N{N}_R{R}"]["Pd"][str(d)][det]["worst"] for z in LN if f"N{N}_R{R}" in z]
            return float(np.mean(v)) if len(v) == 3 else float("nan")
        s_at = [lnv("GapCNN-SURRXM", N, 0) for N in [2048, 8192, 32768]]
        claim("J5: SURRXM (trained at N=2048) at the u=1 dB wall rises with N on longer windows and >= 0.3 at N=32768",
              (not np.isnan(s_at).any()) and s_at[0] < s_at[1] < s_at[2] and s_at[2] >= 0.3, f"{np.round(s_at,3)}")
        below = [lnv("GapCNN-SURRXM", N, -2, R) for N in [2048, 8192, 32768] for R in ["1.0", "2.0"]]
        claim("J5: SURRXM two dB below either wall <= 3% at every length (does not escape below the wall)",
              (not np.isnan(below).any()) and max(below) <= 0.03, f"max {np.nanmax(below):.3f}")
        w_at = [lnv("GapCNN-WHITE", N, 0, R) for N in [2048, 8192, 32768] for R in ["1.0", "2.0"]]
        claim("J5: white-trained CNN stays <= 10% at both walls at every length", (not np.isnan(w_at).any()) and max(w_at) <= 0.10,
              f"max {np.nanmax(w_at):.3f}")
    lfs = sorted(glob.glob(os.path.join(ROOT, "RESULTS", "wall_test_long_s*.json")))
    if lfs:
        LD = [json.load(open(x)) for x in lfs]
        Ns = [2048, 8192, 32768]
        ok = all(f"N{N}_R1.0" in z for z in LD for N in Ns)
        claim("Long wall test complete (3 seeds x N=2048, 8192, 32768)", ok and len(LD) == 3, f"{len(LD)} seeds")
        if ok:
            band = [np.mean([z[f"N{N}_R1.0"]["Pd"]["0"]["BAND"]["worst"] for z in LD]) for N in Ns]
            cycv = [np.mean([z[f"N{N}_R1.0"]["Pd"]["0"]["CYC"]["worst"] for z in LD]) for N in Ns]
            claim("BAND stays <= 8% at the wall up to N=32768", max(band) <= 0.08, f"{np.round(band,3)}")
            claim("CYC escapes: worst-case Pd at the wall grows with N and >= 0.4 at N=32768", cycv[0] < cycv[1] < cycv[2] and cycv[2] >= 0.4, f"{np.round(cycv,3)}")
    # Task T9 / Plan C: Escape below the noise shape wall (wall_test_escape)
    esc_files = sorted(glob.glob(os.path.join(ROOT, "RESULTS", "wall_test_escape_s*.json")))
    if esc_files:
        ESC = [json.load(open(x)) for x in esc_files]

        def esc_m(det, N, d, R="1.0"):
            vals = [z[f"N{N}_R{R}"]["Pd"][str(d)][det]["worst"] for z in ESC if f"N{N}_R{R}" in z and "Pd" in z[f"N{N}_R{R}"]]
            return float(np.mean(vals)) if len(vals) > 0 else float("nan")

        w_below = [esc_m("GapCNN-WHITE-N8192", N, -2, R) for N in [8192, 32768] for R in ["1.0", "2.0"]]
        w_wall = [esc_m("GapCNN-WHITE-N8192", N, 0, R) for N in [8192, 32768] for R in ["1.0", "2.0"]]
        if not np.isnan(w_below).any() and not np.isnan(w_wall).any():
            claim("T9: GapCNN-WHITE (trained at N=8192) worst-case Pd <= 5% at and below the wall",
                  allle(w_below, 0.05) and allle(w_wall, 0.05), f"below {np.round(w_below,3)}, wall {np.round(w_wall,3)}")

        p_below_1 = esc_m("GapCNN-PAIRED-N8192", 32768, -2, "1.0")
        p_below_2 = esc_m("GapCNN-PAIRED-N8192", 32768, -2, "2.0")
        if not np.isnan(p_below_1) and not np.isnan(p_below_2):
            claim("T9: GapCNN-PAIRED escapes below the wall (Pd >= 10% at -2 dB for u=1 and u=2 dB at N=32768)",
                  p_below_1 >= 0.10 and p_below_2 >= 0.35, f"u=1 {p_below_1:.3f}, u=2 {p_below_2:.3f}")

        c_below_1 = esc_m("CycNet-WHITE-N8192", 32768, -2, "1.0")
        cyc_below_1 = esc_m("CYC", 32768, -2, "1.0")
        if not np.isnan(c_below_1) and not np.isnan(cyc_below_1):
            claim("T9: CycNet escapes below the wall (Pd >= 50% at -2 dB, beating classical CYC by > 1.5x at N=32768)",
                  c_below_1 >= 0.50 and c_below_1 >= 1.5 * cyc_below_1, f"CycNet {c_below_1:.3f} vs CYC {cyc_below_1:.3f}")
    # Theorem 2 numeric check (theorem_check.py)
    ft = os.path.join(ROOT, "RESULTS", "theorem_check.json")
    if not os.path.exists(ft):
        claim("Theorem 2 numeric check present", False, "RESULTS/theorem_check.json missing")
    else:
        T = json.load(open(ft))
        Ns = sorted(int(k[1:]) for k in T if k.startswith("N"))
        pf = np.array([T[f"N{N}"]["max_Pf_over_shapes"] for N in Ns])
        pd1 = np.array([T[f"N{N}"]["worst_Pd"]["1.0"] for N in Ns])
        pd0 = np.array([T[f"N{N}"]["worst_Pd"]["0.0"] for N in Ns])
        pdm = np.array([T[f"N{N}"]["worst_Pd"]["-1.0"] for N in Ns])
        claim("Thm 2 check: max Pf over shapes decreases with N and <= 0.01 at the largest N",
              bool(np.all(np.diff(pf) < 0)) and pf[-1] <= 0.01, f"N={Ns} {np.round(pf, 3)}")
        claim("Thm 2 check: worst Pd at wall+1 dB increases with N and >= 0.99 at the largest N",
              bool(np.all(np.diff(pd1) > 0)) and pd1[-1] >= 0.99, f"{np.round(pd1, 3)}")
        claim("Thm 2 check: worst Pd at and 1 dB below the wall <= 0.05 at the largest N",
              max(pd0[-1], pdm[-1]) <= 0.05, f"wall {pd0[-1]:.3f}, -1 dB {pdm[-1]:.3f}")
    # Second receiver: TrevTron R828D tuner (T4)
    ftt = os.path.join(ROOT, "RESULTS", "real_noise_trevtron_N128.json")
    if not os.path.exists(ftt):
        claim("Second receiver (TrevTron R828D) present", False, "RESULTS/real_noise_trevtron_N128.json missing")
    else:
        Ttt = json.load(open(ftt))["recordings"]
        clean_tt = [v for v in Ttt.values() if not v["shape"]["narrowband"]]
        claim("TrevTron noise: 100 recordings evaluated, >= 70 clean after narrowband screening",
              len(Ttt) == 100 and len(clean_tt) >= 70, f"{len(clean_tt)}/100 clean")
        pf_tt_w = np.mean([np.mean([r["Pf"]["CLDNN-WHITE-s0"], r["Pf"]["GapCNN-WHITE-s0"]]) for r in clean_tt])
        pf_tt_wb = np.mean([np.mean([r["Pf_wbig"]["CLDNN-WHITE-s0"], r["Pf_wbig"]["GapCNN-WHITE-s0"]]) for r in clean_tt])
        claim("TrevTron unwhitened white-trained networks Pf in 15-20% (matches ~2.4 dB roll-off)",
              0.15 <= pf_tt_w <= 0.20, f"{100*pf_tt_w:.1f}%")
        claim("TrevTron whitened networks Pf restored to <= 5.5%",
              pf_tt_wb <= 0.055, f"{100*pf_tt_wb:.1f}%")
    # Overlap check (T4)
    f_ov = os.path.join(ROOT, "RESULTS", "real_noise_overlap.json")
    if os.path.exists(f_ov):
        ov_data = json.load(open(f_ov))
        claim("Zero SHA-256 sample overlap between TrevTron and HF composite noise",
              ov_data["sha256_overlap_count"] == 0, f"{ov_data['sha256_overlap_count']} overlaps")
    # Adaptive baselines (T5)
    f_ad1 = os.path.join(ROOT, "RESULTS", "adaptive_baselines.json")
    f_ad2 = os.path.join(ROOT, "RESULTS", "adaptive_baselines_trevtron.json")
    if os.path.exists(f_ad1) and os.path.exists(f_ad2):
        ad1 = json.load(open(f_ad1))["recorded_summary"]
        ad2 = json.load(open(f_ad2))["recorded_summary"]
        claim("Adaptive ACE holds Pf <= 6.0% on both receivers",
              ad1["ACE"]["Pf_mean"] <= 0.060 and ad2["ACE"]["Pf_mean"] <= 0.060,
              f"HF: {100*ad1['ACE']['Pf_mean']:.1f}%, TrevTron: {100*ad2['ACE']['Pf_mean']:.1f}%")
        claim("Adaptive ACE reaches >= 45% Pd(-10 dB) on both receivers",
              ad1["ACE"]["Pd_m10_mean"] >= 0.45 and ad2["ACE"]["Pd_m10_mean"] >= 0.45,
              f"HF: {100*ad1['ACE']['Pd_m10_mean']:.1f}%, TrevTron: {100*ad2['ACE']['Pd_m10_mean']:.1f}%")
        claim("Non-scale-invariant adaptive baselines (AMF / Kelly) exceed 6% Pf under real noise",
              ad1["AMF"]["Pf_mean"] > 0.060 and ad2["AMF"]["Pf_mean"] > 0.060,
              f"HF AMF: {100*ad1['AMF']['Pf_mean']:.1f}%, TrevTron AMF: {100*ad2['AMF']['Pf_mean']:.1f}%")
    # Real-signal surrogate test (T6)
    frs = os.path.join(ROOT, "RESULTS", "real_signal_summary.json")
    if os.path.exists(frs):
        RS = json.load(open(frs))["results"]
        gap_fm_tt = RS["trevtron_R828D|FM_broadcast|GapCNN-WHITE|sur"]["gap"]
        gap_fm_hf = RS["hf_composite|FM_broadcast|GapCNN-WHITE|sur"]["gap"]
        gap_klo_tt = RS["trevtron_R828D|FM_broadcast|KAPPALO|sur"]["gap"]
        gap_klo_hf = RS["hf_composite|FM_broadcast|KAPPALO|sur"]["gap"]
        gap_surrx_hf = RS["hf_composite|FM_broadcast|GapCNN-SURRX|sur"]["gap"]
        gap_adsb_w = RS["hf_composite|ADS_B|GapCNN-WHITE|sur"]["gap"]
        gap_adsb_b = RS["hf_composite|ADS_B|BAND|sur"]["gap"]
        gap_adsb_k = RS["hf_composite|ADS_B|KAPPA|sur"]["gap"]
        claim("Real-signal surrogate: white-trained network gap <= 2.0 points on FM broadcast (both receivers)",
              abs(gap_fm_tt) <= 0.02 and abs(gap_fm_hf) <= 0.02,
              f"TT: {100*gap_fm_tt:+.1f} pp, HF: {100*gap_fm_hf:+.1f} pp")
        claim("Real-signal surrogate: constant-envelope detector (KAPPALO) gap > 15.0 points on FM broadcast (both receivers)",
              gap_klo_tt > 0.15 and gap_klo_hf > 0.15,
              f"TT: {100*gap_klo_tt:+.1f} pp, HF: {100*gap_klo_hf:+.1f} pp")
        claim("Real-signal surrogate: surrogate-trained network (SURRX) gap > 10.0 points on FM broadcast",
              gap_surrx_hf > 0.10, f"{100*gap_surrx_hf:+.1f} pp")
        claim("Real-signal surrogate: white-trained network tracks oracle band ratio on ADS-B within 1.0 point",
              abs(gap_adsb_w - gap_adsb_b) <= 0.01,
              f"net: {100*gap_adsb_w:+.1f} pp, band: {100*gap_adsb_b:+.1f} pp")
        claim("Real-signal surrogate: pulse-burst structure detector (KAPPA) gap > 10.0 points on ADS-B",
              gap_adsb_k > 0.10, f"{100*gap_adsb_k:+.1f} pp")
    # Empirical shape uncertainty (T7)
    feu = os.path.join(ROOT, "RESULTS", "empirical_uncertainty.json")
    if os.path.exists(feu):
        EU = json.load(open(feu))
        r1 = EU.get("receiver_1_r820t2", {})
        r2 = EU.get("receiver_2_r828d", {})
        cr = EU.get("cross_receiver_145mhz", {})
        if r1:
            sh_p90 = r1["split_half_noise"]["p90"]
            dr_p90 = r1["steady_drift"]["p90"]
            head_med = r1["startup_transient"]["median"]
            head_max = r1["startup_transient"]["max"]
            ret_med = r1["retune"]["median"]
            w_same = r1["empirical_wall"]["same_freq"]["max_g_within_noise_db"]
            ret_min = r1["empirical_wall"]["retune"]["min_rel_residual"]
            claim("Empirical uncertainty: split-half estimation floor p90 <= 0.75 dB and steady drift p90 <= 1.15 dB",
                  sh_p90 <= 0.75 and dr_p90 <= 1.15,
                  f"split p90: {sh_p90:.2f} dB, drift p90: {dr_p90:.2f} dB")
            claim("Empirical uncertainty: startup transient median >= 1.0 dB and max >= 5.0 dB",
                  head_med >= 1.0 and head_max >= 5.0,
                  f"startup med: {head_med:.2f} dB, max: {head_max:.2f} dB")
            claim("Empirical uncertainty: retuning across frequencies causes median shift >= 3.0 dB",
                  ret_med >= 3.0, f"retune med: {ret_med:.2f} dB")
            claim("Empirical uncertainty: same-frequency pairs admit empirical wall <= -18.0 dB",
                  w_same is not None and w_same <= -18.0, f"wall: {w_same:.1f} dB")
        if cr and cr.get("empirical_wall"):
            cr_min = cr["empirical_wall"]["min_rel_residual"]
            claim("Empirical uncertainty: cross-receiver shape shifts at 145 MHz cannot fake signals (min residual >= 10%)",
                  cr_min >= 0.10, f"min residual: {100*cr_min:.1f}% vs 2.8% noise floor")
    # Statistical rigor / bootstrap CIs (T8)
    fsci = os.path.join(ROOT, "RESULTS", "stats_ci.json")
    if os.path.exists(fsci):
        SCI = json.load(open(fsci))
        t1 = SCI.get("table1", {})
        t2 = SCI.get("table2", {})
        paired = {p["key"] + "_" + p["model"] + "_" + p["rec2"] + "_" + p["rec1"]: p for p in SCI.get("paired", [])}
        if "GapCNN_WHITE" in t1 and "surgap" in t1["GapCNN_WHITE"]:
            sg_w = t1["GapCNN_WHITE"]["surgap"]
            claim("Statistical rigor: GapCNN-WHITE surrogate gap 95% CI encompasses zero",
                  sg_w["ci_lo"] <= 0.0 <= sg_w["ci_hi"] and abs(sg_w["ci_hi"] - sg_w["ci_lo"]) <= 0.025,
                  f"CI: [{100*sg_w['ci_lo']:+.1f}, {100*sg_w['ci_hi']:+.1f}] pp")
        if "GapCNN_SURRX" in t2 and "surgap" in t2["GapCNN_SURRX"]:
            sg_sx = t2["GapCNN_SURRX"]["surgap"]
            claim("Statistical rigor: GapCNN-SURRX surrogate gap 95% CI strictly positive (> 5.0 pp)",
                  sg_sx["ci_lo"] > 0.05, f"CI: [{100*sg_sx['ci_lo']:+.1f}, {100*sg_sx['ci_hi']:+.1f}] pp")
        p_surrxm = paired.get("h1_lpf0.8_-10_GapCNN_SURRXM_SURRX")
        if p_surrxm:
            claim("Statistical rigor: Matched recipe SURRXM restores >= 20 pp detection on LP .8 (McNemar p < 1e-50)",
                  p_surrxm["mean_gain"] >= 0.20 and p_surrxm["chi2"] > 1000,
                  f"Gain: +{100*p_surrxm['mean_gain']:.1f} pp, chi2: {p_surrxm['chi2']:.1f}")
        p_randm = paired.get("h1_lpf0.8_-10_GapCNN_RANDM_RAND")
        if p_randm:
            claim("Statistical rigor: Matched recipe RANDM restores >= 12 pp detection on LP .8 (McNemar p < 1e-50)",
                  p_randm["mean_gain"] >= 0.12 and p_randm["chi2"] > 1000,
                  f"Gain: +{100*p_randm['mean_gain']:.1f} pp, chi2: {p_randm['chi2']:.1f}")
        p_pinkm = paired.get("h1_pink_-10_GapCNN_PINKM_PINK")
        if p_pinkm:
            claim("Statistical rigor: Matched recipe PINKM restores >= 20 pp detection on Pink (McNemar p < 1e-50)",
                  p_pinkm["mean_gain"] >= 0.20 and p_pinkm["chi2"] > 1000,
                  f"Gain: +{100*p_pinkm['mean_gain']:.1f} pp, chi2: {p_pinkm['chi2']:.1f}")
    # --- Breadth: OFDM and multipath fading (T11 / Plan H)
    full = load_runs()
    ofdm_runs = full[full.signal == "OFDM"]
    ofdm_net = ofdm_runs[ofdm_runs.model == "GapCNN"]
    ofdm_ed = ofdm_runs[ofdm_runs.model == "ED"]
    if len(ofdm_net):
        claim("OFDM breadth: 5 seeds evaluated under 4-tap Rayleigh fading, CP, and CFO",
              len(ofdm_net) == 5, f"{len(ofdm_net)} seeds")
        claim("OFDM breadth: surrogate gap stays <= 2.0 pp under multipath OFDM",
              abs(ofdm_net.surgap.mean()) <= 0.02, f"mean gap {100*ofdm_net.surgap.mean():+.2f} pp")
        claim("OFDM breadth: at rho=0.75, deep learning has no gain over energy detection (Corollary 1)",
              ofdm_net["Pd_-10"].mean() <= ofdm_ed["Pd_-10"].mean(),
              f"GapCNN {100*ofdm_net['Pd_-10'].mean():.1f}% vs ED {100*ofdm_ed['Pd_-10'].mean():.1f}%")
        claim("OFDM breadth: false alarms on out-of-distribution receiver roll-off >= 50% (LP .8 & RTL 3dB)",
              ofdm_net["Pf_lpf0.8"].mean() >= 0.99 and ofdm_net["Pf_rtl3dB"].mean() >= 0.50,
              f"LP.8 {100*ofdm_net['Pf_lpf0.8'].mean():.1f}%, RTL {100*ofdm_net['Pf_rtl3dB'].mean():.1f}%")
    verification_fix_claims()
    upgrade_claims()
    n_ok = sum(r[0] for r in results)
    print(f"\n{n_ok}/{len(results)} claims pass")
    json.dump([{"ok": r[0], "claim": r[1], "detail": r[2]} for r in results],
              open(os.path.join(ROOT, "RESULTS", "claim_verification.json"), "w"), indent=1)
    return n_ok == len(results)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
