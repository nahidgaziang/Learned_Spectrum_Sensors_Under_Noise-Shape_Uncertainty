"""Independent verification of the paper's numbers. Written separately from analysis.py / paper_numbers.py and imports
neither: reads RESULTS/ directly with json + numpy and compares with RESULTS/paper_numbers.json.
  A. raw scores (RESULTS/scores/*.npz) at the stored thresholds -> Pf, Pd, surrogate gap stored in each run record
  B. run records -> headline macros
  C. real-noise records -> real-noise macros
  D. wall-test records -> wall macros
Writes RESULTS/independent_check.json; prints every comparison and a summary."""
import glob, json, os, re
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = json.load(open(os.path.join(ROOT, "RESULTS", "paper_numbers.json")))
rows, fails = [], []


def cmp(name, mine, tol, note=""):
    """Compare an independently computed value with the macro's raw value (or with a parsed tex range)."""
    v = P.get(name, {})
    ref = v.get("raw")
    if ref is None or isinstance(ref, (dict, list)):
        nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", v.get("tex", "").replace("$", "").replace("--", " "))]
        ref = nums if isinstance(mine, (list, tuple)) else (nums[0] if nums else None)
        scale = 100 if "\\%" in v.get("tex", "") or "points" in v.get("tex", "") else 1
        mine_cmp = [m * scale for m in mine] if isinstance(mine, (list, tuple)) else mine * scale
        ok = ref is not None and np.allclose(np.atleast_1d(mine_cmp), np.atleast_1d(ref), atol=tol * scale)
        shown_mine = np.round(np.atleast_1d(mine_cmp), 2).tolist()
    else:
        ok = abs(mine - ref) <= tol
        shown_mine = round(mine, 4)
    rows.append({"name": name, "ok": bool(ok), "independent": shown_mine, "paper": v.get("tex"), "note": note})
    if not ok:
        fails.append(name)
    print(f"[{'OK ' if ok else 'DIFF'}] {name:24s} independent={shown_mine}  paper={v.get('tex')}  {note}")


# ---------------------------------------------------------------- load run records (no analysis.py)
recs = []
for f in glob.glob(os.path.join(ROOT, "RESULTS", "runs", "*.json")):
    r = json.load(open(f))
    if "eval" not in r:
        continue
    c = r["config"]
    recs.append({"id": r["id"], "kind": c["kind"], "model": c["detector"] if c["kind"] == "classical" else c["arch"],
                 "recipe": c.get("recipe") if c["kind"] == "learned" else "-", "N": c["N"], "sps": c["sps"],
                 "seed": c["seed"], "tag": c.get("tag") or "", "signal": c.get("signal", "modulated"), "e": r["eval"]})


def sel(**kw):
    if "signal" not in kw:
        kw["signal"] = "modulated"
    return [r for r in recs if all(r.get(k) == v for k, v in kw.items())]


ARCHS = ["CLDNN", "ResNet1D", "CovCNN", "STFTCNN", "GapCNN"]

# ---------------------------------------------------------------- A. raw scores -> stored metrics
print("A. raw scores at stored thresholds vs stored metrics")
maxdiff, n_runs = 0.0, 0
for r in recs:
    f = os.path.join(ROOT, "RESULTS", "scores", r["id"] + ".npz")
    if not os.path.exists(f):
        continue
    z, e = np.load(f), r["e"]
    thr = e["thresholds"]["nominal"]
    n_runs += 1
    for sh in ["white", "lpf0.8", "rtl3dB", "pink", "ar+0.1", "white+1dB"]:
        if f"h0_{sh}" in z.files:
            maxdiff = max(maxdiff, abs(np.mean(z[f"h0_{sh}"] > thr) - e["Pf"][sh]["nominal"]))
    for d in ["-10", "0"]:
        maxdiff = max(maxdiff, abs(np.mean(z[f"h1_white_{d}"] > thr) - e["Pd"][d]["white"]["nominal"]))
        maxdiff = max(maxdiff, abs(np.mean(z[f"sur_white_{d}"] > thr) - e["Pd_surrogate"][d]["nominal"]))
    snrs = sorted(int(k.split("_")[-1]) for k in z.files if k.startswith("h1_white_"))
    sig = {d: np.mean(z[f"h1_white_{d}"] > thr) for d in snrs}
    sur = {d: np.mean(z[f"sur_white_{d}"] > thr) for d in snrs}
    mid = [d for d in snrs if 0.1 < sig[d] < 0.9]
    if mid and e["surrogate_gap"] is not None:
        maxdiff = max(maxdiff, abs(np.mean([sig[d] - sur[d] for d in mid]) - e["surrogate_gap"]))
print(f"   {n_runs} runs re-scored; largest |difference| = {maxdiff:.2e}")
rows.append({"name": "A_rescoring", "ok": bool(maxdiff < 1e-9), "independent": maxdiff, "paper": "stored metrics", "note": f"{n_runs} runs"})
if maxdiff >= 1e-9:
    fails.append("A_rescoring")

# ---------------------------------------------------------------- B. run records -> headline macros
print("\nB. headline macros from run records")
W = [r for r in sel(kind="learned", recipe="WHITE", N=128, sps=8, tag="") if r["model"] in ARCHS]


def arch_means(get):
    return [np.mean([get(r) for r in W if r["model"] == a]) for a in ARCHS]


g = arch_means(lambda r: r["e"]["surrogate_gap"])
cmp("SurGapRange", [min(g), max(g)], 0.0006)
cmp("SurGapAbsMax", max(abs(r["e"]["surrogate_gap"]) for r in W), 0.0006)
for sh, name in [("lpf0.8", "PfLpfEightRange"), ("rtl3dB", "PfRtlThreeRange"), ("pink", "PfPinkRange"),
                 ("ar+0.1", "PfArOneRange"), ("ar-0.1", "PfArMinusOneRange"), ("white+1dB", "PfPowerUpRange")]:
    m = arch_means(lambda r: r["e"]["Pf"][sh]["nominal"])
    cmp(name, [min(m), max(m)] if round(min(m), 2) != round(max(m), 2) else [min(m)], 0.006)
m = arch_means(lambda r: r["e"]["Pd"]["-10"]["white"]["nominal"]); cmp("PdTenRange", [min(m), max(m)], 0.006)
m = arch_means(lambda r: r["e"]["Pd_worstcase"]["U_receiver"]["-10"]); cmp("WcPdReceiverRange", [min(m), max(m)], 0.006)
for det, key in [("BAND", "BandPdTen"), ("ED", "EdPdTen"), ("CYC", "CycPdTen")]:
    cmp(key, np.mean([r["e"]["Pd"]["-10"]["white"]["nominal"] for r in sel(kind="classical", model=det, N=128, sps=8, tag="")]), 0.006)
cmp("CycSurGap", np.mean([r["e"]["surrogate_gap"] for r in sel(kind="classical", model="CYC", N=128, sps=8, tag="")]), 0.0006)
cmp("EdPfPowerUp", np.mean([r["e"]["Pf"]["white+1dB"]["nominal"] for r in sel(kind="classical", model="ED", N=128, sps=8, tag="")]), 0.006)
for N, L in [(128, "NA"), (512, "NB"), (2048, "NC")]:
    s = sel(kind="learned", model="GapCNN", recipe="WHITE", N=N, sps=8, tag="")
    cmp(f"PfLpfWhite{L}", np.mean([r["e"]["Pf"]["lpf0.8"]["nominal"] for r in s]), 0.006, f"n={len(s)}")
    cmp(f"WcRecWhite{L}", np.mean([r["e"]["Pd_worstcase"]["U_receiver"]["-10"] for r in s]), 0.006)
for sps, L in [(8, "Eight"), (4, "Four"), (2, "Two")]:
    a = np.mean([r["e"]["Pd"]["-10"]["white"]["nominal"] for r in sel(kind="learned", model="GapCNN", recipe="WHITE", N=128, sps=sps, tag="")])
    ed = np.mean([r["e"]["Pd"]["-10"]["white"]["nominal"] for r in sel(kind="classical", model="ED", N=128, sps=sps, tag="")])
    cmp(f"OccGain{L}", a - ed, 0.006)
pk = [np.mean([r["e"]["Pd"]["-10"]["pink"]["nominal"] for r in sel(kind="learned", model=a, recipe="PINK", N=128, sps=8, tag="")]) for a in ["CLDNN", "GapCNN"]]
cmp("MaskPinkPdPink", [min(pk), max(pk)], 0.006)
cmp("MaskRandPdLpf", np.mean([r["e"]["Pd"]["-10"]["lpf0.8"]["nominal"] for r in sel(kind="learned", model="GapCNN", recipe="RAND", N=128, sps=8, tag="")]), 0.006)

# colour-calibrated detection directly from raw scores (threshold = 95th percentile of the colour's own noise)
def pdcal(model, recipe, shape="lpf0.8"):
    v = []
    for r in sel(model=model, recipe=recipe, N=128, sps=8, tag=""):
        z = np.load(os.path.join(ROOT, "RESULTS", "scores", r["id"] + ".npz"))
        v.append(np.mean(z[f"h1_{shape}_-10"] > np.quantile(z[f"h0_{shape}"], 0.95)))
    return float(np.mean(v))
for rec, nm in [("RAND", "Rand"), ("RANDM", "Randm"), ("SURRX", "Surrx"), ("SURRXM", "Surrxm")]:
    cmp(f"PdCalLpf{nm}", pdcal("GapCNN", rec), 0.006)
wl = [pdcal(a, "WHITE") for a in ARCHS]
cmp("PdCalLpfWhiteRange", [min(wl), max(wl)], 0.006)
scale_runs = [r for r in recs if r.get("c", {}).get("tag") == "scale4x_600k"]
if scale_runs:
    cmp("AblScaleLpf", np.mean([r["e"]["Pf"]["lpf0.8"]["nominal"] for r in scale_runs]), 0.006, f"{len(scale_runs)} seeds")
    cmp("AblScalePdTen", np.mean([r["e"]["Pd"]["-10"]["white"]["nominal"] for r in scale_runs]), 0.006, f"{len(scale_runs)} seeds")
ofdm_runs = [r for r in recs if r.get("c", {}).get("signal") == "OFDM"]
ofdm_net = [r for r in ofdm_runs if r["c"].get("kind") == "learned" and r["c"].get("arch") == "GapCNN"]
ofdm_ed = [r for r in ofdm_runs if r["c"].get("kind") == "classical" and r["c"].get("detector") == "ED"]
if ofdm_net:
    cmp("OfdmSeeds", len(ofdm_net), 0)
    cmp("OfdmGap", np.mean([r["e"]["surrogate_gap"] for r in ofdm_net]), 0.0006, f"{len(ofdm_net)} seeds")
    cmp("OfdmPdTen", np.mean([r["e"]["Pd"]["-10"]["white"]["nominal"] for r in ofdm_net]), 0.006)
    if ofdm_ed:
        cmp("OfdmEdPdTen", np.mean([r["e"]["Pd"]["-10"]["white"]["nominal"] for r in ofdm_ed]), 0.006)
    cmp("OfdmPfLpf", np.mean([r["e"]["Pf"]["lpf0.8"]["nominal"] for r in ofdm_net]), 0.006)
    cmp("OfdmPfRtl", np.mean([r["e"]["Pf"]["rtl3dB"]["nominal"] for r in ofdm_net]), 0.006)


# ---------------------------------------------------------------- C. real noise
print("\nC. recorded noise")
fv3 = os.path.join(ROOT, "RESULTS", "real_noise_v3_N128.json")
fv2 = os.path.join(ROOT, "RESULTS", "real_noise_v2_N128.json")
fn = fv3 if os.path.exists(fv3) else fv2
R = json.load(open(fn))["recordings"]
live = [v for v in R.values() if v["group"] == "live_rtl" and not v["shape"]["narrowband"]]
ctrl = [v for v in R.values() if v["group"] == "unlabelled"]
cmp("RealNoiseNrec", len(live), 0)
cmp("RealNoiseNctrl", len(ctrl), 0)


def per_arch(recset, field):
    out = []
    for a in ARCHS:
        out.append(np.mean([np.mean([x for k, x in r[field].items() if k.startswith(a + "-WHITE")]) for r in recset]))
    return out
pr = per_arch(live, "Pf"); cmp("PfRealRange", [min(pr), max(pr)], 0.006)
cmp("PfRealControlMax", max(per_arch(ctrl, "Pf")), 0.0006)
pw = per_arch(live, "Pf_wbig"); cmp("PfRealWhitenedRange", [min(pw), max(pw)], 0.0006)
if os.path.exists(fv2) and os.path.exists(fv3):
    R2 = json.load(open(fv2))["recordings"]
    live2 = [v for v in R2.values() if v["group"] == "live_rtl" and not v["shape"]["narrowband"]]
    pw2 = per_arch(live2, "Pf_wbig"); cmp("PfRealWhitenedRangeVTwo", [min(pw2), max(pw2)], 0.0006)
x = np.array([r["shape"]["edge_rolloff_db"] for r in live + ctrl])
y = np.array([np.mean([v for k, v in r["Pf"].items() if any(k.startswith(a + "-WHITE") for a in ARCHS)]) for r in live + ctrl])
rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))      # Spearman via ranks (no ties expected in x)
rho = np.corrcoef(rx, ry)[0, 1]
cmp("RealDoseRho", rho, 0.01, "rank correlation, own implementation")

ftt = os.path.join(ROOT, "RESULTS", "real_noise_trevtron_N128.json")
if os.path.exists(ftt):
    Rtt = json.load(open(ftt))["recordings"]
    cl_tt = [v for v in Rtt.values() if not v["shape"]["narrowband"]]
    cmp("TrevNoiseNrec", len(cl_tt), 0)
    cmp("TrevNoiseNtotal", len(Rtt), 0)
    rott = [v["shape"]["edge_rolloff_db"] for v in cl_tt]
    cmp("TrevRolloff", np.median(rott), 0.05)

fa1 = os.path.join(ROOT, "RESULTS", "adaptive_baselines.json")
fa2 = os.path.join(ROOT, "RESULTS", "adaptive_baselines_trevtron.json")
if os.path.exists(fa1) and os.path.exists(fa2):
    ad1 = json.load(open(fa1))["recorded_summary"]
    ad2 = json.load(open(fa2))["recorded_summary"]
    cmp("AdaptAcePfMeanHf", ad1["ACE"]["Pf_mean"], 0.005)
    cmp("AdaptAcePfMeanTrev", ad2["ACE"]["Pf_mean"], 0.005)
    cmp("AdaptAmfPfMeanHf", ad1["AMF"]["Pf_mean"], 0.005)
    cmp("AdaptKellyPfMeanHf", ad1["KELLY"]["Pf_mean"], 0.005)

frs = os.path.join(ROOT, "RESULTS", "real_signal_summary.json")
if os.path.exists(frs):
    RS = json.load(open(frs))["results"]
    cmp("RealSigFmGapTt", RS["trevtron_R828D|FM_broadcast|GapCNN-WHITE|sur"]["gap"] * 100, 0.1)
    cmp("RealSigFmGapHf", RS["hf_composite|FM_broadcast|GapCNN-WHITE|sur"]["gap"] * 100, 0.1)
    cmp("RealSigFmKappaLoGapTt", RS["trevtron_R828D|FM_broadcast|KAPPALO|sur"]["gap"] * 100, 0.1)
    cmp("RealSigFmKappaLoGapHf", RS["hf_composite|FM_broadcast|KAPPALO|sur"]["gap"] * 100, 0.1)
    cmp("RealSigAdsbGapHf", RS["hf_composite|ADS_B|GapCNN-WHITE|sur"]["gap"] * 100, 0.1)

feu = os.path.join(ROOT, "RESULTS", "empirical_uncertainty.json")
if os.path.exists(feu):
    EU = json.load(open(feu))
    r1 = EU.get("receiver_1_r820t2", {})
    r2 = EU.get("receiver_2_r828d", {})
    cr = EU.get("cross_receiver_145mhz", {})
    if r1:
        cmp("EmpSplitHalfMedianHf", r1["split_half_noise"]["median"], 0.01)
        cmp("EmpSteadyDriftMedianHf", r1["steady_drift"]["median"], 0.01)
        cmp("EmpStartupMedianHf", r1["startup_transient"]["median"], 0.01)
        cmp("EmpSessionMedianHf", r1["session"]["median"], 0.01)
        cmp("EmpRetuneMedianHf", r1["retune"]["median"], 0.01)
        w_same = r1["empirical_wall"]["same_freq"]["max_g_within_noise_db"]
        if w_same is not None:
            cmp("EmpWallSameFreqDb", w_same, 0.1)
    if r2 and r2.get("split_half_noise"):
        cmp("EmpSplitHalfMedianTt", r2["split_half_noise"]["median"], 0.01)
    if cr and cr.get("uncertainty"):
        cmp("EmpCrossRecMedian", cr["uncertainty"]["median"], 0.01)

# ---------------------------------------------------------------- D. wall tests
print("\nD. wall tests")
WT = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(ROOT, "RESULTS", "wall_test_s*.json")))]
for N, L in [(128, "A"), (512, "B"), (2048, "C")]:
    cmp(f"WallWhiteN{L}Minus", np.mean([d[f"N{N}_R1.0_GapCNN-WHITE"]["Pd"]["-2"]["worst"] for d in WT]), 0.006, f"{len(WT)} seeds")
    cmp(f"WallSurrxmN{L}", np.mean([d[f"N{N}_R1.0_GapCNN-SURRXM"]["Pd"]["0"]["worst"] for d in WT]), 0.006)
    cmp(f"WallTwoSurrxmN{L}", np.mean([d[f"N{N}_R2.0_GapCNN-SURRXM"]["Pd"]["0"]["worst"] for d in WT]), 0.006)
LT = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(ROOT, "RESULTS", "wall_test_long_s*.json")))]
for N, L in [(2048, "C"), (8192, "D"), (32768, "E")]:
    cmp(f"WallLongCycN{L}", np.mean([d[f"N{N}_R1.0"]["Pd"]["0"]["CYC"]["worst"] for d in LT]), 0.006, f"{len(LT)} seeds")
    cmp(f"WallLongBandN{L}", np.mean([d[f"N{N}_R1.0"]["Pd"]["0"]["BAND"]["worst"] for d in LT]), 0.006)
ET = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(ROOT, "RESULTS", "wall_test_escape_s*.json")))]
if ET:
    for N, L in [(8192, "ND"), (32768, "NE")]:
        for det, nm in [("GapCNN-PAIRED-N8192", "Paired"), ("CycNet-WHITE-N8192", "CycNet")]:
            v0 = np.mean([d[f"N{N}_R1.0"]["Pd"]["0"][det]["worst"] for d in ET if f"N{N}_R1.0" in d and "Pd" in d[f"N{N}_R1.0"]])
            vm = np.mean([d[f"N{N}_R1.0"]["Pd"]["-2"][det]["worst"] for d in ET if f"N{N}_R1.0" in d and "Pd" in d[f"N{N}_R1.0"]])
            cmp(f"Escape{nm}Wall{L}", v0, 0.006)
            cmp(f"Escape{nm}Below{L}", vm, 0.006)
J = json.load(open(os.path.join(ROOT, "RESULTS", "wall_vs_theory.json")))
cmp("ThConfigs", len(J), 0)
cmp("ThMaxBelow", max(v["Pd"][d]["worst"] for v in J.values() for d in v["Pd"] if float(d) <= 0), 0.0006)

# ---------------------------------------------------------------- E. statistical rigor (paired tests)
print("\nE. statistical rigor (paired tests)")


def indep_paired_gain(model, rec1, rec2, snr_key):
    diffs = []
    for s in range(5):
        f1 = os.path.join(ROOT, "RESULTS", "scores", f"{model}_{rec1}_N128_sps8_s{s}.npz")
        f2 = os.path.join(ROOT, "RESULTS", "scores", f"{model}_{rec2}_N128_sps8_s{s}.npz")
        j1 = f1.replace("scores", "runs").replace(".npz", ".json")
        j2 = f2.replace("scores", "runs").replace(".npz", ".json")
        if not (os.path.exists(f1) and os.path.exists(f2) and os.path.exists(j1) and os.path.exists(j2)):
            continue
        d1 = np.load(f1)
        d2 = np.load(f2)
        t1 = json.load(open(j1))["eval"]["thresholds"]["nominal"]
        t2 = json.load(open(j2))["eval"]["thresholds"]["nominal"]
        diffs.append(np.mean(d2[snr_key] > t2) - np.mean(d1[snr_key] > t1))
    return np.mean(diffs) * 100 if diffs else 0.0


cmp("CiSurrxmGainLpf", indep_paired_gain("GapCNN", "SURRX", "SURRXM", "h1_lpf0.8_-10"), 0.2)
cmp("CiRandmGainLpf", indep_paired_gain("GapCNN", "RAND", "RANDM", "h1_lpf0.8_-10"), 0.2)
cmp("CiPinkmGainPinkGap", indep_paired_gain("GapCNN", "PINK", "PINKM", "h1_pink_-10"), 0.2)
cmp("CiPinkmGainPinkCldnn", indep_paired_gain("CLDNN", "PINK", "PINKM", "h1_pink_-10"), 0.2)

# ---------------------------------------------------------------- E. novelty upgrades (2026-10-05): independent routes
print("\nE. Upgrades (independent routes)")


def rrc_pulse(beta=0.35, sps=8, span=8):          # re-implemented here (not imported from sim)
    t = np.arange(-span * sps, span * sps + 1) / sps
    with np.errstate(divide="ignore", invalid="ignore"):
        h = (np.sin(np.pi * t * (1 - beta)) + 4 * beta * t * np.cos(np.pi * t * (1 + beta))) / (np.pi * t * (1 - (4 * beta * t) ** 2))
    h[np.isclose(t, 0)] = 1 - beta + 4 * beta / np.pi
    sing = np.isclose(np.abs(t), 1 / (4 * beta))
    h[sing] = beta / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
    return h / np.sqrt(np.sum(h ** 2))


# E1 kappa of the escape law via the FREQUENCY domain (escape_law.py uses the time-domain sum)
h = rrc_pulse(); K = 4096; sps = 8
H = np.fft.fft(h, K)
Hs = np.roll(H, K // sps)                         # H[k - K/sps]
r = np.array([np.sum(H * np.conj(Hs) * np.exp(2j * np.pi * np.arange(K) * tau / K)) / K for tau in range(5)]) / np.sum(np.abs(h) ** 2)
kap5 = float(2 * np.sum(np.abs(r) ** 2))
cmp("EscLawKappaFive", kap5, 1e-6, "frequency-domain spectral correlation of the pulse")

# E2 exact wall by bisection on a pair-FEASIBILITY LP (different formulation from the gamma-maximising LP)
from scipy.optimize import linprog
from scipy import sparse, stats
Kg = 2048
f = np.fft.fftfreq(Kg)
p = 0.2 + np.exp(-(f / 0.08) ** 2); p = p / p.mean()
Rr = 10 ** 0.1


def pair_feasible(g):
    I = sparse.identity(Kg, format="csr"); z = sparse.csr_matrix((Kg, 1)); e = sparse.csr_matrix(np.ones((Kg, 1)))
    A_eq = sparse.vstack([sparse.hstack([(1 + g) * I, -I, z, z]),
                          sparse.hstack([sparse.csr_matrix(np.full((1, Kg), 1.0 / Kg)), sparse.csr_matrix((1, Kg + 2))])], format="csr")
    b_eq = np.concatenate([g * p, [1.0]])
    Z = sparse.csr_matrix((Kg, Kg))
    A_ub = sparse.vstack([sparse.hstack([-I, Z, e, z]), sparse.hstack([I, Z, -Rr * e, z]),
                          sparse.hstack([Z, -I, z, e]), sparse.hstack([Z, I, z, -Rr * e])], format="csr")
    res = linprog(np.zeros(2 * Kg + 2), A_ub=A_ub, b_ub=np.zeros(4 * Kg), A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * (2 * Kg + 2), method="highs")
    return res.status == 0


lo, hi = -15.0, 0.0
for _ in range(30):
    mid = 0.5 * (lo + hi)
    lo, hi = (mid, hi) if pair_feasible(10 ** (mid / 10)) else (lo, mid)
cmp("ExactDWall", 0.5 * (lo + hi), 0.005, "bisection on pair feasibility, Gaussian bump + floor, u = 1 dB")

# E3 escape law: recompute every prediction of the existing wall tests from the stored thresholds and kappa (E1)
errs = []
for fn in sorted(glob.glob(os.path.join(ROOT, "RESULTS", "wall_test_long_s*.json"))) + sorted(glob.glob(os.path.join(ROOT, "RESULTS", "wall_test_escape_s*.json"))):
    d = json.load(open(fn))
    for key, v in d.items():
        if not key.startswith("N") or "CYC" not in v.get("thr", {}):
            continue
        if "escape" in fn and key.endswith("R1.0"):
            continue                              # identical streams to wall_test_long at u = 1 dB
        for dl, pdv in v["Pd"].items():
            g = 10 ** ((v["wall_db"] + float(dl)) / 10)
            meas = pdv["CYC"]["white"]
            if 0.08 < meas < 0.97:
                pred = stats.ncx2.sf(2 * v["N"] * v["thr"]["CYC"], 10, kap5 * v["N"] * g * g / (1 + g) ** 2)
                errs.append(abs(meas - pred))
cmp("EscLawMaxErr", max(errs), 0.006, f"{len(errs)} points")

# ---------------------------------------------------------------- data integrity
print("\nIntegrity")
bad = [r["id"] for r in recs if any(not np.isfinite(r["e"]["Pf"][s]["nominal"]) for s in r["e"]["Pf"])]
print(f"   {len(recs)} evaluated run records; non-finite Pf: {bad or 'none'}")
leak = [r["id"] for r in W if r["tag"]]
print(f"   tagged runs in the main white-recipe set: {leak or 'none'}")
counts = {}
for r in sel(kind="learned", N=128, sps=8, tag=""):
    counts[(r["model"], r["recipe"])] = counts.get((r["model"], r["recipe"]), 0) + 1
print(f"   N=128 learned seed counts: {sorted(set(counts.values()))} (min {min(counts.values())})")
json.dump({"rows": rows, "fails": fails}, open(os.path.join(ROOT, "RESULTS", "independent_check.json"), "w"), indent=1, default=str)
print(f"\nSUMMARY: {len(rows) - len(fails)}/{len(rows)} agree; differences: {fails or 'none'}")
