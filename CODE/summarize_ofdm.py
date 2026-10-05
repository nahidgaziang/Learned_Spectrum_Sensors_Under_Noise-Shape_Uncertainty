"""Summarize OFDM breadth experiments and compare against theory (Corollary 1) and Energy Detection."""
import json, os, glob
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(ROOT, "RESULTS", "runs")

def summarize():
    ed_path = os.path.join(RUNS_DIR, "cls_ED_N128_OFDM_s0.json")
    band_path = os.path.join(RUNS_DIR, "cls_BAND_N128_OFDM_s0.json")
    
    with open(ed_path) as f:
        ed_data = json.load(f)["eval"]
    with open(band_path) as f:
        band_data = json.load(f)["eval"]
        
    net_files = sorted(glob.glob(os.path.join(RUNS_DIR, "GapCNN_WHITE_N128_OFDM_s*.json")))
    if not net_files:
        print("No GapCNN OFDM runs found yet.")
        return None
        
    net_results = []
    for fp in net_files:
        with open(fp) as f:
            net_results.append(json.load(f)["eval"])
            
    n_seeds = len(net_results)
    
    # Extract metrics
    surgaps = [r["surrogate_gap"] for r in net_results if r["surrogate_gap"] is not None]
    pd_m10 = [r["Pd"]["-10"]["white"]["nominal"] for r in net_results]
    pf_lpf = [r["Pf"]["lpf0.8"]["nominal"] for r in net_results]
    pf_rtl = [r["Pf"]["rtl3dB"]["nominal"] for r in net_results]
    pf_pink = [r["Pf"]["pink"]["nominal"] for r in net_results]
    
    ed_pd_m10 = ed_data["Pd"]["-10"]["white"]["nominal"]
    band_pd_m10 = band_data["Pd"]["-10"]["white"]["nominal"]
    
    mean_surgap = float(np.mean(surgaps))
    mean_pd_m10 = float(np.mean(pd_m10))
    mean_pf_lpf = float(np.mean(pf_lpf))
    mean_pf_rtl = float(np.mean(pf_rtl))
    mean_pf_pink = float(np.mean(pf_pink))
    
    gain_over_ed = mean_pd_m10 - ed_pd_m10
    gain_over_band = mean_pd_m10 - band_pd_m10
    
    summary = {
        "n_seeds": n_seeds,
        "surgap_mean": mean_surgap,
        "surgap_std": float(np.std(surgaps)),
        "surgap_seeds": surgaps,
        "pd_m10_mean": mean_pd_m10,
        "pd_m10_seeds": pd_m10,
        "ed_pd_m10": ed_pd_m10,
        "band_pd_m10": band_pd_m10,
        "gain_over_ed_pp": gain_over_ed * 100.0,
        "gain_over_band_pp": gain_over_band * 100.0,
        "pf_lpf_mean": mean_pf_lpf,
        "pf_rtl_mean": mean_pf_rtl,
        "pf_pink_mean": mean_pf_pink,
        "ed_surgap": ed_data.get("surrogate_gap"),
        "band_surgap": band_data.get("surrogate_gap")
    }
    
    out_path = os.path.join(ROOT, "RESULTS", "ofdm_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
        
    print("=" * 70)
    print("           OFDM & MULTIPATH BREADTH EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"Seeds evaluated:            {n_seeds}")
    print(f"Surrogate Gap (GapCNN):     {mean_surgap * 100:+.2f} pp (std: {np.std(surgaps)*100:.2f} pp)")
    print(f"Pd(-10 dB) GapCNN:          {mean_pd_m10 * 100:.1f}%")
    print(f"Pd(-10 dB) Energy Detector: {ed_pd_m10 * 100:.1f}%")
    print(f"Pd(-10 dB) Band Detector:   {band_pd_m10 * 100:.1f}%")
    print(f"GapCNN Gain over ED:        {gain_over_ed * 100:+.1f} pp (Corollary 1 prediction: ~0 pp at rho=0.75)")
    print(f"Pf(LPF 0.8) on OFDM net:    {mean_pf_lpf * 100:.1f}% (Receiver filter false alarm)")
    print(f"Pf(RTL 3dB) on OFDM net:    {mean_pf_rtl * 100:.1f}%")
    print(f"Pf(Pink) on OFDM net:       {mean_pf_pink * 100:.1f}%")
    print("=" * 70)
    return summary

if __name__ == "__main__":
    summarize()
