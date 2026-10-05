import json

with open("RESULTS/runs/DetectNetExact_WHITE_N128_sps8_s0_QAM16.json") as f:
    d = json.load(f)["eval"]

print("DetectNetExact (seed 0) Detection vs Surrogate across SNRs:")
for snr in sorted(d["Pd"].keys(), key=lambda x: int(x)):
    pd = d["Pd"][snr]["white"]["nominal"]
    pd_sur = d["Pd_surrogate"][snr]["nominal"]
    print(f"SNR {snr:>3} dB | Pd = {pd:.3f} | Pd_sur = {pd_sur:.3f} | diff = {pd - pd_sur:+.3f}")
