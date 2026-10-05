"""Generates the experiment queues (JSON lists of configs) for CODE/run.py. Order = priority."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS5, SEEDS3 = [0, 1, 2, 3, 4], [0, 1, 2]
EPOCHS = {128: 12, 512: 10, 2048: 8}

q = []
# X1/X2: architecture sweep, standard (WHITE) recipe, N=128, sps=8
for seed in SEEDS5:
    for arch in ["CLDNN", "ResNet1D", "CovCNN", "STFTCNN", "GapCNN"]:
        q.append({"kind": "learned", "arch": arch, "recipe": "WHITE", "N": 128, "sps": 8, "seed": seed, "epochs": 12})
# classical baselines at N=128 (seed = test-draw stream)
for seed in SEEDS3:
    for det in ["ED", "MME", "SFM", "CYC", "BAND"]:
        q.append({"kind": "classical", "detector": det, "N": 128, "sps": 8, "seed": seed})
# X3/X8: training recipes
for seed in SEEDS5:
    for arch in ["CLDNN", "GapCNN"]:
        for rec in ["PINK", "RAND", "SURR"]:
            q.append({"kind": "learned", "arch": arch, "recipe": rec, "N": 128, "sps": 8, "seed": seed, "epochs": 12})
json.dump(q, open(os.path.join(HERE, "queue_A_core.json"), "w"), indent=1)

qb = []
# X4/X8: wall test, N sweep (GapCNN), classical at the same N
for N in [512, 2048]:
    for seed in SEEDS3:
        for rec in ["WHITE", "SURR"]:
            qb.append({"kind": "learned", "arch": "GapCNN", "recipe": rec, "N": N, "sps": 8, "seed": seed, "epochs": EPOCHS[N]})
        for det in ["ED", "MME", "SFM", "CYC", "BAND"]:
            qb.append({"kind": "classical", "detector": det, "N": N, "sps": 8, "seed": seed})
json.dump(qb, open(os.path.join(HERE, "queue_B_wall.json"), "w"), indent=1)

qc = []
# X5: occupancy sweep
for sps in [4, 2]:
    for seed in SEEDS3:
        qc.append({"kind": "learned", "arch": "GapCNN", "recipe": "WHITE", "N": 128, "sps": sps, "seed": seed, "epochs": 12})
        for det in ["ED", "BAND", "MME"]:
            qc.append({"kind": "classical", "detector": det, "N": 128, "sps": sps, "seed": seed})
# X7: reference whitening on trained WHITE checkpoints
for seed in SEEDS3:
    for arch in ["GapCNN", "CLDNN"]:
        qc.append({"kind": "whitening", "arch": arch, "recipe": "WHITE", "N": 128, "sps": 8, "seed": seed})
json.dump(qc, open(os.path.join(HERE, "queue_C_occupancy_whitening.json"), "w"), indent=1)
print(len(q), len(qb), len(qc))
