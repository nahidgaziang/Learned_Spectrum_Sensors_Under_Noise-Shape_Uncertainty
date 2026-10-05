"""Train/evaluate one configuration, or a resumable queue of configurations.

Usage:
  py -3.12 run.py --queue ../CONFIGS/queue_X1.json       # runs every config whose result file is missing
  py -3.12 run.py --one '{"kind":"learned","arch":"GapCNN","recipe":"WHITE","N":128,"sps":8,"seed":0}'

Outputs (paths relative to PROJECT/):
  RESULTS/runs/<id>.json      metrics (evaluate.evaluate) + config + training record
  RESULTS/scores/<id>.npz     raw test scores (for post-hoc analysis)
  EXPERIMENTS/ckpt/<id>.pt    model weights (learned detectors; git-ignored)
  EXPERIMENTS/runs.jsonl      one machine-readable line per finished run
  EXPERIMENTS/EXPERIMENT_LOG.md  one human-readable row per finished run
"""
import argparse, datetime, json, os, platform, shutil, subprocess, sys, time
import numpy as np
import torch, torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim, detectors as D, evaluate as E

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GIT = shutil.which("git") or "git"
DEFAULTS = {"kind": "learned", "arch": "GapCNN", "recipe": "WHITE", "N": 128, "sps": 8, "seed": 0,
            "ntrain": 60000, "epochs": 12, "lr": 1e-3, "batch": 128, "beta": 0.35, "tag": "", "amp": False}


def run_id(c):
    sig = c.get("signal")
    if sig:
        if c["kind"] == "classical":
            base = f"cls_{c['detector']}_N{c['N']}_{sig}_s{c['seed']}"
        elif c["kind"] == "whitening":
            base = f"wht_{c['arch']}_{c['recipe']}_N{c['N']}_{sig}_s{c['seed']}"
        else:
            base = f"{c['arch']}_{c['recipe']}_N{c['N']}_{sig}_s{c['seed']}"
    else:
        if c["kind"] == "classical":
            base = f"cls_{c['detector']}_N{c['N']}_sps{c['sps']}_s{c['seed']}"
        elif c["kind"] == "whitening":
            base = f"wht_{c['arch']}_{c['recipe']}_N{c['N']}_sps{c['sps']}_s{c['seed']}"
        else:
            base = f"{c['arch']}_{c['recipe']}_N{c['N']}_sps{c['sps']}_s{c['seed']}"
    return base + (f"_{c['tag']}" if c.get("tag") else "")


def git_version():
    try:
        h = subprocess.run([GIT, "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run([GIT, "-C", ROOT, "status", "--porcelain", "CODE"], capture_output=True, text=True).stdout.strip()
        return (h or "nocommit") + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


# ----------------------------------------------------------------------------- training data
# Matched ('M') recipes: the background noise of BOTH classes is drawn from the same distribution, so that the
# noise colour carries no information about the label. (In PINK/RAND/SURR/SURRX only the negatives are coloured;
# positives are signal + white noise, which lets a network learn the shortcut 'coloured background = noise'.)
MATCHED = ("PINKM", "RANDM", "SURRXM", "PAIRED")


def smooth_shape_noise(n, N, rng, max_db=3.0):
    """Random smooth noise shapes: log-PSD = random 8-term cosine series, peak-to-peak up to max_db (as in SURRX)."""
    f = np.fft.fftfreq(N)
    out = []
    for _ in range(n):
        a = rng.normal(0, 1, 8) / np.arange(1, 9)
        logp = sum(a[k] * np.cos(2 * np.pi * (k + 1) * f) for k in range(8))
        logp = logp / max(np.ptp(logp), 1e-9) * rng.uniform(0, max_db) * np.log(10) / 10
        out.append(sim.from_psd(1, N, rng, np.exp(logp))[0])
    return np.stack(out) if out else np.zeros((0, N), complex)


def background(recipe, n, N, rng):
    """Unit-power background noise of the matched recipes (used for positives AND negatives)."""
    if recipe == "PINKM":
        k = n // 2
        x = np.concatenate([sim.white(k, N, rng), sim.pink(n - k, N, rng)])
    elif recipe == "RANDM":
        x = sim.ar1(n, N, rng, rng.uniform(-0.3, 0.3, n))
    elif recipe in ("SURRXM", "PAIRED"):
        k1, k2 = n // 2, n // 4
        x = np.concatenate([smooth_shape_noise(k1, N, rng), sim.ar1(k2, N, rng, rng.uniform(-0.5, 0.5, k2)),
                            sim.white(n - k1 - k2, N, rng)])
    else:
        raise ValueError(recipe)
    return x[rng.permutation(n)]


def negatives(recipe, n, N, rng, sps_sig=8, beta_sig=0.35):
    if recipe in ("PINKM", "RANDM"):
        return background(recipe, n, N, rng)
    if recipe == "SURRXM":
        # as SURRX (40 % exact-shape surrogates, 20 % random-shape surrogates, 40 % noise only), but every example's
        # background comes from the matched mixture (50 % smooth random shapes, 25 % AR(1), 25 % white)
        n1, n2 = int(0.4 * n), int(0.2 * n)
        n3 = n - n1 - n2
        snr1 = rng.uniform(-20, 20, n1)
        p1 = (sim.surrogate(n1, N, rng, sps=sps_sig, beta=beta_sig, cfo_max=0.01) * np.sqrt(10 ** (snr1 / 10))[:, None]
              + background(recipe, n1, N, rng))
        sur = []
        for i in range(0, n2, 1000):
            j = min(n2, i + 1000)
            sur.append(sim.surrogate(j - i, N, rng, sps=int(rng.choice([4, 8, 16])), beta=float(rng.uniform(0.15, 0.6)), cfo_max=0.05))
        snr2 = rng.uniform(-20, 20, n2)
        p2 = np.concatenate(sur) * np.sqrt(10 ** (snr2 / 10))[:, None] + background(recipe, n2, N, rng)
        return np.concatenate([p1, p2, background(recipe, n3, N, rng)])
    if recipe == "WHITE":
        return sim.white(n, N, rng)
    if recipe == "PINK":
        k = n // 2
        return np.concatenate([sim.white(k, N, rng), sim.pink(n - k, N, rng)])
    if recipe == "RAND":
        return sim.ar1(n, N, rng, rng.uniform(-0.3, 0.3, n))
    if recipe == "SURR":
        k = n // 4
        parts = [sim.white(k, N, rng), sim.ar1(k, N, rng, rng.uniform(-0.5, 0.5, k))]
        m = n - 2 * k
        sur = []
        for i in range(0, m, 1000):
            j = min(m, i + 1000)
            sur.append(sim.surrogate(j - i, N, rng, sps=int(rng.choice([4, 8, 16])), beta=float(rng.uniform(0.15, 0.6)),
                                     cfo_max=0.05))
        sur = np.concatenate(sur)
        snr = rng.uniform(-20, 20, m)
        parts.append(sur * np.sqrt(10 ** (snr / 10))[:, None] + sim.white(m, N, rng))
        return np.concatenate(parts)
    if recipe == "SURRX":
        # Exact-shape surrogates dominate the negatives, so that PSD shape carries NO information about the label:
        # 40 % surrogates with the signal's own PSD (same pulse, sps, roll-off and CFO law) + white noise at random SNR,
        # 20 % randomly shaped surrogates, 20 % random smooth shapes (log-PSD = random 8-term cosine series, up to
        # about +-3 dB: covers receiver roll-offs), 10 % AR(1), 10 % white.
        n1, n2, n3, n4 = int(0.4 * n), int(0.2 * n), int(0.2 * n), int(0.1 * n)
        n5 = n - n1 - n2 - n3 - n4
        snr1 = rng.uniform(-20, 20, n1)
        exact = sim.surrogate(n1, N, rng, sps=sps_sig, beta=beta_sig, cfo_max=0.01)
        p1 = exact * np.sqrt(10 ** (snr1 / 10))[:, None] + sim.white(n1, N, rng)
        sur = []
        for i in range(0, n2, 1000):
            j = min(n2, i + 1000)
            sur.append(sim.surrogate(j - i, N, rng, sps=int(rng.choice([4, 8, 16])), beta=float(rng.uniform(0.15, 0.6)), cfo_max=0.05))
        snr2 = rng.uniform(-20, 20, n2)
        p2 = np.concatenate(sur) * np.sqrt(10 ** (snr2 / 10))[:, None] + sim.white(n2, N, rng)
        f = np.fft.fftfreq(N)
        p3 = []
        for i in range(n3):
            a = rng.normal(0, 1, 8) / np.arange(1, 9)
            logp = sum(a[k] * np.cos(2 * np.pi * (k + 1) * f) for k in range(8))
            logp = logp / max(np.ptp(logp), 1e-9) * rng.uniform(0, 3.0) * np.log(10) / 10
            p3.append(sim.from_psd(1, N, rng, np.exp(logp))[0])
        p3 = np.stack(p3)
        p4 = sim.ar1(n4, N, rng, rng.uniform(-0.5, 0.5, n4))
        p5 = sim.white(n5, N, rng)
        return np.concatenate([p1, p2, p3, p4, p5])
    raise ValueError(recipe)


def training_set(c, rng, chunk=None):
    n = c["ntrain"]
    half = n // 2
    if chunk is None:
        chunk = 2500 if c["N"] >= 8192 else 5000
    rep_dummy = D.represent(c["arch"], np.ones((1, c["N"]), dtype=np.complex128), c.get("norm", True), sps=c["sps"])
    rep_shape = rep_dummy.shape[1:]
    X = torch.empty((n, *rep_shape), dtype=torch.float32)
    y = torch.empty(n, dtype=torch.float32)
    p = rng.permutation(n)
    idx_pos = p[:half]
    idx_neg = p[half:]
    if c["recipe"] == "PAIRED":
        for i in range(0, half, chunk):
            m = min(chunk, half - i)
            snr = rng.uniform(-20, 20, m)
            if c.get("signal") == "OFDM":
                sig = sim.ofdm(m, c["N"], rng, multipath=c.get("multipath", True))
            else:
                sig = sim.modulated(m, c["N"], rng, sps=c["sps"], beta=c["beta"], mods=tuple(c.get("mods") or sim.MODS))
            bg = background("SURRXM", m, c["N"], rng)
            g = np.sqrt(10 ** (snr / 10))[:, None]
            pos = sig * g + bg

            if c.get("signal") == "OFDM":
                sur = sim.ofdm_surrogate(m, c["N"], rng, multipath=c.get("multipath", True))
            else:
                S = np.fft.fft(sig, axis=1)
                sur = np.fft.ifft(S * np.exp(2j * np.pi * rng.random(S.shape)), axis=1)
                sur = sur / np.sqrt(np.mean(np.abs(sur) ** 2, axis=1, keepdims=True))

            m_p = int(0.8 * m)
            neg = np.empty_like(pos)
            neg[:m_p] = sur[:m_p] * g[:m_p] + bg[:m_p]
            neg[m_p:] = background("SURRXM", m - m_p, c["N"], rng)
            p_chunk = rng.permutation(m)
            neg = neg[p_chunk]

            X[idx_pos[i:i + m]] = torch.from_numpy(D.represent(c["arch"], pos, c.get("norm", True), sps=c["sps"]))
            y[idx_pos[i:i + m]] = 1.0
            X[idx_neg[i:i + m]] = torch.from_numpy(D.represent(c["arch"], neg, c.get("norm", True), sps=c["sps"]))
            y[idx_neg[i:i + m]] = 0.0
    else:
        for i in range(0, half, chunk):
            m = min(chunk, half - i)
            snr = rng.uniform(-20, 20, m)
            if c.get("signal") == "OFDM":
                sig = sim.ofdm(m, c["N"], rng, multipath=c.get("multipath", True))
            else:
                sig = sim.modulated(m, c["N"], rng, sps=c["sps"], beta=c["beta"], mods=tuple(c.get("mods") or sim.MODS))
            if c["recipe"] in MATCHED:
                pos = sig * np.sqrt(10 ** (snr / 10))[:, None] + background(c["recipe"], m, c["N"], rng)
            else:
                pos = sim.h1(sig, snr, rng)          # signal + unit-power white noise (unchanged)
            X[idx_pos[i:i + m]] = torch.from_numpy(D.represent(c["arch"], pos, c.get("norm", True), sps=c["sps"]))
            y[idx_pos[i:i + m]] = 1.0
        if c["N"] >= 8192:
            for i in range(0, n - half, chunk):
                m = min(chunk, n - half - i)
                neg_chunk = negatives(c["recipe"], m, c["N"], rng, sps_sig=c["sps"], beta_sig=c["beta"])
                X[idx_neg[i:i + m]] = torch.from_numpy(D.represent(c["arch"], neg_chunk, c.get("norm", True), sps=c["sps"]))
                y[idx_neg[i:i + m]] = 0.0
        else:
            neg_all = negatives(c["recipe"], n - half, c["N"], rng, sps_sig=c["sps"], beta_sig=c["beta"])
            for i in range(0, n - half, chunk):
                m = min(chunk, n - half - i)
                X[idx_neg[i:i + m]] = torch.from_numpy(D.represent(c["arch"], neg_all[i:i + m], c.get("norm", True), sps=c["sps"]))
                y[idx_neg[i:i + m]] = 0.0
    return X, y


def train(c, dev="cuda"):
    rtr, _, _ = sim.streams(c["seed"])
    torch.manual_seed(c["seed"])
    t0 = time.time()
    X, y = training_set(c, rtr)
    t_data = time.time() - t0
    nval = len(y) // 20
    Xv, yv, Xt, yt = X[:nval], y[:nval], X[nval:], y[nval:]
    model = D.ARCHS[c["arch"]]().to(dev)
    lr = c["lr"] if c["arch"] != "CLDNN" else min(c["lr"], 3e-4)
    opt = torch.optim.Adam(model.parameters(), lr)
    steps = c["epochs"] * int(np.ceil(len(yt) / c["batch"]))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    lf = nn.BCEWithLogitsLoss()
    use_amp = bool(c.get("amp", False) and dev == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    hist = []
    best = None   # (val_loss, epoch, state) when c["select"] == "best_val" (early stopping by validation loss)
    val_bs = 256 if c["N"] >= 8192 else (1024 if c["N"] >= 2048 else 2048)
    for ep in range(c["epochs"]):
        model.train()
        perm = torch.randperm(len(yt))
        tot = 0.0
        for i in range(0, len(yt), c["batch"]):
            idx = perm[i:i + c["batch"]]
            xb, yb = Xt[idx].to(dev, non_blocking=True), yt[idx].to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out = model(xb)
                loss = lf(out, yb)
            if use_amp:
                scale_before = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                if scale_before <= scaler.get_scale():
                    sch.step()
            else:
                loss.backward()
                opt.step()
                sch.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                vl = float(np.mean([lf(model(Xv[i:i + val_bs].to(dev)), yv[i:i + val_bs].to(dev)).item() for i in range(0, nval, val_bs)]))
        hist.append({"epoch": ep, "train_loss": tot / len(yt), "val_loss": vl})
        if c.get("select") == "best_val" and (best is None or vl < best[0]):
            best = (vl, ep, {k: v.detach().clone() for k, v in model.state_dict().items()})
    if best is not None:
        model.load_state_dict(best[2])
        model.eval()
    del X, y, Xt, yt
    return model, {"history": hist, "selected_epoch": best[1] if best else c["epochs"] - 1, "t_data_s": t_data,
                   "t_train_s": time.time() - t0 - t_data, "n_params": D.n_params(model), "lr_used": lr}


def learned_score_fn(model, arch, dev="cuda", bs=None, norm=True, sps=8):
    if bs is None:
        bs = 512 if "GapCNN" in arch else 2048

    def f(x):
        z = D.represent(arch, x, norm, sps=sps)
        out = []
        with torch.no_grad():
            for i in range(0, len(z), bs):
                out.append(model(torch.from_numpy(z[i:i + bs]).to(dev)).float().cpu().numpy())
        return np.concatenate(out)
    return f


def classical_score_fn(c):
    det = c["detector"]
    if det == "CYC":
        return lambda x: D.cyc(x, sps=c["sps"])
    if det == "BAND":
        if c.get("signal") == "OFDM":
            hw = 48.0 / 64.0 / 2.0
        else:
            hw = sim.occupancy(c["sps"], c["beta"]) / 2
        return lambda x: D.band_ratio(x, half_width=hw)
    return D.CLASSICAL[det]


# ----------------------------------------------------------------------------- one run
def run_one(cfg):
    c = dict(DEFAULTS); c.update(cfg)
    rid = run_id(c)
    out_json = os.path.join(ROOT, "RESULTS", "runs", rid + ".json")
    if os.path.exists(out_json):
        print("skip (exists)", rid, flush=True)
        return
    for d in ["RESULTS/runs", "RESULTS/scores", "EXPERIMENTS/ckpt"]:
        os.makedirs(os.path.join(ROOT, d), exist_ok=True)
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    rec = {"id": rid, "config": c, "date": datetime.datetime.now().isoformat(timespec="seconds"),
           "code_version": git_version(), "host": platform.node(), "python": platform.python_version(),
           "torch": torch.__version__, "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"}
    ckpt = ""
    if c["kind"] in ("learned", "whitening"):
        sig_str = f"_{c['signal']}" if c.get("signal") else f"_sps{c['sps']}"
        ckpt_path = os.path.join(ROOT, "EXPERIMENTS", "ckpt", f"{c['arch']}_{c['recipe']}_N{c['N']}{sig_str}_s{c['seed']}"
                                 + (f"_{c['tag']}" if c.get('tag') and c['kind'] == 'learned' else "") + ".pt")
        if c["kind"] == "whitening" and os.path.exists(ckpt_path):
            model = D.ARCHS[c["arch"]]().to(dev)
            model.load_state_dict(torch.load(ckpt_path, map_location=dev))
            model.eval()
            rec["training"] = {"loaded": ckpt_path}
        else:
            model, tr = train(c, dev)
            rec["training"] = tr
            torch.save(model.state_dict(), ckpt_path + ".tmp")     # atomic: concurrent readers never see a partial file
            os.replace(ckpt_path + ".tmp", ckpt_path)
        ckpt = os.path.relpath(ckpt_path, ROOT)
        fn = learned_score_fn(model, c["arch"], dev, norm=c.get("norm", True), sps=c["sps"])
    else:
        fn = classical_score_fn(c)
    t1 = time.time()
    if c["kind"] == "whitening":
        rec["whitening"] = E.evaluate_whitening(fn, c["N"], c["sps"], c["seed"], beta=c["beta"])
        raw = {}
    else:
        # long observations: halve evaluation sizes (Pf at 5 % keeps a ~+-1 pp 95 % CI)
        sz = {} if c["N"] < 1024 else {"n_cal": 10000, "n_cal_shape": 2000, "n_h0": 2000, "n_h1": 1000}
        res, raw = E.evaluate(fn, c["N"], c["sps"], c["seed"], beta=c["beta"], mods=tuple(c.get("mods") or sim.MODS),
                              signal=c.get("signal", "modulated"), multipath=c.get("multipath", True), **sz)
        rec["eval_sizes"] = sz or "default"
        rec["eval"] = res
    rec["t_eval_s"] = time.time() - t1
    rec["runtime_s"] = time.time() - t0
    rec["peak_vram_MB"] = (torch.cuda.max_memory_allocated() / 2 ** 20) if dev == "cuda" else 0.0
    rec["checkpoint"] = ckpt
    # atomic write: a power loss mid-write must not leave a truncated record that the resumable queue would skip
    with open(out_json + ".tmp", "w") as fh:
        json.dump(rec, fh, indent=1)
    os.replace(out_json + ".tmp", out_json)
    if raw:
        np.savez_compressed(os.path.join(ROOT, "RESULTS", "scores", rid + ".npz"), **raw)
    summ = summary_line(rec)
    with open(os.path.join(ROOT, "EXPERIMENTS", "runs.jsonl"), "a") as fh:
        fh.write(json.dumps({"id": rid, "date": rec["date"], "code": rec["code_version"], "runtime_s": round(rec["runtime_s"], 1),
                             "peak_vram_MB": round(rec["peak_vram_MB"], 1), "summary": summ}) + "\n")
    log_md(rec, summ)
    print(f"done {rid} {rec['runtime_s']:.0f}s {summ}", flush=True)
    if dev == "cuda":
        del fn
        torch.cuda.empty_cache()


def summary_line(rec):
    if "eval" not in rec:
        w = rec["whitening"]
        return "whitening: " + ", ".join(f"{k}:Pf={v['Pf']:.3f}" for k, v in w.items() if k.startswith("K16_"))
    e = rec["eval"]
    pf = e["Pf"]
    pd = e["Pd"]
    g = e["surrogate_gap"]
    return (f"Pf(lpf0.8)={pf['lpf0.8']['nominal']:.3f} Pf(rtl3dB)={pf['rtl3dB']['nominal']:.3f} Pf(pink)={pf['pink']['nominal']:.3f} "
            f"Pd(-10)={pd['-10']['white']['nominal']:.3f} surgap={g if g is None else round(g, 3)}")


def log_md(rec, summ):
    p = os.path.join(ROOT, "EXPERIMENTS", "EXPERIMENT_LOG.md")
    new = not os.path.exists(p)
    c = rec["config"]
    with open(p, "a", encoding="utf-8") as fh:
        if new:
            fh.write("# EXPERIMENT_LOG.md\n\nOne row per finished run, appended automatically by `CODE/run.py`. "
                     "Full records: `RESULTS/runs/<id>.json`; raw scores: `RESULTS/scores/<id>.npz`.\n\n"
                     "| id | date | code | kind | model/detector | recipe | N | sps | seed | ntrain/epochs/lr/batch | runtime s | device | peak VRAM MB | checkpoint | key metrics |\n"
                     "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        model = c.get("arch") if c["kind"] != "classical" else c.get("detector")
        hp = f"{c['ntrain']}/{c['epochs']}/{rec.get('training', {}).get('lr_used', c['lr'])}/{c['batch']}" if c["kind"] != "classical" else "-"
        fh.write(f"| {rec['id']} | {rec['date']} | {rec['code_version']} | {c['kind']} | {model} | {c.get('recipe','-') if c['kind']!='classical' else '-'} | "
                 f"{c['N']} | {c['sps']} | {c['seed']} | {hp} | {rec['runtime_s']:.0f} | {rec['device']} | {rec['peak_vram_MB']:.0f} | "
                 f"{rec['checkpoint'] or '-'} | {summ} |\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue")
    ap.add_argument("--one")
    a = ap.parse_args()
    cfgs = json.load(open(a.queue)) if a.queue else [json.loads(a.one)]
    for cfg in cfgs:
        try:
            run_one(cfg)
        except Exception as ex:  # keep the queue going; record the failure
            import traceback
            traceback.print_exc()
            with open(os.path.join(ROOT, "EXPERIMENTS", "failures.log"), "a") as fh:
                fh.write(f"{datetime.datetime.now().isoformat()} {json.dumps(cfg)} {ex!r}\n")



