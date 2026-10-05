"""Integrity check after GPU-driver events / power-source changes.

For every run record: (1) the JSON loads, (2) training losses are finite, (3) the run's time interval is
reconstructed (file mtime = end, runtime_s = duration) and runs overlapping the given event windows are listed,
(4) each overlapping run's key metrics are compared with its sibling seeds (same model/recipe/N/sps/tag).
Writes RESULTS/run_integrity.json.
Usage: py -3.12 check_run_integrity.py "2026-10-02 03:09:38/2026-10-02 03:11:22" [...more windows]
"""
import datetime as dt, glob, json, math, os, re, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYS = [("Pf", "lpf0.8"), ("Pf", "rtl3dB"), ("Pf", "pink"), ("Pd-10", None), ("PdSur0", None)]


def metrics(r):
    e = r.get("eval")
    if not e:
        return {}
    out = {}
    for k, sh in KEYS:
        try:
            if k == "Pf":
                out[f"Pf_{sh}"] = e["Pf"][sh]["nominal"] if isinstance(e["Pf"][sh], dict) else e["Pf"][sh]
            elif k == "Pd-10":
                out["Pd_-10"] = e["Pd"]["-10"]["white"]["nominal"]
            elif k == "PdSur0":
                out["PdSur_0"] = e["Pd_surrogate"]["0"]["nominal"]
        except (KeyError, TypeError):
            pass
    return out


def main(windows):
    wins = []
    for w in windows:
        a, b = w.split("/")
        wins.append((dt.datetime.fromisoformat(a), dt.datetime.fromisoformat(b)))
    recs, bad = {}, []
    for f in sorted(glob.glob(os.path.join(ROOT, "RESULTS", "runs", "*.json"))):
        try:
            r = json.load(open(f))
        except Exception as ex:                      # truncated / corrupt file
            bad.append((os.path.basename(f), f"unreadable: {ex}"))
            continue
        h = (r.get("training") or {}).get("history") or []
        for x in h:
            for v in x.values():
                if isinstance(v, float) and not math.isfinite(v):
                    bad.append((r["id"], "non-finite training value"))
                    break
        end = dt.datetime.fromtimestamp(os.path.getmtime(f))
        start = end - dt.timedelta(seconds=float(r.get("runtime_s") or 0))
        recs[r["id"]] = dict(start=start, end=end, m=metrics(r), cfg=r.get("config", {}))

    def family(rid):
        return re.sub(r"_s\d+(?=(_|$))", "_s*", rid)

    report = {"unreadable_or_nonfinite": bad, "windows": []}
    for (a, b) in wins:
        hits = [rid for rid, v in recs.items() if v["start"] <= b and v["end"] >= a]
        rows = []
        for rid in hits:
            fam = family(rid)
            sib = [x for x in recs if x != rid and family(x) == fam]
            row = {"id": rid, "start": recs[rid]["start"].isoformat(timespec="seconds"),
                   "end": recs[rid]["end"].isoformat(timespec="seconds"), "n_siblings": len(sib), "metrics": {}}
            for k, v in recs[rid]["m"].items():
                sv = [recs[x]["m"].get(k) for x in sib if recs[x]["m"].get(k) is not None]
                row["metrics"][k] = {"run": round(v, 4),
                                     "siblings_min": round(min(sv), 4) if sv else None,
                                     "siblings_max": round(max(sv), 4) if sv else None}
            rows.append(row)
        report["windows"].append({"from": a.isoformat(), "to": b.isoformat(), "runs": rows})
    json.dump(report, open(os.path.join(ROOT, "RESULTS", "run_integrity.json"), "w"), indent=1)
    print(f"{len(recs)} run records loaded; unreadable/non-finite: {bad or 'none'}")
    for w in report["windows"]:
        print(f"\nwindow {w['from']} .. {w['to']}: {len(w['runs'])} run(s) in progress")
        for row in w["runs"]:
            print(f"  {row['id']}  ({row['start'][11:]}-{row['end'][11:]}, {row['n_siblings']} siblings)")
            for k, v in row["metrics"].items():
                lo, hi = v["siblings_min"], v["siblings_max"]
                flag = "" if lo is None or (lo - 0.05 <= v["run"] <= hi + 0.05) else "   <-- outside sibling range +-5 pts"
                print(f"      {k:12s} run {v['run']:.3f}   siblings [{lo}, {hi}]{flag}")


if __name__ == "__main__":
    main(sys.argv[1:])
