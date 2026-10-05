#!/usr/bin/env python3
"""Re-run the verification and the analysis from the stored run records (no GPU, no downloads).

    python reproduce.py --verify     # claim checks (CODE/verify_claims.py) and independent recomputation (CODE/independent_check.py)
    python reproduce.py --analysis   # tables, figures and MANUSCRIPT/numbers.tex from RESULTS/
    python reproduce.py --all        # both

Full re-training and the real-data experiments: see REPRODUCIBILITY.md."""
import argparse, os, subprocess, sys

CODE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CODE")


def step(script):
    print(f"\n=== {script}", flush=True)
    r = subprocess.run([sys.executable, script], cwd=CODE)
    if r.returncode:
        sys.exit(f"{script} failed with exit code {r.returncode}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--analysis", action="store_true")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if not (a.verify or a.analysis or a.all):
        ap.print_help(); sys.exit(0)
    if a.analysis or a.all:
        for s in ("analysis.py", "paper_tables.py", "figures.py", "paper_numbers.py"):
            step(s)
    if a.verify or a.all:
        for s in ("verify_claims.py", "independent_check.py"):
            step(s)
