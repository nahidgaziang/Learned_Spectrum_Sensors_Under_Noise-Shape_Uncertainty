# REPRODUCIBILITY.md

Code, configurations and results for the manuscript "Learned Spectrum Sensors Under Noise-Shape Uncertainty: An Invariant-Detection Analysis" and its supplementary material. Every number in the paper
is a LaTeX macro that `CODE/paper_numbers.py` writes from the files in `RESULTS/`; no number is typed by hand.
`CODE/verify_claims.py` and `CODE/independent_check.py` check the statements and numbers; their last outputs are
`RESULTS/claim_verification.json` and `RESULTS/independent_check.json`.

## 1. Environment
| Item | Version used |
|---|---|
| OS | Windows 11 Home (any OS with Python 3.12 should work) |
| GPU | NVIDIA GeForce RTX 3050 6 GB Laptop GPU (CUDA 12.4); CPU-only also works, slower |
| Python | 3.12.10 |
| PyTorch | 2.6.0+cu124 |
| NumPy / SciPy / pandas / matplotlib | 2.5.3 / 1.18.1 / 3.0.6 / 3.11.2 |

```
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install numpy scipy pandas matplotlib pymupdf
```

## 2. Data
- **Simulated data** (all main experiments) are generated on the fly by `CODE/sim.py`. Training, calibration and test
  sets come from independent seeds (`sim.streams`); nothing has to be downloaded.
- **Receiver 1 noise (R820T2)**: noise class of the public Hugging Face dataset `ishisan28/rf-signal-dataset`. The
  dataset card states no licence, so the data are used for evaluation only and are **not included** here. Download with
  `py -3.12 CODE/fetch_real_noise.py` (writes `DATA/real_noise_hf/`).
- **Receiver 2 noise and real signals (R828D)**: `TrevTron/rtl-ml-dataset` v2 (MIT licence). Download with
  `py -3.12 CODE/fetch_real_signals.py` (writes `DATA/trevtron/`). `CODE/check_overlap.py` confirms by SHA-256 that
  the two datasets share no samples.

## 3. What is in this archive
| Folder | Content |
|---|---|
| `CODE/` | simulation (`sim.py`), detectors (`detectors.py`), training and evaluation (`run.py`, `evaluate.py`), experiment scripts, figure/table/number generators, verification scripts |
| `CONFIGS/` | experiment queues (`queue_*.json`) for `run.py`; `make_queues.py` writes queues A–C |
| `RESULTS/runs/` | one JSON record per trained or evaluated configuration (configuration, code version, training history, runtime, peak memory, all metrics) |
| `RESULTS/*.json` | outputs of the experiment and analysis scripts listed in §5 |
| `TABLES/`, `FIGURES/` | generated tables (LaTeX/CSV) and figures (PDF) |

`RESULTS/scores/` holds the raw test scores of every run (used by `alpha_sensitivity.py`, `stats_ci.py` and
`independent_check.py`). Not included: downloaded data (§2) and model checkpoints (125 MB). Scripts that need
checkpoints (e.g. `wall_test.py`, `lmpi_equivalence.py`, `offband_sweep.py`) require re-running the training queues first. Some docstrings refer to internal planning notes ("the project plan",
"internal notes", step or task numbers); those notes are not part of the archive, and the paper and supplementary
material are the reference.

## 4. Commands (run from `CODE/`, in this order)
```
py -3.12 smoke_test.py                                             # sanity checks (powers, PSDs, classical Pf ~ 5 %)
# training and evaluation (resumable: a configuration whose RESULTS/runs/<id>.json exists is skipped)
py -3.12 ../CONFIGS/make_queues.py                                 # writes queues A-C
py -3.12 run.py --queue ../CONFIGS/queue_A_core.json               # architectures, classical detectors, recipes (N = 128)
py -3.12 run.py --queue ../CONFIGS/queue_B_wall.json               # N = 512, 2048
py -3.12 run.py --queue ../CONFIGS/queue_B1_learned_remaining.json
py -3.12 run.py --queue ../CONFIGS/queue_B2_classical_remaining.json
py -3.12 run.py --queue ../CONFIGS/queue_C_occupancy_whitening.json  # occupancy, reference whitening
py -3.12 run.py --queue ../CONFIGS/queue_D_detectnet_exact.json    # DetectNet replica (QAM16)
py -3.12 run.py --queue ../CONFIGS/queue_E_ablations.json          # training-set size, no power normalisation
py -3.12 run.py --queue ../CONFIGS/queue_F_resnet_es.json          # ResNet-1D with best-validation selection
py -3.12 run.py --queue ../CONFIGS/queue_G_surrx.json              # exact-surrogate negatives, N = 128/512/2048
py -3.12 run.py --queue ../CONFIGS/queue_I_matched.json            # matched-colour recipes
py -3.12 run.py --queue ../CONFIGS/queue_J_surrx_seeds.json        # extra seeds
py -3.12 run.py --queue ../CONFIGS/queue_K_escape.json             # long-window models (N = 8192) for the escape tests
py -3.12 run.py --queue ../CONFIGS/queue_L_scale.json              # 4x capacity, 600k examples
py -3.12 run.py --queue ../CONFIGS/queue_L_seeds.json              # extra seeds
py -3.12 run.py --queue ../CONFIGS/queue_M_ofdm.json               # OFDM with multipath fading
py -3.12 run.py --queue ../CONFIGS/queue_N_transformer.json        # attention-based sensor
# wall tests (need the checkpoints of the queues above)
py -3.12 wall_test.py 0 ; py -3.12 wall_test.py 1 ; py -3.12 wall_test.py 2
py -3.12 wall_test_long.py 0 ; py -3.12 wall_test_long.py 1 ; py -3.12 wall_test_long.py 2          # CPU, N up to 32768
py -3.12 wall_test_long_nets.py 0 ; py -3.12 wall_test_long_nets.py 1 ; py -3.12 wall_test_long_nets.py 2
py -3.12 wall_test_escape.py 0 ; py -3.12 wall_test_escape.py 1 ; py -3.12 wall_test_escape.py 2
py -3.12 wall_test_escape_controls.py 0 ; py -3.12 wall_test_escape_controls.py 1 ; py -3.12 wall_test_escape_controls.py 2
py -3.12 cyc_lag_ablation.py ; py -3.12 summarize_controls.py
# theory checks (CPU)
py -3.12 theorem_check.py                                          # achievability of the flat-band wall
py -3.12 wall_vs_theory.py ; py -3.12 summarize_wall_vs_theory.py  # wall across uncertainty and occupancy
py -3.12 wall_exact.py                                             # exact wall for arbitrary spectra (closed form vs LP)
py -3.12 u1_theory_check.py                                        # first-order surrogate indistinguishability
py -3.12 escape_law.py                                             # escape law of the cyclostationary detector
py -3.12 gaussian_ceiling.py                                       # most powerful Gaussian (second-order) detector
# mechanism experiments (GPU inference; need checkpoints)
py -3.12 lmpi_equivalence.py                                       # learned sensors vs the LMPI test
py -3.12 offband_sweep.py                                          # off-band signals
py -3.12 dynamics.py                                               # training dynamics
# real data (after the downloads in section 2)
py -3.12 real_noise_eval.py 128                                    # recorded noise, receiver 1 (see usage line for options)
py -3.12 empirical_uncertainty.py                                  # noise-shape uncertainty of both receivers
py -3.12 check_receiver_wall.py ; py -3.12 hardware_wall_budget.py # power vs shape uncertainty on the same receivers
py -3.12 adaptive_baselines.py                                     # AMF/ACE/Kelly/ED with reference noise
py -3.12 noise_gaussianity.py
py -3.12 real_signal_surrogate.py ; py -3.12 summarize_real_signal.py ; py -3.12 real_signal_lmpi.py
# statistics, tables, figures, numbers, checks
py -3.12 analysis.py ; py -3.12 stats_ci.py ; py -3.12 alpha_sensitivity.py ; py -3.12 supp_tables.py
py -3.12 summarize_ofdm.py ; py -3.12 compute_summary.py ; py -3.12 profile_complexity.py ; py -3.12 profile_transformer.py
py -3.12 paper_tables.py                                           # TABLES/*.tex
py -3.12 figures.py                                                # FIGURES/*.pdf
py -3.12 paper_numbers.py                                          # MANUSCRIPT/numbers.tex + RESULTS/paper_numbers.json
py -3.12 independent_check.py                                      # recomputes the numbers by a separate route
py -3.12 verify_claims.py                                          # checks the statements; needs the LaTeX sources of the paper
```
One GPU job at a time is the safest. Result files, wall-test files and checkpoints are written atomically.

Integrity tools (used after an interrupted run): `check_run_integrity.py "<from>/<to>" ...` lists runs that were in
progress during given time windows and compares them with sibling seeds; `check_real_noise_repro.py 40` replays the
real-noise evaluation exactly; `CONFIGS/queue_H_rerun_check.json` re-runs configurations with the tag `rerun`.

Exploratory scripts whose outputs are not used in the paper: `novelty_probes.py`, `novelty_probe_p2b.py`,
`novelty_probe_wallformula.py` and `novelty_probe_power_vs_shape.py` (preliminary feasibility probes, marked
PRELIMINARY in their JSON output; superseded by `wall_exact.py`, `lmpi_equivalence.py` and `hardware_wall_budget.py`;
`wall_exact.py` imports the LP solver `lp_wall` from `novelty_probes.py`), `whitening_probe*.py`, `preview_wall.py`,
`bench_speed.py`, `check_convergence*.py`, `inspect_detectnet.py`, `diag_real_noise.py`, `probe_real_signals.py`,
`scan_hf_dataset.py`.

## 5. Seeds and randomness
- Learned detectors: seeds 0–4 at N = 128 (more in queues J and L), 0–2 at N = 512, 2048 and in the occupancy runs.
  `torch.manual_seed(seed)` plus `sim.streams(seed)` for the training data.
- Evaluation draws: `sim.streams(10000 + seed)` (main), `20000 + seed` (whitening), `30000 + seed` (wall test;
  re-created per detector, so all detectors see the same draws), `40000 + seed` (long wall test). Calibration and test
  streams are independent.
- GPU training is not bit-exact across hardware and library versions. Regenerated numbers can differ in the last
  digit; per-seed values are in `RESULTS/runs/` and `RESULTS/summary_runs.csv`.
- Runtime on the RTX 3050: about 1 min per N = 128 run and 5–12 min per N = 2048 run; `RESULTS/compute_summary.json`
  totals the recorded runtimes.

## 6. Mapping from paper to code
| Paper item | Produced by |
|---|---|
| Fig. 1 (wall vs uncertainty) | `figures.py: fig_theory` (closed forms in `wall.py`) |
| Table I | `paper_tables.py` → `TABLES/tab_main.tex` ← `RESULTS/runs/*` (untagged runs) |
| Fig. 2 (surrogate test) | `figures.py: fig_surrogate` ← `RESULTS/runs/*` |
| Fig. 3 (learned templates vs LMPI) | `figures.py: fig_templates` ← `RESULTS/lmpi_equivalence.json` |
| Fig. 4 (scaling with N) | `figures.py: fig_nscaling` ← `RESULTS/runs/*` |
| Fig. 5 (recorded RTL-SDR noise) | `figures.py: fig_real_noise` ← `RESULTS/real_noise_v2_N128.json`, `real_noise_v3_N128.json` |
| Fig. 6 (wall tests, escape law) | `figures.py: fig_wall_journal` ← `RESULTS/wall_test_s*.json`, `wall_test_long_s*.json`, `wall_test_long_nets_s*.json`, `escape_law.json` |
| Theorems 1–2, Corollary 1 | `wall.py`; checks `theorem_check.py`, `wall_vs_theory.py`, `wall_test*.py` |
| Theorem 3, Corollary 2 | `wall_exact.py` → `RESULTS/wall_exact.json` |
| Proposition 3 | `u1_theory_check.py` → `RESULTS/u1_theory_check.json` |
| Proposition 4 (LMPI) | `lmpi_equivalence.py`, `real_signal_lmpi.py` |
| Supplement figures S-1 … S-5 | `empirical_uncertainty.py` (S-1); `figures.py`: `fig_masking`, `fig_wall_theory`, `fig_dynamics`, `fig_offband` |
| Supplement tables | `paper_tables.py` (`tab_alpha`, `tab_recipes`, `tab_real_signals`, `tab_whitening`, `tab_impl`, `tab_adaptive`, `supp_ci_fit`) |
| Every number in the text | `paper_numbers.py` → `MANUSCRIPT/numbers.tex` (+ `RESULTS/paper_numbers.json`) |
