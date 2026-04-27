# Project layout

Directory tree and module topology for the opx ML thermobarometer repository, regenerated at the close of the pre-submission audit.

## Top level

```
Final Project/
├── README.md                            # install, run order, design decisions
├── PROJECT_OVERVIEW.md                  # status log + module topology
├── PROJECT_LAYOUT.md                    # this file
├── config.py                            # paths, seeds, pre-registered constants
├── run_all.py                           # papermill pipeline driver
├── requirements.txt                     # core dependencies
├── requirements-tabpfn.txt              # TabPFN v2 add-on (separate venv)
├── .gitignore
│
├── data/                                # read-only training and natural-sample data
│   ├── raw/                             # ExPetDB, LEPR, external train sets
│   ├── processed/                       # cleaned per-track parquet files
│   ├── splits/                          # canonical test-index NPYs
│   ├── external/                        # Agreda, Jorgenson, Wang artifacts
│   ├── natural/                         # ArcPL natural-sample inputs
│   ├── optuna_studies/                  # Optuna hyperparameter-search DBs
│   └── hashes.json                      # SHA256 manifest cited in §7
│
├── src/                                 # library code
│   ├── bias_correction.py               # Form A and Form B + ship_decision
│   ├── calibration.py                   # split conformal
│   ├── data.py                          # per-track loaders
│   ├── ensembles.py                     # Ridge stacking
│   ├── evaluation.py                    # bootstrap CIs, regime assignment
│   ├── external_models.py               # Putirka, Agreda, Jorgenson, Wang wrappers
│   ├── features.py                      # opx feature construction
│   ├── geotherm.py                      # equilibrium-flag helpers
│   ├── io_utils.py                      # CSV, JSON, joblib helpers
│   ├── models.py                        # 8 tuned families + TabPFN
│   ├── opx_tb_analysis.py               # opx-side benchmark utilities
│   ├── plot_style.py                    # Okabe-Ito palette + figure defaults
│   ├── prepare_train_test.py            # citation-grouped split builder
│   ├── resampling.py                    # tempered class resampling
│   ├── stacking.py                      # Ridge meta-learner
│   ├── thermobar_adapter.py             # Thermobar v1.0.70 K to C contract
│   ├── ablations/                       # ablation experiments
│   └── external/                        # external-model adapters (ArcPL etc.)
│
├── scripts/                             # pipeline scripts
│   ├── ablations/                       # feature ablation drivers
│   ├── audits/                          # repo audit utilities
│   ├── benchmarks/                      # benchmark runners
│   ├── bias_correction/                 # Form A/B fitting + canonical rescore
│   ├── data_prep/                       # cleaning drivers
│   ├── evaluation/                      # multi-seed eval, scorecard build
│   ├── external_eval/                   # external-method evaluation
│   ├── figures/                         # core and SI figure builders
│   ├── interpretability/                # SHAP, classical-equivalence regression
│   ├── manuscript/                      # docx build, citation linking
│   ├── pairing/                         # LEPR pairing-matrix scripts
│   ├── shap/                            # SHAP attribution drivers
│   └── tabpfn/                          # TabPFN inference drivers
│
├── notebooks/                           # papermill-driven analysis notebooks
│   ├── nb01_data_cleaning.ipynb
│   ├── nb02_eda_pca.ipynb
│   ├── nb03_opx_baseline_models.ipynb
│   ├── nb03_tabpfn_baseline.ipynb
│   ├── nb04_putirka_benchmark.ipynb
│   ├── nb04_regime_benchmark.ipynb
│   ├── nb04b_aug_test.ipynb
│   ├── nb05_loso_validation.ipynb
│   ├── nb06_shap_analysis.ipynb
│   ├── nb07_bias_correction.ipynb
│   ├── nb07b_arcpl_bias_probe.ipynb
│   ├── nb09_manuscript_compilation.ipynb
│   └── nbF_figures.ipynb
│
├── tests/                               # reviewer-runnable test suite
│   ├── test_preregistration.py          # T1-T22 ship rule + constants
│   └── ...
│
├── models/                              # canonical model joblibs
│   ├── canonical/opx/                   # 8 families x 4 cells (T/P x liq/only) joblibs
│   ├── ablation/                        # ablation checkpoints
│   └── external/                        # third-party model artifacts
│
├── results/                             # canonical CSVs and JSONs
│   ├── bias_correction_shipped.csv      # canonical ship verdicts
│   ├── preregistered_scorecard_postcorrection.csv  # canonical scorecard
│   ├── opx_multiseed_*.csv              # 20-seed RMSE statistics
│   ├── pairing_matrix_22rows.csv        # LEPR pairing comparisons
│   └── ...
│
├── figures/
│   ├── core/                            # manuscript-bound figures + AUDIT.md
│   ├── SI/                              # supplementary figures
│   └── (no interactive maps post-audit)
│
├── tables/                              # manuscript tables (Tables 1, S8 etc.)
│
├── manuscripts/
│   └── opx_2026/                        # JGR:MLC submission
│       ├── text/draft/sections/         # 00a-09 manuscript prose
│       ├── figures/                     # publication-bound figure copies
│       ├── tables/                      # publication-bound tables
│       ├── arxiv_submission/            # built docx + arXiv tarball
│       └── SUBMISSION_CHECKLIST.md
│
├── docs/preregistration/                # locked methodological documents
│   ├── p_regime_preregistration.md      # bin edges + ship rule
│   └── nb03_test_protocol.md
│
├── archive/                             # one retained snapshot
│   ├── README.md
│   └── pre_v10_rebuild_2026_04_16/      # nb07 q_hat_P re-derivation source
│
└── logs/                                # per-run execution logs (gitignored content)
    └── .gitkeep
```

## Module topology

The training pipeline runs in a single direction: `data/processed/` is consumed by `src/prepare_train_test.py` to produce citation-grouped splits, which feed the eight tuned families in `src/models.py` plus TabPFN v2 (loaded externally). Per-seed RMSE statistics flow into `results/opx_multiseed_summary.csv`. Bias correction (`src/bias_correction.py`) consumes the per-seed table to produce per-cell winners under the locked ship rule, written to `results/bias_correction_shipped.csv` and the canonical scorecard `results/preregistered_scorecard_postcorrection.csv`. Figures are built from the canonical CSVs by scripts in `scripts/figures/`; manuscript prose is compiled by `scripts/manuscript/build_manuscript_docx.py` into `manuscripts/opx_2026/arxiv_submission/manuscript.docx`.

The pre-registered constants (ship rule, regime edges, sample-size floor) live in `config.py` and `docs/preregistration/p_regime_preregistration.md`. The reviewer-runnable test suite at `tests/test_preregistration.py` asserts the constants and CSV integrity against the locked document.
