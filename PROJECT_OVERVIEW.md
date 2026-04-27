# Project overview

Data flow and module topology for the opx ML thermobarometer pipeline. For operational instructions (install, run order), see [README.md](README.md). For the directory tree, see [PROJECT_LAYOUT.md](PROJECT_LAYOUT.md).

## What this repository contains

The training, evaluation, bias-correction, and figure-generation code for an orthopyroxene machine-learning thermobarometer benchmarked against Putirka (2008), Agreda-Lopez (2024), Jorgenson (2022), and Wang (2021). Eight tuned families plus TabPFN v2 are evaluated on the ExPetDB experimental corpus with citation-grouped 10-fold cross-validation and a 20-seed multiseed refit (seeds 42-61). Per-regime bias correction (Form A regime-piecewise OLS; Form B Agreda-Lopez quantile-thresholded piecewise) is fit and accepted under the pre-registered tolerance-band ship rule registered in [docs/preregistration/p_regime_preregistration.md](docs/preregistration/p_regime_preregistration.md).

The natural-sample evaluation is on the ArcPL Kd-equilibrated subset (n = 96) and on the LEPR two-pyroxene corpus.

## Data flow

```
data/raw/         data/external/
   |                  |
   v                  v
nb01 cleaning      external/* loaders
   |                  |
   v                  v
data/processed/   src/external_models.py wrappers
   |                  |
   +-> src/prepare_train_test.py (citation-grouped splits in data/splits/)
        |
        v
   src/models.py (8 tuned families) + nb03_tabpfn_baseline (TabPFN v2)
        |
        v
   nb04_putirka_benchmark + nb04_regime_benchmark + nb05_loso_validation
   nb06_shap_analysis + nb07_bias_correction + nb07b_arcpl_bias_probe
        |
        v
   results/*.csv  (multiseed RMSE, scorecard, ship verdicts, pairing matrix)
        |
        v
   nbF_figures + scripts/figures/*  ->  figures/core/Core_*.{pdf,png}
        |
        v
   nb09_manuscript_compilation + scripts/manuscript/build_manuscript_docx.py
        |
        v
   manuscripts/opx_2026/arxiv_submission/manuscript.docx
```

## Pre-registered constants

Locked in [docs/preregistration/p_regime_preregistration.md](docs/preregistration/p_regime_preregistration.md):

- Pressure-regime bin edges (kbar): `[0, 5, 15, 30, 100]`
- Sample-size floor: `N_MIN_FOR_VETO = 20`
- Ship rule absolute tolerances: `T_ABS_T = 10.0` °C; `T_ABS_P = 1.0` kbar
- Ship rule relative tolerance: `T_REL = 0.10`
- Numerical floor: `SHIP_TOL = 1e-6`

These constants are asserted by `tests/test_preregistration.py` (T15-T22) against the locked document.

## Reproducibility

All input data files have SHA256 hashes recorded in [data/hashes.json](data/hashes.json) and cited in manuscript Section 7. The reviewer-runnable test suite at [tests/](tests/) verifies pre-registration adherence and canonical-CSV integrity. The papermill driver [run_all.py](run_all.py) executes the thirteen notebooks in dependency order; total wall-clock cost is approximately two CPU days on a workstation-class machine without GPU.
