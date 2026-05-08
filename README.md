# Pyroxene ML Thermobarometer (opx pipeline)

Reproducible analysis pipeline for the orthopyroxene machine-learning thermobarometer paper submitted to the Journal of Geophysical Research: Machine Learning and Computation.

## Project description

This repository contains the data-cleaning, training, evaluation, bias-correction, and figure-generation pipeline for an opx-only and opx-liq machine-learning thermobarometer benchmarked against the classical Putirka (2008) calibrations and against three published external machine-learning thermobarometers (Agreda-Lopez 2024, Jorgenson 2022, Wang 2021). Eight tuned model families plus TabPFN v2 are evaluated on the ExPetDB experimental corpus with citation-grouped 20-seed cross-validation; per-regime bias correction (Form A regime-piecewise OLS; Form B Agreda-Lopez quantile-thresholded piecewise) is fit and accepted under a pre-registered tolerance-band ship rule. The natural-sample validation is on the ArcPL Kd-equilibrated subset (n = 96) and on the LEPR two-pyroxene corpus.

## Data sources

Training data: ExPetDB experimental opx corpus, distributed under open-access terms by the original maintainers. Natural-sample evaluation data: ArcPL (literature P-T per sample) and LEPR (two-pyroxene pairs). All input parquet, csv, and split-index files are tracked at known SHA256 hashes recorded in `data/hashes.json`; the same hashes are cited in the manuscript Section 7 (Data and Code Availability).

## Install

```
pip install -r requirements.txt
pip install -r requirements-tabpfn.txt   # in a separate environment per notes in the requirements file
```

Two virtual environments are expected at the project root: `.venv/` (main) and `.venv-tabpfn/` (TabPFN v2 add-on). The `nb03_tabpfn_baseline.ipynb` kernel is the main `.venv/`; the notebook itself shells out to `.venv-tabpfn/` via subprocess for the foundation-model inference.

## Reproducing the pipeline

Run the notebooks in this order. A single canonical seed (42) is used for all stochastic steps; the 20-seed multiseed run uses seeds 42 through 61.

1. `nb01_data_cleaning` - ExPetDB ingest, Kd-equilibrium and oxide-total filtering, parquet writes
2. `nb02_eda_pca` - exploratory PCA, regime-stratified summaries
3. `nb03_opx_baseline_models` - 8 tuned families with Optuna search and 20-seed refit
4. `nb03_tabpfn_baseline` - TabPFN v2 ninth-family evaluation (run after nb03 baseline; before nb04 and nb07)
5. `nb04_putirka_benchmark` - Putirka 2008 28a/29a/29c head-to-head
6. `nb04_regime_benchmark` - regime-stratified scorecard generation
7. `nb04b_aug_test` - 15x Gaussian augmentation sensitivity (requires `scripts/ablations/run_augmentation_ablation_opx.py` and `scripts/ablations/run_augmentation_oof_bias.py` to run first; that compute is ~3.3h plus ~30 min)
8. `nb05_loso_validation` - leave-one-citation-out validation
9. `nb06_shap_analysis` - SHAP attribution per cell winner
10. `nb07_bias_correction` - Form A and Form B fitting plus ship decision
11. `nb07b_arcpl_bias_probe` - ArcPL natural-sample bias probe
12. `nb09_manuscript_compilation` - numeric autofill for manuscript prose

Each notebook reads the upstream parquet/CSV/JSON, calls into `src/`, and writes its outputs to `results/` and `models/canonical/`.

After the notebooks, build figures and the manuscript:

```
python scripts/figures/make_fig_dataset_map.py
python scripts/figures/make_fig_methods_flowchart.py
# ... see scripts/figures/ and scripts/pairing/ for the full set
python scripts/manuscript/build_manuscript_docx.py
```

The figure builders write to `figures/opx_only/` as `main_fig_{1..14}.{pdf,png,txt}` and `supp_fig_*.{pdf,png,txt}`. The build script reads those PNGs plus `manuscripts/opx_2026/text/draft/sections/00..09.md` and writes `manuscripts/opx_2026/arxiv_submission/manuscript.docx`.

End-to-end wall clock is approximately two CPU days on a workstation-class machine with no GPU.

## Note on `run_all.py`

A papermill driver `run_all.py` is checked in but currently stale: its notebook list references `nb03_cpx_baseline_models`, `nb03_twopx_baseline_models`, and `nb08_natural_twopx`, which were removed during the opx-only trim, and omits `nb04_regime_benchmark` and `nb07b_arcpl_bias_probe`. Run the notebooks manually in the order above until the driver is repaired.

## Tests

```
python -m pytest tests/
```

The test suite covers pre-registration adherence (T1-T22), bias-correction unit logic, bootstrap CIs, regime assignment, train-test prep parity, and the TabPFN model registry. T15-T22 in `tests/test_preregistration.py` assert the ship-rule and per-regime claim-eligibility constants against `docs/preregistration/p_regime_preregistration.md`.

## Layout

`PROJECT_LAYOUT.md` shows the directory tree. `PROJECT_OVERVIEW.md` describes the data flow and module topology.

## License

The code in this repository is released under the MIT License. The raw experimental data follows the upstream license attached to ExPetDB and ArcPL; please consult those projects for redistribution terms.

## Citation

Citation information will be added on acceptance.

## Contact

Lead author and corresponding author information is in the manuscript front matter (`manuscripts/opx_2026/text/draft/sections/00a_front_matter.md`).
