# Pyroxene ML Thermobarometer (opx pipeline)

Reproducible analysis pipeline for the orthopyroxene machine-learning thermobarometer paper submitted to the Journal of Geophysical Research: Machine Learning and Computation.

## Project description

This repository contains the data-cleaning, training, evaluation, bias-correction, and figure-generation pipeline for an opx-only and opx-liq machine-learning thermobarometer benchmarked against the classical Putirka (2008) calibrations and against three published external machine-learning thermobarometers (Agreda-Lopez 2024, Jorgenson 2022, Wang 2021). Eight tuned model families plus TabPFN v2 are evaluated on the ExPetDB experimental corpus with citation-grouped 10-fold cross-validation; per-regime bias correction (Form A regime-piecewise OLS; Form B Agreda-Lopez quantile-thresholded piecewise) is fit and accepted under a pre-registered tolerance-band ship rule. The natural-sample validation is on the ArcPL Kd-equilibrated subset (n = 96) and on the LEPR two-pyroxene corpus.

## Data sources

Training data: ExPetDB experimental opx corpus, distributed under open-access terms by the original maintainers. Natural-sample evaluation data: ArcPL (literature P-T per sample) and LEPR (two-pyroxene pairs). All input parquet, csv, and split-index files are tracked at known SHA256 hashes recorded in `data/hashes.json`; the same hashes are cited in the manuscript Section 7 (Data and Code Availability).

## Reproducing the pipeline

The full pipeline is driven by `run_all.py`, which papermills the thirteen notebooks `notebooks/nb01_*.ipynb` through `notebooks/nbF_figures.ipynb` in order. A single canonical seed (42) is used for all stochastic steps; the 20-seed multiseed run uses seeds 42 through 61. The pre-registered constants and tests are in `tests/test_preregistration.py` and verify the ship-rule and per-regime claim-eligibility constants against `docs/preregistration/p_regime_preregistration.md`.

```
pip install -r requirements.txt
pip install -r requirements-tabpfn.txt   # in a separate environment per notes in the requirements file
python run_all.py
python -m pytest tests/
```

The pipeline takes approximately two CPU days end-to-end on a workstation-class machine with no GPU.

## Layout

`PROJECT_LAYOUT.md` shows the directory tree and module topology.

## License

The code in this repository is released under the MIT License. The raw experimental data follows the upstream license attached to ExPetDB and ArcPL; please consult those projects for redistribution terms.

## Citation

Citation information will be added on acceptance.

## Contact

Lead author and corresponding author information is in the manuscript front matter (`manuscripts/opx_2026/text/draft/sections/00a_front_matter.md`).
