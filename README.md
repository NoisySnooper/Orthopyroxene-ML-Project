# opx ML thermobarometer

Machine-learning thermobarometer for orthopyroxene compositions. Predicts
pressure (kbar) and temperature (C) of equilibrium from orthopyroxene oxide
chemistry, with and without a paired liquid. Companion manuscript targeting
*JGR ML & Computation*.

## Pipeline layout

```
config.py             # every path, constant, and seed (single source of truth)
src/
  features.py         # raw / alr / pwlr feature engineering + EPMA noise aug
  models.py           # RF/ERT/XGB/GB factories, predict_median, predict_iqr
  data.py             # cleaned-data loaders + canonical model bookkeeping
  evaluation.py       # metrics, LOSO / Cluster-KFold / Gridded-PT splits
  geotherm.py         # Hasterok & Chapman 2011 layered geotherm
  io_utils.py         # save_figure, save_table, with_progress (tqdm)
  plot_style.py       # Tol palette, rcParams preset, axis helpers
notebooks/
  nb01_data_cleaning.ipynb
  nb02_eda_pca.ipynb                # writes opx_clean_core_with_clusters.parquet
  nb03_baseline_models.ipynb        # 3-method x 4-model multi-seed benchmark
  nb04_putirka_benchmark.ipynb
  nb04b_lepr_arcpl_validation.ipynb
  nb05_loso_validation.ipynb        # LOSO + Cluster-KFold + Gridded-PT
  nb06_shap_analysis.ipynb          # SHAP + robustness appendix
  nb07_bias_correction.ipynb        # QRF pressure-range correction
  nb08_natural_samples.ipynb
  nb09_manuscript_compilation.ipynb
  nb10_extended_analyses.ipynb
  nbF_figures.ipynb                 # canonical manuscript figures 1-14
```

Everything downstream of NB03 reads the Phase 3R winning feature set from
`results/nb03_winning_configurations.json`. No feature method name is
hardcoded in any notebook downstream of NB03.

## Running the pipeline

```
python -m venv .venv
.venv\Scripts\activate            # Windows bash
pip install -r requirements.txt
```

Then run notebooks in order:

1. `nb01_data_cleaning` -> `data/processed/opx_clean_core.parquet`
2. `nb02_eda_pca` -> `opx_clean_core_with_clusters.parquet` + EDA figures
3. `nb03_baseline_models` -> winning config, canonical models, test prediction arrays
4. `nb04_putirka_benchmark` (CPxOpx thermobar benchmark)
5. `nb04b_lepr_arcpl_validation` (external validation on ArcPL)
6. `nb05_loso_validation` (three grouped-CV strategies)
7. `nb06_shap_analysis` (SHAP + robustness checks)
8. `nb07_bias_correction` (QRF pressure-range correction)
9. `nb08_natural_samples` (natural Lin 2023 peridotite dataset)
10. `nb10_extended_analyses` (two-pyroxene, H2O, IQR, MC, OOD)
11. `nb09_manuscript_compilation` (tables 1-10)
12. `nbF_figures` (all 14 canonical figures)

NB01 through NB03 are mandatory for everything else; the rest are independent
given the NB03 outputs.

## Key design decisions

- **One source of truth for features and prediction.** `src/features.py` and
  `src/models.py` hold the canonical implementations. Notebooks import from
  `src.` - no duplicated `make_pwlr_features` or `predict_median` anywhere.
- **No hardcoded feature set.** NB03 benchmarks raw / alr / pwlr across 10
  split seeds and writes the winning method to `nb03_winning_configurations.json`.
  Every downstream notebook reads that JSON.
- **Augmentation disabled by default** (`N_AUG = 1`). The NB03 appendix
  documents the sensitivity test that justified dropping the `_aug` variants.
- **Canonical splits and seeds.** `config.SEED_SPLIT = 42` drives every
  `GroupShuffleSplit` call. Test indices are persisted in
  `data/splits/test_indices_opx_liq.npy` and `test_indices_opx.npy`.
- **Figures at 300 dpi.** `src.io_utils.save_figure` writes PNG, PDF, and an
  optional caption TXT. NBF uses this helper for the canonical figure set.

## Data sources

- **Training core:** ExPetDB opx + paired liquid, Putirka 2008 KD filter,
  Wo < 5 mol% orthopyroxene cut. See `nb01_data_cleaning` for the full
  quality-control chain.
- **External validation:** LEPR Wet Stitched April 2023 (ArcPL subset).
- **Natural samples:** Lin et al. 2023 NE China peridotite xenoliths (from
  the author-provided Supplementary Table S2).

## License and citation

This repository supports a manuscript in preparation. Cite via the
forthcoming DOI once published. Source code is MIT licensed.
