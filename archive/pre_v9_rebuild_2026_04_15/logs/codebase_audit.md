# Codebase audit — opx ML thermobarometer (post-v6, pre-20-seed rerun)

Audited: 2026-04-14. Scope: read-only review of notebooks/, src/, scripts/, config.py, results/, models/, figures/, logs/. No files were modified during this audit. Line numbers cite cell source as rendered by `nbformat` (the line within the code cell, not the raw `.ipynb` JSON line).

## Executive summary

The v6 consolidation is structurally clean: 11 live notebooks, 56 result CSVs, 36 figures, 29 model joblibs, 5 archived v5 notebooks, and all live notebooks pass `nbformat.validate`. The pipeline can run end-to-end on the current branch.

However, there is one defect that propagates through every downstream notebook and invalidates the opx_liq P numbers the paper currently reports, and a second defect that silently picks a non-canonical model/feature-set combination for the "Ours" column of the ArcPL external validation and the v6 three-way benchmark. Both stem from the same root cause: `canonical_model_filename()` in [src/data.py:85-90](src/data.py#L85-L90) takes `model_name` as a free parameter and does not consult `per_combo_winners` from `nb03_winning_configurations.json`. Every caller must manually know which family won each (track, target) combo, and three of the six callers guess wrong.

A third issue, related but separate: nb04b's `get_or_train_rf` ([notebooks/nb04b_lepr_arcpl_validation.ipynb](notebooks/nb04b_lepr_arcpl_validation.ipynb) cell 9) implements its own "best RF" rule (filter to RF, pick min mean RMSE per target independently) that both contradicts the nb03 winner (XGB for P) and can select different feature sets for T and P within the same notebook (T=pwlr, P=raw on the current multi-seed table). The resulting `nb04b_arcpl_predictions.csv` is the input for both fig13 and the v6 three-way benchmark's "Ours" column.

The remaining findings are file-reference drift (one typo in nbF's file-audit list produces a false MISSING warning on every run), a stale canonical figure roster (v5 fig10 expected but never produced; v6 fig_nb04_three_way / fig_nb08_twopx_1to1 / fig_nb11_model_family_ceiling / fig_nb02_clusters not promoted into the roster), column-name inconsistency across CSVs, a heuristic K→C conversion in `src/external_models.py:191` that cannot actually distinguish K from C for magmatic temperatures, and a cluster of stale FAILURE_*.log files from pre-v6 runs that no longer reflect current pipeline state.

Before the 20-seed rerun, the CRITICAL and HIGH findings below should be fixed, because the rerun will re-execute every downstream cell and re-materialize the wrong-model artifacts.

---

## Issues by severity

### CRITICAL

**C1. Every downstream notebook loads RF for opx_liq P; the nb03 winner is XGB.**

[results/nb03_winning_configurations.json](results/nb03_winning_configurations.json) names `"opx_liq_P_kbar": {"model_name": "XGB", "filename": "model_XGB_P_kbar_opx_liq_pwlr.joblib"}`. The XGB model is present on disk. Every live notebook that loads the opx_liq P model loads RF instead:

- [notebooks/nb06_shap_analysis.ipynb:108-109](notebooks/nb06_shap_analysis.ipynb#L108-L109) — `canonical_model_filename('RF', 'P_kbar', 'opx_liq', RESULTS)`
- [notebooks/nb07_bias_correction.ipynb:142-143](notebooks/nb07_bias_correction.ipynb#L142-L143) — same call
- [notebooks/nb09_manuscript_compilation.ipynb:98-99](notebooks/nb09_manuscript_compilation.ipynb#L98-L99) — inserted by [scripts/merge_nb10_into_nb09.py:51-52](scripts/merge_nb10_into_nb09.py#L51-L52)
- [notebooks/nbF_figures.ipynb:180-181](notebooks/nbF_figures.ipynb#L180-L181) and [:461-462](notebooks/nbF_figures.ipynb#L461-L462) — both opx_only and opx_liq rows hardcode RF; RF is correct for opx_only but wrong for opx_liq

Downstream impact on the paper: SHAP importances for P, piecewise bias-correction A/B for P, conformal qhat for P, and every reported opx_liq P RMSE / residual / bias statistic were computed on the non-winning family. Fig 3 (pred-vs-obs P), Fig 7 (SHAP P beeswarm), Fig 9a (SHAP P dependence), Fig 11 (bias-correction residuals, P panel), Fig 13 (OOD vs residual P), Table 2, Table 5, Table 7 are all affected.

Root cause: [src/data.py:85-90](src/data.py#L85-L90). `canonical_model_filename(model_name, target, track, results_dir)` takes `model_name` as a parameter and only consults `config['global_feature_set']`. It does not look up `config['per_combo_winners'][f'{track}_{target}']['model_name']`.

Fix shape: either (a) change the function signature to `canonical_model_filename(target, track)` and read the per-combo winner internally, or (b) leave the current function and add a sibling `winning_model_name(target, track)` that every caller is required to use. Option (a) is safer because it removes the sharp edge entirely.

**C2. nb04b uses an ad-hoc RF-only selection rule and lands on RF+raw for P.**

[notebooks/nb04b_lepr_arcpl_validation.ipynb](notebooks/nb04b_lepr_arcpl_validation.ipynb) cell 9 filters `nb03_multi_seed_summary.csv` to `model_name=='RF'` and then picks the RF row with the lowest `rmse_test_mean` per target. On the current summary table this resolves to T_C=pwlr, P_kbar=raw. The global winner from nb03 is pwlr for both, and the per-combo winner for opx_liq_P_kbar is XGB+pwlr (not RF at all).

Consequences:
- `nb04b_arcpl_predictions.csv` columns `T_pred` (RF+pwlr) and `P_pred` (RF+raw) come from different feature sets; the CSV header does not record this.
- The v6 three-way ML benchmark ([scripts/build_nb04_three_way.py:156-166](scripts/build_nb04_three_way.py#L156-L166)) reads `ours_df['T_pred']`, `ours_df['P_pred']` from that same CSV, so the "Ours opx-liq RF" row in `results/nb04_three_way_ml_benchmark.csv` and panel (a)/(b)/(c) of `fig_nb04_three_way.{png,pdf}` report the RF+raw/RF+pwlr composite rather than the canonical XGB+pwlr / RF+pwlr combination.
- Fig 13 (OOD vs residual) and all ArcPL H2O-dependence plots inherit the same predictions.

Fix shape: drive nb04b off `per_combo_winners` like the other notebooks (once C1 is fixed) or explicitly document that nb04b uses the "best-RF" variant as a deliberate choice and regenerate the three-way "Ours" column from the true canonical models.

---

### HIGH

**H1. `verify_v6.py` checks model filenames that cannot exist under the v6 naming convention.**

[scripts/verify_v6.py:48-54](scripts/verify_v6.py#L48-L54) asserts that `models/model_RF_T_C_opx_liq.joblib` and `models/model_RF_P_kbar_opx_liq.joblib` exist. Canonical models are named with a feature-set suffix (`_pwlr`), so these literal filenames cannot exist and the verifier prints `Missing models: [...]` on every run. It is both a false positive and, more dangerously, it mirrors the same RF-hardcoding pattern that hides C1.

**H2. `nbF_figures.ipynb` cell 3 file audit refers to a filename that never exists.**

[notebooks/nbF_figures.ipynb:3](notebooks/nbF_figures.ipynb) cell 3 lists `RESULTS / 'nb03_winning_config.json'` (missing `urations`). The real file written by nb03 and named in `config.py:70` is `nb03_winning_configurations.json`. The audit always reports this row as `MISSING` and writes the misleading result to `results/nbF_file_audit.csv`.

**H3. Canonical figure roster is stale; fig10 never materializes; v6 figures are not promoted.**

`results/figure_inventory.csv` lists `fig10_putirka_vs_ml.png,False`. The generator cell exists at [notebooks/nbF_figures.ipynb](notebooks/nbF_figures.ipynb) cell 10 and the input CSV (`nb04_putirka_vs_ml.csv`) is present with the expected columns, but no `figures/fig10_putirka_vs_ml.{png,pdf}` is on disk. Meanwhile [notebooks/nb04_putirka_benchmark.ipynb](notebooks/nb04_putirka_benchmark.ipynb) produces `figures/fig_nb04_putirka_comparison.{png,pdf}` which effectively replaces the legacy fig10 but is not added to the canonical roster in [notebooks/nb09_manuscript_compilation.ipynb](notebooks/nb09_manuscript_compilation.ipynb) cells 22 and 25.

V6 figures not on the canonical roster:
- `figures/fig_nb02_clusters.png`
- `figures/fig_nb04_three_way.{png,pdf}`
- `figures/fig_nb08_twopx_1to1.{png,pdf}`
- `figures/fig_nb11_model_family_ceiling.{png,pdf}`
- `figures/fig_nb06_correlation_check.png`, `fig_nb06_proxy_check.png`

These appear under the nb09 "Additional figures" bonus gallery but do not have assigned figure numbers, captions, or positions in the manuscript.

**H4. nb03 split writes happen AFTER training uses a locally re-generated split.**

[notebooks/nb03_baseline_models.ipynb](notebooks/nb03_baseline_models.ipynb) cell 21 (Phase 3R.5) writes `train_indices_opx_liq.npy` and `test_indices_opx_liq.npy` to `data/splits/` at the end. But Phase 3R.3b (cell 11) and Phase 3R.4 (cell 13) regenerate the seed-42 split in memory from `GroupShuffleSplit(random_state=TUNE_SEED)` and do not consult the files on disk. The splits are numerically identical today, but any future change to the train-side split strategy that happens before the NPY files are written will cause downstream notebooks (which load the NPY files) to see a different split than what nb03's internal reporting used. Make the on-disk splits the single source of truth before training starts.

---

### MEDIUM

**M1. Fragile K→C heuristic in Thermobar extract.**

[src/external_models.py:191-194](src/external_models.py#L191-L194): `if celsius and np.nanmean(arr) > 400: arr = arr - 273.15`. Magmatic T in Celsius always has mean > 400 (training range 700-1500 C), so this heuristic cannot distinguish C from K when Thermobar returns a Series/ndarray. It currently works because today's Thermobar always returns Kelvin for T on array/Series returns — but that is a silent contract. If a future Thermobar release changes convention, every predicted T will be silently off by 273.15. Replace with an explicit check on the Thermobar function's documented return type per version, or always pass via a DataFrame path which has unambiguous column names.

**M2. `eval(json.load(f))` on vendor bias files.**

[src/external_models.py:77](src/external_models.py#L77): the Agreda-Lopez `.json` bundles store a Python dict literal inside a JSON string (confirmed by reading `models/external/agreda_cpx_T.json`). `eval()` on trusted bundled files is not a security issue, but it is an unusual format and fragile against whitespace or format changes. `json.loads(json.load(f))` — or better, re-serialising the vendor files as proper JSON during ingestion — is safer.

**M3. CSV column-name inconsistency across 56 result CSVs.**

No single convention for predicted/observed T and P columns:
- `T_pred, P_pred`: nb04b_arcpl_predictions, nb08_natural_predictions_all, nb08_natural_predictions_filtered, nb10_arcpl_ood_scores
- `T_true, P_true`: nb07_test_predictions
- `T_C_true, P_kbar_true`: nb08_natural_predictions
- `y_T_true, y_P_true, T_pred_{MODEL}, P_pred_{MODEL}`: nb03_canonical_test_predictions
- `obs_T_C, obs_P_kbar, putirka_pred_T_C, putirka_pred_P_kbar, ml_pred_T_C, ml_pred_P_kbar`: nb04_putirka_vs_ml
- `T_ml_opx, T_jorgenson_cpx, T_putirka_2px_eq36`: nb08_natural_predictions
- `T_ml_opx_only, T_jorgenson, T_putirka_eq36`: nb10b_two_pyroxene_predictions (archived but still referenced)

Each notebook reads only its own outputs, so nothing is currently broken. But any cross-CSV join in future analysis will have to special-case every file. Recommend a one-time rename pass to the `{quantity}_{qualifier}` convention (e.g., `T_true`, `T_pred_ml_opx`, `T_pred_putirka_eq36`) before the 20-seed rerun materializes everything fresh.

**M4. Low markdown/code ratios in several notebooks.**

Per `scripts/verify_v6.py`:
- nb01_data_cleaning 0.06 (1 md / 16 code)
- nbF_figures 0.08 (1 md / 12 code)
- nb04b_lepr_arcpl_validation 0.18
- nb04_putirka_benchmark 0.22
- nb09_manuscript_compilation 0.24

For a JGR ML & Computation submission that will ship the notebooks as supplementary material, 0.06 means a reader has to read pure code to understand cleaning decisions. Adding a short markdown preamble to each phase of nb01 and nbF is cheap insurance.

**M5. Stale FAILURE_*.log files from pre-v6 runs.**

[logs/](logs/) contains:
- `FAILURE_nb04_putirka_benchmark.log` (2026-04-14T04:28)
- `FAILURE_nb09_manuscript_compilation.log` (2026-04-14T11:13)
- `FAILURE_nb10_extended_analyses.log` (2026-04-14T10:57)
- `FAILURE_nbF_figures.log` (2026-04-14T11:30)

All four point to `pipeline_resume_*` log files that also exist in `logs/`. The most recent successful resume (`pipeline_resume_20260414_113152.log`) shows nb09 passing at 11:32:24, superseding all of these. The FAILURE files are pointers to fixed failures and should be moved to `logs/archive/` or deleted so a future reader does not conclude the pipeline is currently broken.

**M6. nb05 Gridded-PT CV uses target-value bins as fold labels.**

[notebooks/nb05_loso_validation.ipynb](notebooks/nb05_loso_validation.ipynb) cell 5: `pt_grid = qcut(T) + '_' + qcut(P)` is then passed to `cluster_kfold_splits`, which is `GroupKFold(n_splits=unique_count)`. This is a target-stratified CV (folds are defined by the label being predicted), not a spatial/regime CV. It may overstate extrapolation difficulty or, depending on how bins fall, be nearly redundant with a random split. Acceptable as a stress test if documented, but the "Gridded-PT" name implies something it is not.

**M7. Manuscript roster in nb09 references `nb10_*` CSVs whose producing notebook is archived.**

After v6, `nb10_extended_analyses.ipynb` is in `archive/` and its analyses are re-produced inside nb09's Phase 9R.EXT block (inserted by `scripts/merge_nb10_into_nb09.py`). The CSV filenames still carry the `nb10_` prefix by design (to keep manuscript references stable) but the documentation and nbF's file audit still refer to "nb10" as the owner. Low confusion cost today, but if a future contributor looks for an `nb10` notebook to regenerate a file, they will not find one.

---

### LOW

**L1. Hardcoded `default_rng(42)` instead of config constants.**

Several sites use a numeric seed instead of pulling from `config.py`:
- [notebooks/nb04_putirka_benchmark.ipynb](notebooks/nb04_putirka_benchmark.ipynb) cells containing `BOOT_RNG = np.random.default_rng(42)` and `rng = np.random.default_rng(42)`
- [notebooks/nb04b_lepr_arcpl_validation.ipynb](notebooks/nb04b_lepr_arcpl_validation.ipynb) `_rng = np.random.default_rng(42)`
- [scripts/build_nb04_three_way.py:173](scripts/build_nb04_three_way.py#L173)

All config seeds in `config.py:40-43` are 42, so behavior is currently identical. But "change SEED_SPLIT to 43 and see what breaks" will miss these sites.

**L2. `canonical_model_filename` naming is misleading.**

The function returns a filename given a caller-supplied `model_name`, so it is really `formatted_model_filename`. True canonicalization would not accept `model_name` as an argument. Rename or add a stricter API (see C1 fix shape).

**L3. Orphan figure from archived notebook.**

`figures/fig_nb10b_two_pyroxene.{png,pdf}` is produced by `archive/nb10b_two_pyroxene_benchmark_v5_superseded_by_nb08.ipynb`. Its v6 replacement is `figures/fig_nb08_twopx_1to1.{png,pdf}` (both present). Delete the orphan or move to `figures/archive/`.

**L4. `models/canonical/` directory is empty.**

The directory exists (auto-created via a path mkdir somewhere) but contains no files. If the intent is to symlink or copy the per-combo-winners there after C1 is fixed, leave it; otherwise remove to avoid implying a convention that does not exist.

**L5. Duplicate deduplicated inventory between `nb09` expected roster and `nbF` generator.**

The list of numbered figures appears twice, once in [notebooks/nb09_manuscript_compilation.ipynb](notebooks/nb09_manuscript_compilation.ipynb) cell 22 (inventory check) and once implicitly in [notebooks/nbF_figures.ipynb](notebooks/nbF_figures.ipynb) `save_figure` calls. When H3 is fixed, dedupe these by reading a single list from `config.py` or a roster JSON.

**L6. Heterogeneous `duplicates='drop'` behavior in `qcut`.**

`pt_grid_labels` in nb05 and `y_bins = qcut(y_tr, q=5, ...)` in nb03 both use `duplicates='drop'`, which silently reduces the effective number of bins when the target has heavy ties. Log the realized bin count whenever this is used so the verification script can detect unusual dropouts.

---

## Findings by category

### 1. Canonical model consistency
- Covered by C1, C2. Canonical-model bookkeeping is scattered across three places: `per_combo_winners` in the JSON (correct, not consulted), `canonical_model_filename()` in `src/data.py` (consults global feature set only), and ad-hoc RF selection in nb04b (contradicts both). Consolidate into a single API.

### 2. Train/test split consistency
- opx_liq splits: 426 train / 174 test, disjoint (verified earlier session). opx splits: 845 train / 190 test, disjoint. See H4 for ordering issue (files written at end of nb03 rather than start).
- nb05 consumes `df_liq` directly from parquet for LOSO/Cluster-KFold/Gridded-PT rather than using the nb03 split indices, which is correct for a pooled CV strategy comparison.

### 3. Feature set consistency
- Global winning feature set is `pwlr` per `nb03_winning_configurations.json`. nb05, nb06, nb07, nb08, nb09, nbF all pull `WIN_FEAT = config['global_feature_set']` at import time. The only live exception is nb04b (see C2).
- `src/features.py:build_feature_matrix(df, feature_set, use_liq)` dispatches on the feature_set string; the same engineered features (`Mg_num, Al_IV, Al_VI, En_frac, Fs_frac, Wo_frac, MgTs`) are merged into every representation.

### 4. Stale file references
- H2: `nb03_winning_config.json` typo in nbF cell 3.
- M7: `nb10_` CSV prefix after nb10 archival.
- L3: `fig_nb10b_two_pyroxene.{png,pdf}` orphan.
- M5: four FAILURE_*.log files.
- nbF cell 3 also lists `nb10_arcpl_ood_scores.csv` and `nb10_mc_per_sample.csv` with owner `nb10`; both files exist (produced by nb09 Phase 9R.EXT) but the owner label is wrong.

### 5. Seed handling
- `config.py:40-43` defines `SEED_SPLIT, SEED_MODEL, SEED_NOISE_AUG, SEED_KMEANS` all set to 42.
- nb03 uses `SPLIT_SEEDS = list(range(42, 42 + N_SPLIT_REPS))` with `N_SPLIT_REPS=10`. For the 20-seed rerun, change `N_SPLIT_REPS` to 20 in cell 6 only; no other site hardcodes 10.
- See L1 for hardcoded `default_rng(42)` call sites.

### 6. Metric computation consistency
- `src/evaluation.py:compute_metrics` is the canonical definition (RMSE, MAE, R2, bias, n). Used by nb05, nb07, nb09 Phase 9R.EXT.
- nb03 Phase 3R.4 computes RMSE inline via `sklearn.metrics.mean_squared_error` rather than `compute_metrics`; numerically identical.
- nb04 three-way uses its own `rmse_ci` function with paired bootstrap (B=2000). Different formula target (paired CI, not point RMSE).
- No functions are using biased vs unbiased variance, no `squared=False` drift, nothing uses `r2_score(multioutput=...)` inconsistently.

### 7. Figure inventory vs references
- See H3 and L3. Canonical numbered figures: fig01-fig09, fig11-fig14 are present; fig10 is the hole. V6 bonus figures (fig_nb02_clusters, fig_nb04_three_way, fig_nb08_twopx_1to1, fig_nb11_model_family_ceiling) are present but not promoted into the roster.
- Figures produced for diagnostic purposes (fig_nb03c_*, fig_eda_*, fig_nb04_failure_analysis, fig_nb04b_arcpl_*, fig_nb06_correlation_check, fig_nb06_proxy_check) are expected to live outside the manuscript roster and appear in nb09's "Additional figures" gallery.

### 8. CSV schema consistency
- See M3 for the predicted/observed column-name drift.
- Schemas inside each notebook family are internally consistent (every `nb03_*` file uses `T_pred_{MODEL}, P_pred_{MODEL}`; every `nb08_*` file uses `T_pred, P_pred, T_lo, T_hi, P_lo, P_hi, T_CV, P_CV`).
- The `Experiment` column is present in both `nb04b_arcpl_predictions.csv` and the LEPR sheets used by the three-way benchmark; the merge in [scripts/build_nb04_three_way.py:159-162](scripts/build_nb04_three_way.py#L159-L162) works.

### 9. Markdown / code ratio
- See M4. Overall 48 md / 156 code = 0.31 ratio. Phase headers in nb01 and nbF would bring several notebooks to ≥0.30 cheaply.

### 10. Hyperparameter consistency
- nb03 persists frozen params in `results/nb03_multi_seed_results.csv` (column `best_params`) and writes a sidecar `nb03c_frozen_params.json` in the project root (not in `results/`). The root-level location is unusual; most other artifacts live in `results/`. nb05 reconstructs `best_params_by_model` from the CSV, not from the JSON, so the JSON is effectively write-only. Either move it into `results/` or remove it.
- A second root-level file `nb03c_training.log` (training progress log from Phase 3R.4) also sits at project root rather than in `logs/`. Same recommendation.
- Hyperparameter grids in [notebooks/nb03_baseline_models.ipynb](notebooks/nb03_baseline_models.ipynb) cell 6 are adequate for the 20-seed rerun (multi-value options for n_estimators, depths, regularization). HalvingRandomSearchCV runs on seed 42 only; all 10 (or future 20) seeds use the frozen params. This is the design — documented in the Phase 3R.3b preamble — and is correct.

### 11. Bias correction application
- nb07 is the only live notebook that applies `apply_piecewise_correction`. The output `nb07_bias_correction_null_result.csv` confirms the delta-RMSE 95% CI contains zero (null result). The corrected predictions are persisted to `nb07_test_predictions.csv` solely for fig11 residual plotting.
- nb04, nb04b, nb08, nb09 Phase 9R.EXT do NOT apply the bias correction. Consistent with the null result — correctly treating it as a sensitivity check, not a production step. No change needed.

### 12. Thermobar external model calls
- [src/external_models.py](src/external_models.py) equation strings:
  - Jorgenson cpx_only: `T_Jorgenson2022_Cpx_only`, `P_Jorgenson2022_Cpx_only`
  - Jorgenson cpx_liq: `T_Jorgenson2022_Cpx_Liq`, `P_Jorgenson2022_Cpx_Liq`
  - Wang 2021: `T_Wang2021_eq2`, `P_Wang2021_eq1`
  - Putirka 2008 cpx-liq: `T_Put2008_eq33`, `P_Put2008_eq30`
- [scripts/build_nb08_v6.py](scripts/build_nb08_v6.py) uses Putirka 2008 two-pyroxene: `T_Put2008_eq36`, `P_Put2008_eq39` via `calculate_cpx_opx_temp/press`.
- All strings match Thermobar 1.0.70 public equation identifiers (verified against user memory note on Thermobar version). No typos.
- `m['Fe3Fet_Liq'] = 0.0` is set in the three-way load block ([scripts/build_nb04_three_way.py:97-98](scripts/build_nb04_three_way.py#L97-L98)) to avoid Thermobar skipping rows; consistent with the project-memory convention.

### 13. OPERATOR DECISION markers
- Live notebooks with OPERATOR DECISION blocks: nb04, nb04b, nb09. Each prints a decision prompt and does not auto-select. If the 20-seed rerun is non-interactive, these three cells will print their prompt and move on without altering any CSV. No blockers.
- Archived OPERATOR DECISION sites (nb08_v5, nb10_v5, nb10b) are in `archive/` and will not execute.

### 14. Logical inconsistencies specific to this project
- C1 and C2 above.
- H4: on-disk splits written after use.
- M1: fragile K→C heuristic.
- M6: Gridded-PT is target-stratified, not spatial.
- The three-way benchmark "coverage" panel (d) in `fig_nb04_three_way.*` implicitly assumes every ArcPL row has cpx; the `has_cpx` filter happens before the `preds` loop ([scripts/build_nb04_three_way.py:106-108](scripts/build_nb04_three_way.py#L106-L108)), so denominator `len(m)` already excludes cpx-less rows. Classical Putirka's coverage can still be less than 100% due to Thermobar row-level equilibrium filters, so the panel label "Fraction of ArcPL each method can predict" should read "Fraction of cpx-present ArcPL rows each method can predict" to avoid overclaiming.

---

## Files reviewed

- `config.py` (full)
- `src/data.py`, `src/evaluation.py`, `src/external_models.py`, `src/features.py` (full or signatures)
- `notebooks/nb01_data_cleaning.ipynb` through `notebooks/nbF_figures.ipynb` (11 live; structural + targeted cell reads)
- `scripts/build_nb04_three_way.py`, `scripts/build_nb08_v6.py`, `scripts/merge_nb10_into_nb09.py`, `scripts/overhaul_nb02.py`, `scripts/smoke_test_imports.py`, `scripts/verify_v6.py` (full)
- `results/nb03_winning_configurations.json`, `nb03_multi_seed_summary.csv` (content audit)
- `results/*.csv` header inventory (56 files)
- `figures/` directory listing (36 PNG/PDF files)
- `models/` directory listing (29 joblibs + 12 vendor files under `external/`)
- `logs/` directory listing (4 FAILURE + 7 pipeline_resume/run logs + cleaning + environment)
- `archive/` listing (5 v5 notebooks + pipeline_v1_legacy + v5_reports)
- Root-level files: `nb03c_frozen_params.json`, `nb03c_training.log`, `PROJECT_OVERVIEW.md`, `README.md`, `run_all.py`, `run_from.py`

## What I did not check

- Cell-by-cell code-correctness review of every line in every notebook (audit focused on cross-file consistency; internal logic bugs inside a single cell are out of scope here).
- Runtime execution of the pipeline; all findings are static (read-only).
- Numerical correctness of persisted models (no joblib.load and prediction diff against saved CSVs).
- Vendor ONNX models under `models/external/` (trusted bundle, treated as a black box with known interface).
- Jupyter cell output metadata (execution counts, stderr output).
- `data/raw/`, `data/external/`, `data/natural/` input file integrity beyond confirming presence.
- `run_all.py` and `run_from.py` orchestrator scripts.
- CLAUDE.md, README.md, PROJECT_OVERVIEW.md narrative accuracy.
- Test coverage (project has no `tests/` directory, so no unit tests to audit).
- Manuscript text — only the pipeline-produced artifacts it will reference.
