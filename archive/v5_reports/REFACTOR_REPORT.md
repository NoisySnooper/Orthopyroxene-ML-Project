# Refactor report: opx ML thermobarometer

**Audience.** An outside AI agent (or reviewer) auditing the work performed
on this repository. This document states what was done, what was not done,
and how to verify each claim independently.

**Repo.** opx ML thermobarometer. Manuscript target: JGR ML & Computation.
Co-author: Dr. Kanani K.M. Lee.

**Scope of this report.** Execution of `FIX_PLAN.md` v5.0, a 9-phase
structural refactor. The refactor changed code and notebook structure only;
it did not rerun the pipeline against real data.

## Ground rules followed

- No git operations. The user is not using git for this project and
  explicitly asked that none be performed.
- Plain language, blunt tone. No em dashes in prose.
- Windows host, Python 3.13, project-local `.venv`, Thermobar 1.0.70.

## Starting state (before the refactor)

A pre-refactor snapshot is preserved in [AUDIT_REPORT.md](AUDIT_REPORT.md)
(generated 2026-04-13). Key problems that refactor was meant to fix:

1. Feature engineering and prediction logic duplicated across several
   notebooks. `make_pwlr_features`, `make_alr_features`, `predict_median`,
   `predict_iqr`, and `hasterok_chapman_geotherm` all existed in more than
   one place.
2. Winning feature method (`raw` / `alr` / `pwlr`) was hardcoded in
   downstream notebooks, so swapping the winner required editing many files.
3. Paths, seeds, and scientific constants were scattered. Some notebooks
   imported from a root-level `figure_style.py`, others from ad hoc helpers.
4. NB02 overwrote its own input parquet with the cluster-augmented version,
   destroying the clean baseline.
5. NB09 both read and rewrote `table5_shap_importance.csv`, so the table's
   ownership was ambiguous.
6. `nb03b_n_aug_sensitivity`, `nb03c_phase3_benchmark`, and
   `nb06b_shap_robustness` were split from their parent notebooks and
   drifted; they had divergent code paths and different feature schemas.
7. NBF figure 12 drew a hand-drawn geotherm instead of calling the real
   H&C 2011 routine.

## What was done, phase by phase

### Phase 1: `src/` package creation

Moved the canonical helpers out of notebooks and into a package. Line counts
(verified via `wc -l`):

| Module                | Lines | Responsibility                                     |
|-----------------------|-------|----------------------------------------------------|
| `src/__init__.py`     | 5     | Package marker                                     |
| `src/features.py`     | 183   | `build_feature_matrix` (raw/alr/pwlr), EPMA noise aug |
| `src/models.py`       | 150   | `BASE_MODELS`, `predict_median`, `predict_iqr`    |
| `src/data.py`         | 100   | Loaders, `canonical_model_filename`, `load_canonical_model` |
| `src/evaluation.py`   | 118   | `compute_metrics`, `loso_splits`, `cluster_kfold_splits` |
| `src/geotherm.py`     | 55    | Hasterok & Chapman 2011 layered geotherm           |
| `src/io_utils.py`     | 44    | `save_figure` (PNG+PDF 300 dpi), `save_table`, `with_progress` |
| `src/plot_style.py`   | 135   | Tol palette, `apply_style`, `panel_label`, `load_winning_config` |

### Phase 2: `config.py` as single source of truth

Every path, seed, and scientific constant lives here. Full file is 69 lines
(see [config.py](config.py)). New canonical filenames appended:

```
OPX_CORE_CLUSTERED_FILE   = 'opx_clean_core_with_clusters.parquet'
WINNING_CONFIG_FILE       = 'nb03_winning_configurations.json'
MULTI_SEED_RESULTS_FILE   = 'nb03_multi_seed_results.csv'
CANONICAL_PREDICTIONS_FILE= 'nb03_canonical_test_predictions.npz'
FEATURE_METHODS           = ('raw', 'alr', 'pwlr')
N_AUG_DEFAULT             = 1
EPMA_NOISE_REL_MAJOR      = 0.03
EPMA_NOISE_REL_MINOR      = 0.08
QRF_QUANTILES             = (0.16, 0.5, 0.84)
GEOTHERM_Q_S_CRATONIC     = 40.0
GEOTHERM_Q_S_AVERAGE      = 60.0
GEOTHERM_Q_S_HOT          = 80.0
```

### Phase 3: NB01 cleaning

No structural change required. Output schema and canonical filename checked.

### Phase 4: NB02 and NB03 canonical outputs

- NB02 now writes `opx_clean_core_with_clusters.parquet` instead of
  overwriting `opx_clean_core.parquet`.
- NB02 EDA figures renamed `fig01_pt_distribution.png` ->
  `fig_eda_pt_distribution.png`, `fig02_pca_biplot.png` ->
  `fig_eda_pca_biplot.png` to free the `fig01`/`fig02` slots for NBF.
- NB03 reads the clustered parquet.
- `nb03b_n_aug_sensitivity` merged into NB03 as an appendix; `nb03b` and
  `nb03c` deleted.
- All NB03 call sites migrated from `get_features_for_method(` to
  `build_feature_matrix(` (the canonical name in `src.features`).

### Phase 5: validation strategies (NB05)

NB05 was missing its execution cells entirely; only imports and a
verification stub remained. Rebuilt with a `STRATEGY_SPLITTERS` dict:

```python
STRATEGY_SPLITTERS = {
    'LOSO':          ('Citation',         loso_splits),
    'Cluster-KFold': ('chemical_cluster', cluster_kfold_splits),
    'Gridded-PT':    ('pt_grid',          cluster_kfold_splits),
}
```

Writes `nb05_loso_pooled.csv` and `nb05_per_fold_rmse.csv`. Wrapped with
tqdm.

### Phase 6: NB06 SHAP robustness

Merged 14 cells from `nb06b_shap_robustness` into NB06 as an appendix:
ablation of `liq_SiO2` + `liq_MgO`, proxy-risk scatter, feature correlation
heatmap, y-randomization, dummy regressor, perfect-signal injection. Each
check is rewired to use `build_feature_matrix(df, WIN_FEAT, use_liq=True)`
and `clone(model_P)` instead of the old ad hoc `df.select_dtypes(...)`
pattern.

### Phase 7: downstream fixups

- NB07 QRF bias correction: confirmed canonical model loader usage.
- NB08 natural samples: removed a stale local copy of
  `hasterok_chapman_geotherm`; now imports from `src.geotherm`.
- NB09 manuscript compilation:
  - Cell 3: `n_seeds=('seed', 'nunique')` -> `n_seeds=('split_seed',
    'nunique')`.
  - Cell 6: stopped rewriting `table5_shap_importance.csv`. NB06 is the
    sole owner.
  - Markdown header cell: prose reference to `figure_style.load_winning_config`
    updated to `src.plot_style.load_winning_config`.
- NB10 extended analyses:
  - Cell 5: `'H2O_liq'` -> `'H2O_Liq'` (matches canonical schema used in
    NB01/NB04/NB04b).
  - Cell 7: tqdm wrapper on the Monte Carlo loop.
- NBF figures:
  - Cell 0 prose: `figure_style.load_winning_config` ->
    `src.plot_style.load_winning_config`.
  - Cell 1: `from src.geotherm import hasterok_chapman_geotherm`.
  - Fig 12 rewritten to plot real H&C geotherms at `q_s` =
    {40, 60, 80} mW/m^2 with Tol gray and line-style distinction per curve.
  - Manual `_iqr()` helper replaced with `predict_iqr` from `src.models`.

### Phase 8: documentation

New files:
- [README.md](README.md): install, run order, design decisions, data
  sources.
- [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md): Mermaid data-flow and
  module-topology diagrams; invariants and robustness tables.

### Phase 9: audit

See next section.

## Verification

Run the rerunnable audit:

```
.venv\Scripts\python.exe scripts\audit_structure.py
```

Result at time of this report: **61 pass / 0 fail**. The script is
[scripts/audit_structure.py](scripts/audit_structure.py). Each check is
short, read-only, and prints a single line to stdout.

Smoke-check notebook imports:

```
.venv\Scripts\python.exe scripts\smoke_test_imports.py
```

Result: `nb01-04b` and `nbF` pass. `nb05-nb10` fail on missing upstream
outputs (`opx_clean_core_with_clusters.parquet`,
`nb03_winning_configurations.json`, etc.), which is the expected behavior
when NB01-NB03 have not yet been run.

## Structural invariants now enforced

Each of these is tested by `audit_structure.py`:

1. Every module in `src/` exists with a non-trivial implementation.
2. All twelve expected notebooks are present. `nb03b`, `nb03c`, and `nb06b`
   are deleted.
3. No notebook redefines `make_pwlr_features`, `make_alr_features`,
   `predict_median`, `predict_iqr`, or `hasterok_chapman_geotherm`. Each
   lives in exactly one place in `src/`.
4. No notebook has `from figure_style import ...` or `import figure_style`.
5. NB02 writes `opx_clean_core_with_clusters.parquet` and does not
   overwrite `opx_clean_core.parquet`.
6. NB09 only reads `table5_shap_importance.csv` and does not rewrite it.
7. NB06 contains the robustness appendix (ablation, y-randomization,
   perfect-signal injection all present).
8. NB03 contains the `n_aug` sensitivity appendix.
9. NBF imports and uses `hasterok_chapman_geotherm`.
10. `README.md` and `PROJECT_OVERVIEW.md` are present.
11. `config.py` defines `FEATURE_METHODS`, `N_AUG_DEFAULT`,
    `QRF_QUANTILES`, `GEOTHERM_Q_S_AVERAGE`, `OPX_CORE_CLUSTERED_FILE`,
    `WINNING_CONFIG_FILE`.

## What was NOT done

1. **End-to-end notebook execution on real data.** Running NB01 through NBF
   in order is estimated at several hours (NB03's tune-once halving search
   alone is 30-60 min across four models x three feature methods x ten
   seeds). The refactor was scoped to structure and verified via
   imports-only smoke test. `data/processed/`, `data/splits/`, `models/`,
   and `results/` are currently empty. Outputs listed in an earlier
   `AUDIT_REPORT.md` (from 2026-04-13) were deleted at some point before
   this refactor began and have not been regenerated.
2. **Papermill-driven CI.** The refactor wires tqdm into the long loops but
   does not add a papermill run script. Easy follow-up.
3. **Manuscript tables and figures.** NB09 and NBF cannot produce their
   outputs until NB01-NB10 are executed. The refactor only fixes their code
   paths.

## Known residual risks for the reviewer to double-check

- The pre-existing `notebooks/figure_style.py` file is still on disk. No
  notebook imports it (verified by the audit). Leaving it in place is
  harmless; deleting it would need user confirmation.
- NB09 cell 6 is now a read-only consumer of `table5_shap_importance.csv`.
  If NB06 fails to produce that table, NB09 will raise. This is intentional:
  downstream notebooks should break loudly on missing upstream outputs, not
  paper over them.
- Three `_test_indices_opx*.npy` files are referenced by downstream figures.
  They are produced by NB03 and do not yet exist on disk.
- `Cluster-KFold` and `Gridded-PT` strategies reuse the same
  `cluster_kfold_splits` function with different group columns. If the
  reviewer expects a distinct gridded-bin implementation, flag for me.

## File manifest

**Created:**
- `src/__init__.py`, `src/features.py`, `src/models.py`, `src/data.py`,
  `src/evaluation.py`, `src/geotherm.py`, `src/io_utils.py`,
  `src/plot_style.py`
- `README.md`, `PROJECT_OVERVIEW.md`, `REFACTOR_REPORT.md` (this file)
- `scripts/audit_structure.py`, `scripts/smoke_test_imports.py`,
  `scripts/merge_nb03.py`, `scripts/merge_nb06.py`,
  `scripts/rewrite_notebook_imports.py`

**Deleted:**
- `notebooks/nb03b_n_aug_sensitivity.ipynb`
- `notebooks/nb03c_phase3_benchmark.ipynb`
- `notebooks/nb06b_shap_robustness.ipynb`

**Edited (structurally):**
- `config.py` (constants block appended)
- `requirements.txt` (added `tqdm`, `papermill`, `seaborn`, `nbformat`)
- `notebooks/nb02_eda_pca.ipynb`
- `notebooks/nb03_baseline_models.ipynb`
- `notebooks/nb05_loso_validation.ipynb`
- `notebooks/nb06_shap_analysis.ipynb`
- `notebooks/nb07_bias_correction.ipynb`
- `notebooks/nb08_natural_samples.ipynb`
- `notebooks/nb09_manuscript_compilation.ipynb`
- `notebooks/nb10_extended_analyses.ipynb`
- `notebooks/nbF_figures.ipynb`

## How to continue from here

The next action the user has two options on:

1. **Full Phase 7 end-to-end run.** Invoke papermill over the ordered
   notebook list. Multi-hour; best run overnight or as a background job.
2. **Targeted spot-check.** Execute a single notebook under `.venv`
   Jupyter and confirm real-data output matches the canonical schema.
   Good candidate: NB01 -> NB02 -> NB03 (the critical path). All other
   downstream notebooks depend on NB03 outputs.

Either path is safe; the refactor did not introduce state the reviewer
cannot undo by reverting a handful of files.
