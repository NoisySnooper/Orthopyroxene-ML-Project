# Pyroxene ML thermobarometer

Machine-learning thermobarometers for orthopyroxene and clinopyroxene
compositions. Predicts pressure (kbar) and temperature (C) of equilibrium from
mineral oxide chemistry, with or without a paired liquid, and extends to
two-pyroxene and universal (any-phase-combination) configurations.

Two manuscripts in preparation:
1. **Opx paper (`manuscripts/opx_2026/`)** — first ML thermobarometer specific
   to orthopyroxene. Target: *JGR Machine Learning and Computation*, 2026.
2. **Cpx paper (`manuscripts/cpx_2026/`)** — framework paper covering
   clinopyroxene, two-pyroxene, and universal masking model, with
   head-to-head benchmarks against Agreda-Lopez 2024, Jorgenson 2022,
   Wang 2021, Petrelli 2020. Target: 2027.

**Status:** v9 complete (2026-04-16). v10+v11 planning complete (2026-04-16).
Phase G.7 bias correction + TabPFN v2 baseline shipped 2026-04-19.
Phase 1.5 cleanup (`BASE_ORDER` rename, `V10_BASE_ORDER` alias removed,
most `v10_*` paths renamed, v9 docs archived) committed 2026-04-19.
TabPFN Option B promoted TabPFN to 9th `BASE_ORDER` family (8 tuned +
TabPFN pretrained), 20-seed protocol across all families, 2026-04-19. See
[`docs/master_plan.md`](docs/master_plan.md) for the active plan and
[`PROJECT_LAYOUT.md`](PROJECT_LAYOUT.md) for the directory tree.

---

## v9 headline numbers (verified 2026-04-16)

### Test set, 20-seed means (opx only, cpx/twopx not yet trained)

| Track | Target | Best model | RMSE |
|---|---|---|---|
| opx_liq | T_C | XGB + alr features | 77.77 C |
| opx_liq | P_kbar | XGB + raw features | 4.81 kbar |
| opx_only | T_C | RF + pwlr | 139.5 C |
| opx_only | P_kbar | GB + pwlr | 10.65 kbar |

### ArcPL external validation, n=197 full scope

| Family | T bias | T RMSE | P bias | P RMSE |
|---|---|---|---|---|
| Forest (RF) | +48.5 C | 71.6 | +1.24 kbar | 2.79 |
| Boosted (XGB, primary) | +20.9 C | 63.7 | -0.09 kbar | 2.70 |

### ArcPL Kd-equilibrated head-to-head, n=96

| Method | T RMSE | P RMSE |
|---|---|---|
| Ours opx-liq stacked | 53.6 | 3.30 |
| Ours opx-liq boosted | 56.2 | 3.12 |
| Putirka 2008 opx-liq | 54.9 | 4.11 |
| Agreda-Lopez cpx-liq | 45.0 | 1.94 |

**Honest framing.** We match Putirka on T, beat Putirka on P by ~20%, lose
to cpx-specific models (Agreda-Lopez, Jorgenson) which benefit from cpx's
intrinsically higher P-T sensitivity. Opx paper claim is "first opx-specific
ML thermobarometer with formal OOD uncertainty quantification", not a claim
to beat cpx models. Cpx paper will claim parity or better vs cpx ML models
per the three-gate criterion (see `docs/master_plan.md` Section 8).

---

## v10 scope (two-paper, unified implementation)

v10 is not a single-paper rebuild. It is the foundation work for both papers
executed in one repo. Concretely:

| Track | Pipeline | Models | Source notebook |
|---|---|---|---|
| opx_only | opx composition only | 8 | `nb03_opx_baseline_models` |
| opx_liq | opx + liquid | 8 | `nb03_opx_baseline_models` |
| cpx_only | cpx composition only | 8 | `nb03_cpx_baseline_models` |
| cpx_liq | cpx + liquid | 8 | `nb03_cpx_baseline_models` |
| twopx | opx + cpx pair | 8 | `nb03_twopx_baseline_models` |
| universal | any phase combination (masking) | 8 | `nb03_universal_exploration` (isolated) |

**Model roster (8 total, all pipelines):** RandomForest, ExtraTrees, XGBoost,
GradientBoosting, CatBoost, LightGBM, ElasticNet, MLPRegressor. Plus 4
ensemble methods compared (Ridge stacking, two-level, greedy Caruana 2004,
AutoGluon) per `docs/ensemble_methods_plan.md`.

**Universal model is intentionally isolated.** Separate notebook, separate
diagnostics directory (`results/universal/`, `figures/universal/`). Never
combined with opx/cpx/twopx primary results. Side-project status. Potential
third paper material.

---

## Pipeline layout

See [PROJECT_LAYOUT.md](PROJECT_LAYOUT.md) for the full tree with
per-entry one-liners. In brief: `src/` holds library code (features,
models, evaluation, bias correction, stacking, Thermobar adapter);
`notebooks/` holds the 4 track-specific `nb03_*_baseline_models.ipynb`
variants plus nb01/nb02/nb04-nb09/nbF; `scripts/` holds Phase G/H
drivers and TabPFN wiring (`opx_tb_nb03_*` scripts); `results/` +
`figures/` + `tables/` hold artifacts; `manuscripts/opx_2026/` holds
paper deliverables; `docs/master_plan.md` is the active plan.

---

## Running the pipeline

```
python -m venv .venv
.venv\Scripts\activate                     # Windows PowerShell
pip install -r requirements.txt
```

Then notebooks in order (or run `python run_all.py`):

1. `nb01_data_cleaning` — ExPetDB + LEPR cleaning for all tracks
2. `nb02_eda_pca` — EDA + per-track chemical clusters
3. `nb03_opx_baseline_models` — opx_only + opx_liq, 8 models, T01-T12
4. `nb03_cpx_baseline_models` — cpx_only + cpx_liq, 8 models
5. `nb03_twopx_baseline_models` — two-pyroxene, 8 models
6. `nb03_universal_exploration` — universal masking (isolated side project)
7. `nb03_tabpfn_baseline` — TabPFN v2 baseline (20 seeds, CPU)
8. `nb04_putirka_benchmark` — ArcPL + Thermobar benchmarks + TabPFN head-to-head
9. `nb05_loso_validation` — LOSO + Cluster + TargetBin + LeaveOneRegion
10. `nb06_shap_analysis` — tree-SHAP + linear-SHAP on stack
11. `nb07_bias_correction` — composition-conditional T (Form A/B)
12. `nb07b_arcpl_bias_probe` — ArcPL-specific bias probe
13. `nb08_natural_twopx` — GEOROC world maps + cross-mineral convergence
14. `nb09_manuscript_compilation` — per-paper table subsets (S8_*, table_4_*, S8_12)
15. `nbF_figures` — canonical figure regen (fig24–fig35)

NB01-NB03 mandatory for everything downstream. NB04-NBF parallelizable
once NB03 winners are frozen.

---

## Key design decisions (locked for v10)

- **8 models in every pipeline.** RF, ERT, XGB, GB (v9 four) + CatBoost,
  LightGBM, ElasticNet, MLPRegressor (v10 additions). Per test T11/T12 if
  CatBoost or MLP beats the v9 four, it ships; otherwise ablated with data.
- **Test-first NB03.** 12 pre-registered tests (T01-T12, plus T13-T14 for
  universal). Each runs per-pipeline; ship/ablate decision logged to
  `results/nb03_test_log.csv`. See `docs/preregistration/nb03_test_protocol.md`.
- **4 ensemble methods compared.** Ridge, two-level Ridge, greedy Caruana,
  AutoGluon. Winner ships; others reported as ablation. See
  `docs/ensemble_methods_plan.md`.
- **Stacking propagates everywhere.** Every downstream benchmark figure
  (NB04-NB10) includes stacked alongside base models. Enforced per
  `docs/stacking_propagation_audit.md`.
- **Markdown discipline.** Every code cell gets a rigorous markdown block:
  Input / Output / Method / Connection / Why. Template in
  `docs/markdown_template.md`.
- **Figure rigor.** Every figure passes the checklist in
  `docs/figure_audit.md`: all methods plotted, on-figure metrics
  (R^2, RMSE, slope, intercept), Okabe-Ito palette, label-collision check,
  JGR-MLC dimension compliance, PDF + PNG + TXT caption.
- **World maps.** Two panels per mineral (opx, cpx, twopx): tectonic setting
  color and predicted T color. Robinson projection (static matplotlib +
  cartopy). Interactive folium version as supplementary.
- **Beat-Agreda/Jorgenson gate is (c).** Must win on ArcPL AND LOSO AND
  natural-sample agreement, each with 95% bootstrap CI excluding theirs. Else
  framed as "matches" or "competitive".
- **N_AUG=1 carried from v9.** EPMA augmentation hurt 6/8 test configs in v9
  per test T04. Re-tested per pipeline in v10; ship/ablate per result.
- **Canonical splits preserved.** `config.SEED_SPLIT=42`. Test indices in
  `data/splits/test_indices_{track}.npy`.
- **TabPFN v2 family (9th in `BASE_ORDER`).** Hollmann et al. 2025
  (*Nature* 637, doi:10.1038/s41586-024-08328-6) pretrained foundation
  model promoted to the 9th first-class model family post-Phase-1.5
  (TabPFN Option B, 2026-04-19). `BASE_ORDER` now 9 entries: 8
  Optuna-tuned families + TabPFN last. `TUNED_BASES = BASE_ORDER[:-1]`
  for Optuna-pipeline consumers. 20 seeds (42-61, matching other
  families), CPU inference, raw oxide features only, no Optuna, no
  stacking, no SHAP. **Excluded from bias correction** (no OOF residuals
  available: in-context foundation model with a single forward pass).
  `STACKING_BASE_ORDER` stays 4 (RF/ERT/XGB/GB) — TabPFN deliberately
  not a stacking base. Outputs: `results/tabpfn_*.csv`,
  `figures/fig35_tabpfn_vs_opx_tb`, `tables/S8_12_tabpfn_benchmark.{csv,md,tex}`,
  `manuscripts/opx_2026/text/tabpfn_paragraph.md`. Head-to-head across 8
  cells: opx_tb wins 6, TabPFN wins 1 (cpx_liq T_C), 1 competitive. See
  `docs/nb03_tabpfn_plan.md`.

---

## Data sources

### Training (experimental, known T and P)
- **ExPetDB** (primary) — opx, cpx, twopx experiments with paired liquid.
  Putirka 2008 Kd filter (0.23-0.35), Wo < 5 mol% cut for opx. n=600 opx-liq
  pairs in v9. Expected cpx-liq ~800-1500, twopx ~500-800.
- **LEPR Wet Stitched April 2023** — secondary, cpx overlap with ExPetDB.
  External papers (Jorgenson 2022, Agreda-Lopez 2024) also trained on LEPR,
  so head-to-head comparisons use same data.

### External validation (experimental, known T and P)
- **ArcPL** — n=197 full scope, n=96 Kd-equilibrated for head-to-head.

### External cpx ML models
- **Agreda-Lopez et al. 2024** — cpx-liq thermobarometer, ET Regressor.
- **Jorgenson et al. 2022** — cpx-only thermobarometer.
- **Wang et al. 2021** — cpx-liq.
- **Petrelli et al. 2020** — cpx-liq baseline ML.

See `docs/external_models_audit.md` for publicly-released training data
status per model. Where training data is public, we re-run their models on
our ArcPL subset for head-to-head. Where not, we use their published model
artifact.

### Natural samples (inference only, no known T/P)
- **GEOROC 2024-12 SGFTFN opx** —
  `data/natural/2024-12-SGFTFN_ORTHOPYROXENES.csv`. 78,532 opx analyses
  with lat/lon and tectonic setting metadata. Driving the opx world map.
- **GEOROC 2024-12 cpx** — TO BE PULLED in Phase H. Expected similar size.
- **Curated localities (experimental T/P only, literature)** — TBD in Phase H
  per `docs/natural_worldwide_plan.md`.

---

## Documentation map

### v10 planning documents (read in this order)

| Doc | Purpose |
|---|---|
| [`docs/master_plan.md`](docs/master_plan.md) | **Start here.** Unified v10+v11 plan. |
| [`docs/markdown_template.md`](docs/markdown_template.md) | Per-cell markdown template (rigorous format) |
| [`docs/cleanup_manifest.md`](docs/cleanup_manifest.md) | Fresh-start cleanup file list |
| [`docs/notebooks_compatibility_audit.md`](docs/notebooks_compatibility_audit.md) | Downstream NB compatibility matrix |
| [`docs/preregistration/nb03_test_protocol.md`](docs/preregistration/nb03_test_protocol.md) | T01-T14 test-first protocol, per-pipeline |
| [`docs/cpx_pipeline_plan.md`](docs/cpx_pipeline_plan.md) | Cpx pipeline design |
| [`docs/twopx_pipeline_plan.md`](docs/twopx_pipeline_plan.md) | Two-pyroxene pipeline design |
| [`docs/universal_model_exploration.md`](docs/universal_model_exploration.md) | Universal masking model (isolated) |
| [`docs/ensemble_methods_plan.md`](docs/ensemble_methods_plan.md) | 4-method ensemble comparison |
| [`docs/stacking_propagation_audit.md`](docs/stacking_propagation_audit.md) | Where stacking must appear in NB04-NB10 |
| [`docs/figure_audit.md`](docs/figure_audit.md) | ~308 figure per-spec checklist |
| [`docs/natural_worldwide_plan.md`](docs/natural_worldwide_plan.md) | GEOROC re-integration + world map |
| [`docs/external_models_audit.md`](docs/external_models_audit.md) | External model training data verification |

### Background methodology (with v10 re-evaluation blocks)

| Doc | Purpose |
|---|---|
| [`docs/stacking_strategy.md`](docs/stacking_strategy.md) | Ridge meta design + v9 outcome + v10 test T09 hook |
| [`docs/resampling_strategy.md`](docs/resampling_strategy.md) | Tempered P-T resampling + v9 ablation + v10 re-test |
| [`docs/optuna_strategy.md`](docs/optuna_strategy.md) | TPE search + v9 outcome + v10 reuse decision |
| [`docs/putirka_kd_filter_lookup.md`](docs/putirka_kd_filter_lookup.md) | Thermobar Kd API behavior |
| [`docs/codebase_consistency_audit_optionB.md`](docs/codebase_consistency_audit_optionB.md) | ArcPL scope reconciliation |

### v9 archive (historical)

| Doc | Purpose |
|---|---|
| [`docs/archive_superseded/v9_outcomes.md`](docs/archive_superseded/v9_outcomes.md) | v9 execution log |
| [`docs/archive_superseded/v9_inventory_report.md`](docs/archive_superseded/v9_inventory_report.md) | v9 artifact inventory |
| [`docs/archive_superseded/v9_archive_plan.md`](docs/archive_superseded/v9_archive_plan.md) | v9 artifact archiving plan |
| [`docs/archive_superseded/v9_deletion_plan.md`](docs/archive_superseded/v9_deletion_plan.md) | v9 cleanup plan (superseded by v10_cleanup_manifest) |

---

## License and citation

This repository supports two manuscripts in preparation. Cite via the
forthcoming DOIs once published. Source code is MIT licensed. Training data
from ExPetDB, LEPR, and GEOROC is subject to each source's respective
license; see `data/README.md` (to be added in Phase A) for redistribution
terms.
