# v10 master plan

**Status:** canonical, pending user approval for Phase A execution
**Author:** NQTa (drafted with Claude 2026-04-16)
**Scope:** combined v10 (opx paper submission) + v11 (cpx paper research) in one unified plan, one repo, two papers

---

## 0. Document index

This doc is the master orchestrator. Sub-docs referenced throughout:

| # | Doc | Purpose |
|---|---|---|
| 1 | `v10_markdown_template.md` | Per-cell markdown structure (rigor) |
| 2 | `v10_master_plan.md` | This doc |
| 3 | `v10_cleanup_manifest.md` | Fresh-start file list |
| 4 | `v10_notebooks_compatibility_audit.md` | Pre-Phase-C gate |
| 5 | `v10_nb03_test_protocol.md` | Test-first rebuild protocol |
| 6 | `v10_cpx_pipeline_plan.md` | Cpx parallel pipeline |
| 7 | `v10_twopx_pipeline_plan.md` | Two-pyroxene pipeline |
| 8 | `v10_universal_model_exploration.md` | Universal masking model (side project) |
| 9 | `v10_ensemble_methods_plan.md` | Ridge / two-level / greedy / AutoGluon comparison |
| 10 | `v10_stacking_propagation_audit.md` | Where stacking must appear |
| 11 | `v10_figure_audit.md` | Per-figure spec, all 120+ figures |
| 12 | `v10_natural_worldwide_plan.md` | GEOROC + world map + locality cross-check |
| 13 | `v10_external_models_audit.md` | Verify external training data sources |

Updates to existing docs: `README.md`, `PROJECT_OVERVIEW.md`, `stacking_strategy.md`, `resampling_strategy.md`, `optuna_strategy.md`.

---

## 1. v9 findings that set the baseline

Verified 2026-04-16 against live results files.

### 1.1 What shipped in v9

- Optuna TPE replaces HalvingRandomSearchCV (48 studies, 2.12 h wall time, 0 failures)
- Ridge stacking implemented per `stacking_strategy.md` — on test: all 4 targets have alpha pegged at endpoint (100), stacked worse than best base by 0.03 to 6.5 C/kbar. On ArcPL: stacked T wins by 2.65 C over boosted.
- P-T tempered resampling implemented, 8 resampled joblibs produced — 6/8 worse on test, 8/8 worse on ArcPL. Correctly ablated.
- N_AUG=1 confirmed via sensitivity test.
- Option B Kd-equilibrated scope preserved (Putirka 2008 head-to-head n=96).

### 1.2 v9 headline numbers (verified)

**Test set 20-seed means (nb03_multi_seed_summary.csv):**

| Track | Target | Best model | RMSE | R^2 |
|---|---|---|---|---|
| opx_liq | T_C | XGB+alr | 77.77 (±9.5) | 0.76 |
| opx_liq | P_kbar | XGB+raw | 4.81 (±0.8) | 0.74 |
| opx_only | T_C | RF+pwlr | 139.5 (±18.4) | 0.50 |
| opx_only | P_kbar | GB+pwlr | 10.65 (±2.8) | 0.28 |

**ArcPL n=197 (recomputed from predictions CSVs):**

| Family | T bias | T RMSE | P bias | P RMSE |
|---|---|---|---|---|
| Forest (RF) | +48.51 C | 71.60 | +1.24 kbar | 2.79 |
| Boosted (XGB) | +20.95 C | 63.68 | -0.09 kbar | 2.70 |

**ArcPL n=96 Kd-equilibrated head-to-head:**

| Method | T RMSE | P RMSE |
|---|---|---|
| Ours opx-liq stacked | 53.59 | 3.30 |
| Ours opx-liq boosted | 56.24 | 3.12 |
| Ours opx-liq forest | 64.06 | 2.98 |
| Putirka 2008 opx-liq | 54.94 | 4.11 |
| Putirka 2008 opx-liq [true P] | 52.31 | 3.99 |
| Agreda-Lopez cpx-liq | 44.97 | 1.94 |
| Jorgenson cpx-only | 69.30 | 1.51 |

### 1.3 Narrative corrections (fix in manuscript)

1. **We do NOT beat Putirka on T.** Match at best against iterative (54.9), lose against [true P] (52.3). Beat on P by ~27% and on coverage (100% vs 98%).
2. **Stacking test-set performance lost**; ArcPL won. Not a clean win. Manuscript framing: "OOD-robustness ablation."
3. **P bias correction is SIGNIFICANT** (not NULL as handoff stated). T bias correction is NULL. Correct this in NB07 text.
4. **ArcPL T bias is +48.5 C forest, +20.95 C boosted.** Boosted is the primary model per test T01.
5. **nb10_two_pyroxene_benchmark.csv is empty** in v9 artifacts. Rerun NB10 cell during Phase D.

---

## 2. v10+v11 scope

### 2.1 Scientific goals

**Opx paper (2026 submission):** first opx-specific ML thermobarometer with formal OOD quantification and composition-conditional bias correction. Primary comparison vs Putirka 2008 opx-liq (we match on T, beat on P + coverage). Secondary framing: first opx-only ML that reports honest uncertainty.

**Cpx paper (2027 submission, research starts now):** framework-paper ambition. Cpx + two-pyroxene + (exploration) universal masking model. Comparisons vs Agreda-Lopez 2024, Jorgenson 2022, Petrelli 2020, Wang 2021. Winning claim: method gate (c) — beat ALL external cpx ML models on ArcPL AND LOSO AND natural-sample agreement.

### 2.2 Deliverables at end of v10+v11

**Opx paper:**
- Manuscript draft (NOT yet — we confirm results first per user decision)
- 20-30 canonical figures, all at JGR-MLC dimensions, Okabe-Ito, 300 dpi + PDF
- Tables 1-11
- Public github repo with pinned requirements, SHA256 data hashes, reproducibility script
- Pre-registered tests log

**Cpx paper (research state):**
- Trained cpx-only, cpx-liq, two-pyroxene, universal models
- Full benchmark vs external cpx ML + classical thermobarometers
- World map for opx + cpx + twopx natural samples (~100k samples total)
- Performance heatmap (methods × eval scope)
- Universal model exploration as separate notebook
- Preliminary Results section drafted, Methods section drafted

### 2.3 Explicit out-of-scope

- Docker container (unless reviewer asks)
- Streamlit deployment
- Cpx paper full draft (research state only in v10+v11 window)
- HuggingFace model release
- Neural network hyperparameter exhaustive search (single Optuna pass, not grid)

---

## 3. Model roster (locked)

**8 base models** across all pipelines:

| Model | Library | Role | Notes |
|---|---|---|---|
| RF | sklearn RandomForestRegressor | Tree baseline | Existing |
| ERT | sklearn ExtraTreesRegressor | Tree baseline | Existing |
| XGB | xgboost.XGBRegressor | Boosted baseline | Existing |
| GB | sklearn HistGradientBoosting | Boosted baseline | Existing |
| CatBoost | catboost.CatBoostRegressor | Tree, often SOTA on tabular | NEW |
| LightGBM | lightgbm.LGBMRegressor | Fast boosted | NEW |
| ElasticNet | sklearn.linear_model.ElasticNetCV | Linear baseline, interpretability story | NEW |
| MLP | sklearn.neural_network.MLPRegressor | NN baseline (sklearn for speed per user decision) | NEW |

**Ensemble methods** (4 compared):

1. Ridge stacking (current v9)
2. Two-level stacking (tree meta + linear meta, then Ridge over both)
3. Greedy ensemble selection (Caruana 2004)
4. AutoGluon-Tabular (automated)

Tested per pipeline. Winner per pipeline ships. See `v10_ensemble_methods_plan.md`.

**Feature sets per pipeline:** raw, alr, pwlr (unchanged from v9). Cpx pipeline adds `cpx_components` (Jd, Di, Aeg, CaTs, CrTs, CaTi, En, Fs fractions from structural formulas — see `v10_cpx_pipeline_plan.md`).

---

## 4. Pipeline roster (locked)

Five pipelines, each with the full 8-model + ensemble treatment:

| Pipeline | Train data | Primary benchmark | Paper |
|---|---|---|---|
| opx_only | ExPetDB opx | ArcPL opx-only subset | Opx (supplementary only per T06) |
| opx_liq | ExPetDB opx+liq | ArcPL opx-liq + Putirka | **Opx (primary)** |
| cpx_only | ExPetDB cpx + LEPR cpx | ArcPL cpx-only + Jorgenson | Cpx |
| cpx_liq | ExPetDB cpx+liq + LEPR | ArcPL cpx-liq + Agreda-Lopez | Cpx (primary) |
| twopx | ExPetDB pairs + LEPR pairs | ArcPL twopx + Putirka eq36/39 | Cpx |

**Plus one exploratory pipeline:**

| Pipeline | Purpose | Notebook | Diagnostics |
|---|---|---|---|
| universal | Masking-based, accepts any phase combo | `nb03_universal_exploration.ipynb` | **Completely isolated** per user request |

Per user decision: universal model diagnostics are NEVER combined with opx/cpx/twopx primary results. It gets its own figures, its own results, its own SHAP, its own ensemble test.

---

## 5. Notebook roster (14 notebooks)

Locked per user approval:

| NB | Name | Scope | Papers |
|---|---|---|---|
| 01 | `nb01_data_cleaning.ipynb` | Unified opx+cpx+twopx cleaning | Both |
| 02 | `nb02_eda_pca.ipynb` | Unified EDA, per-track PCA + clusters | Both |
| 03a | `nb03_opx_baseline_models.ipynb` | Opx test-first rebuild T01-T12 | Opx |
| 03b | `nb03_cpx_baseline_models.ipynb` | Cpx test-first rebuild T01-T12 | Cpx |
| 03c | `nb03_twopx_baseline_models.ipynb` | TwoPX test-first rebuild T01-T12 | Cpx |
| 03d | `nb03_universal_exploration.ipynb` | Universal masking model, isolated | Cpx (side) |
| 04 | `nb04_benchmark.ipynb` | External benchmarks + model heatmaps (merged NB04+NBM) | Both |
| 05 | `nb05_generalization.ipynb` | LOSO + Cluster-KFold + TargetBinKFold + LeaveOneRegionOut | Both |
| 06 | `nb06_shap_analysis.ipynb` | Tree-SHAP + Linear-SHAP on stack + KernelSHAP on MLP | Both |
| 07 | `nb07_bias_correction.ipynb` | Merged NB07+NB07b. ArcPL probes + composition-conditional T correction | Both |
| 08 | `nb08_natural_samples.ipynb` | Merged NB08+NB08b. Cross-mineral + world map (static + interactive) | Both |
| 09 | `nb09_manuscript_compilation.ipynb` | Tables, per-paper subsets | Both |
| 10 | `nb10_extended_analyses.ipynb` | OOD + MC + H2O + fix empty twopx_benchmark.csv | Both |
| F | `nbF_figures.ipynb` | Canonical manuscript figures per paper | Both |

NB03c (twopx) and NB03d (universal) are NEW. NB03 split from original v9 NB03.

---

## 6. Phase sequencing

### Phase A — cleanup + external models audit (1-2 days active)

1. Tag git: `pre_v10_cleanup_YYYY_MM_DD`
2. Execute `scripts/v10_phase_a_cleanup.py` (archive then delete per `v10_cleanup_manifest.md`)
3. Run `scripts/v10_external_models_audit.py` — verify Agreda/Jorgenson/Wang/Petrelli training data sources. Report in `v10_external_models_audit.md`.
4. Pull any external training data NOT already covered by our LEPR copy.
5. Pull GEOROC cpx global dataset (parallel to opx SGFTFN).
6. Pull GEOROC liquid/glass compositions.
7. Add `manuscripts/opx_2026/` and `manuscripts/cpx_2026/` dir skeletons.
8. Write `scripts/v10_phase_b_nb_audit.py`.

### Phase B — compat audit (1 day)

1. Run Phase B audit script per `v10_notebooks_compatibility_audit.md`.
2. Confirm 0 BLOCKING issues.
3. Patch any surface issues before Phase C.

### Phase C — opx rebuild (2-3 days active, ~3 h compute)

1. Rebuild `nb03_opx_baseline_models.ipynb` per test-first protocol T01-T12.
2. All 8 models in Optuna (48 × 2 = 96 studies for opx_liq and opx_only).
3. 4 ensemble methods tested.
4. Results logged to `results/v10_opx_test_log.csv`.

### Phase D — cpx rebuild (2-3 days active, ~4 h compute)

1. Build `nb03_cpx_baseline_models.ipynb` mirroring opx structure.
2. 8 models × 3 feature sets × 2 targets × 2 tracks = 96 Optuna studies for cpx.
3. Add cpx-specific engineered features (Jd, Di, Aeg, etc.).
4. Same test protocol T01-T12 re-run.

### Phase E — twopx rebuild (2 days active, ~3 h compute)

1. Build `nb03_twopx_baseline_models.ipynb`.
2. 48-96 Optuna studies (depends on how many feature sets make sense).
3. Same test protocol T01-T12.

### Phase F — universal exploration (3-4 days active, ~4 h compute)

1. Build `nb03_universal_exploration.ipynb` — side project.
2. Masking architecture: `[opx_oxides, cpx_oxides, liq_oxides, engineered, 3 mask bits]`.
3. Training on union of all experiments with any phase present.
4. 8-model Optuna.
5. **Diagnostics isolated** — separate figures dir, separate results files.

### Phase G — downstream rebuild (3-5 days active, ~2 h compute)

1. Rebuild NB04 benchmark (unified, all 5 pipelines + external models + Putirka).
2. Rebuild NB05 generalization (add LeaveOneRegionOut).
3. Rebuild NB06 SHAP (tree + linear + kernel).
4. Rebuild NB07 bias correction (merged + composition-conditional T).
5. Rebuild NB10 extended (fix twopx benchmark).
6. Build `v10_figure_audit.md` checklist and apply to every figure.
7. **Phase G.7 bias-correction mini-project (Option C, 2026-04-18).** Unified `src/prepare_train_test.py` dispatcher (A1) + `src/bias_correction.py` module with Form A (per-regime OLS) and Form B (quantile-thresholded piecewise) + ship-if-better decision (overall Delta-RMSE > SHIP_TOL AND no regime degrades). D2 driver runs 8 aggregate-best cells (opx-liq, opx-only, cpx-liq, cpx-only × T/P) × 20 seeds with checkpointing. A4 bin-edge sensitivity (+/-1 kbar on inner edges). A5 Form B CV-reseed stability (5 CV seeds). D3 post-correction scorecards. D3b GEOROC natural opx post-correction inference (regime-from-predicted-P, Agreda-Lopez convention). D4 figures 30-34. D5 tables T4/S9/S10. D7 manuscript autofill (bias_correction_autofilled.md). Registered T16-T18 (Section 13 of test protocol). Twopx and universal pipelines excluded from this mini-project.

### Phase H — natural worldwide (2-3 days active, ~1 h compute)

1. Re-merge lat/lon into `natural_opx_cleaned.csv` from raw SGFTFN.
2. Pull GEOROC cpx global if not already in Phase A.
3. Pull GEOROC liquid composition for context.
4. Curate ~15 localities with literature P-T for cross-check (see `v10_natural_worldwide_plan.md`).
5. Run inference with all final models on natural samples.
6. Opx-cpx-twopx convergence analysis.
7. Build world map (static + interactive, separate maps per mineral).

### Phase I — NB08 rebuild + figures (2-3 days active)

1. Build merged NB08 natural_samples.
2. Produce all comparison figures per `v10_figure_audit.md` NB08 section.
3. 2-way pairwise scatter matrices (my best vs every other method) with full metrics overlay (R², RMSE, slope, intercept, 1:1 deviation).

### Phase J — manuscript drafting

**Paused per user decision** — user wants to see all results first. Phase J starts only when user explicitly approves after reviewing Phase I output.

### Phase K — final figure polish + submission prep

Only reached after user approves manuscript drafting in J.

---

## 7. Test-first protocol summary

See `v10_nb03_test_protocol.md` for details. **Twelve tests per primary pipeline plus two additional tests in the universal exploration track.**

### Primary tests T01-T12 (opx, cpx, twopx pipelines)

| Test | Hypothesis |
|---|---|
| T01 | Boosted is primary |
| T02 | Ensemble beats best base (compares 4 ensemble methods) |
| T03 | Resampling hurts (replicates v9 finding; re-tests per pipeline) |
| T04 | N_AUG=1 beats N_AUG=5 |
| T05 | Feature set winners reproduce v9 (opx only) / are stable (cpx, twopx) |
| T06 | Mineral-only pipeline is worse than +liq (for opx_only and cpx_only) |
| T07 | Composition-conditional bias correction improves ArcPL T |
| T08 | P piecewise bias correction improves test P |
| T09 | CV-predict stacking beats full-fit stacking |
| T10 | IsolationForest OOD correlates with residual magnitude |
| T11 | CatBoost or LightGBM beats XGB on any pipeline (NEW) |
| T12 | MLP or ElasticNet beats tree models on any pipeline (NEW) |

Each test runs per pipeline. Results logged to `results/v10_nb03_test_log.csv` with `pipeline` column.

### Universal-only tests T13-T14 (isolated)

Defined in `docs/v10_universal_model_exploration.md`. Run only in `nb03_universal_exploration.ipynb`. Logged to separate `results/universal/v10_nb03_test_log.csv`. Never combined with primary pipeline results.

| Test | Hypothesis |
|---|---|
| T13 | Universal model degrades gracefully (1-phase < 2-phase < 3-phase in R^2 ordering) |
| T14 | Universal with all 3 phases beats specialized opx-liq / cpx-liq / twopx models on same sample |

Decision tree: T13 fails -> ablate universal architecture, exploration ends. T13 passes but T14 fails -> keep as honest-negative exploration artifact. Both pass -> universal becomes lead novelty claim in cpx paper.

### Regime claim and bias-correction tests T15-T18

| Test | Hypothesis |
|---|---|
| T15 | Pre-registered regime headline claim: at least one regime with n >= 20 shows opx-liq v10-outperforms-Putirka for P_kbar at the two-axis honesty bar |
| T16 | Phase G.7 D2: at least one cell ships a bias correction (canonical seed=42 winner in {A, B}) |
| T17 | Phase G.7 D2: shipped corrections persist on a majority of 20 SPLIT_SEEDS |
| T18 | Phase G.7 A4+A5: Form A edge swing and Form B CV-reseed stability within thresholds |

See `docs/v10_nb03_test_protocol.md` Sections 12-13 for full specifications.

---

## 8. "Beat Agreda / Jorgenson" gate

Per user decision on gate (c):

To claim "our cpx model beats Agreda-Lopez 2024" or "our cpx model beats Jorgenson 2022", ALL of these must hold:

1. Lower RMSE on ArcPL Kd-eq subset n=96 (both T and P)
2. Lower RMSE on LOSO cross-validation (both T and P)
3. Better natural-sample cross-mineral agreement (lower |T residual| and |P residual| on the 327-sample natural set)
4. All three with 95% bootstrap CI excluding their numbers

If 2 of 3 hold: claim "competitive with." If 1 of 3 holds: claim "matches on [specific metric]." If 0 of 3: claim "confirms their result."

Honest framing is the goal. Reviewer cannot catch us inflating the claim.

---

## 9. Repository structure

Per user decision:

```
Final Project/
  .git/
  .venv/
  archive/
    pre_v10_rebuild_YYYY_MM_DD/        <- v9 artifacts
    v7_preparation_20260414_164844/
    ...
  config.py                            <- updated with cpx + twopx constants
  data/
    raw/
      ExPetDB*.xlsx
    processed/
      opx_clean_core.parquet
      cpx_clean_core.parquet           <- NEW Phase A
      twopx_clean_core.parquet         <- NEW Phase E
      universal_clean_core.parquet     <- NEW Phase F
    external/
      agreda_lopez_2024/
      jorgenson_2022/
      thermobar_examples/
    natural/
      2024-12-SGFTFN_ORTHOPYROXENES.csv
      2024-XX-GEOROC_CLINOPYROXENES.csv <- NEW Phase A
      natural_opx_with_coords.csv       <- NEW Phase H
      natural_cpx_with_coords.csv       <- NEW Phase H
      natural_twopx_pairs.csv           <- NEW Phase H
    splits/
      *_opx*.npy, *_cpx*.npy, *_twopx*.npy <- expanded
    hashes.json                         <- NEW Phase A, SHA256 per file
  docs/
    v10_*.md                            <- 13 new docs
    stacking_strategy.md                <- v10 update block
    resampling_strategy.md              <- v10 update block
    optuna_strategy.md                  <- v10 update block
    [v9 historical docs kept]
  figures/
    opx/                                <- opx paper figures
    cpx/                                <- cpx paper figures
    twopx/
    universal/                          <- isolated per user decision
  logs/
  manuscripts/
    opx_2026/
      figures/                          <- soft-linked to ../figures/opx/
      tables/
      text/
      arxiv_submission/
    cpx_2026/                           <- research state only in v10+v11
      figures/
      tables/
      text/
      arxiv_submission/
  models/
    canonical/
      opx/                              <- 8 base + 4 ensemble per target
      cpx/
      twopx/
    external/                           <- vendor: Agreda, Jorgenson, Wang, Petrelli
    ablation/
      resampled/
  notebooks/
    nb01_data_cleaning.ipynb
    nb02_eda_pca.ipynb
    nb03_opx_baseline_models.ipynb
    nb03_cpx_baseline_models.ipynb
    nb03_twopx_baseline_models.ipynb
    nb03_universal_exploration.ipynb
    nb04_benchmark.ipynb
    nb05_generalization.ipynb
    nb06_shap_analysis.ipynb
    nb07_bias_correction.ipynb
    nb08_natural_samples.ipynb
    nb09_manuscript_compilation.ipynb
    nb10_extended_analyses.ipynb
    nbF_figures.ipynb
  README.md
  PROJECT_OVERVIEW.md
  requirements.txt                     <- updated: catboost, lightgbm, autogluon, folium, cartopy
  results/
    {pipeline}/                        <- opx/, cpx/, twopx/, universal/
    v10_opx_test_log.csv
    v10_cpx_test_log.csv
    v10_twopx_test_log.csv
    v10_universal_test_log.csv
  scripts/
    audit_*                            <- v9 kept
    v10_phase_a_cleanup.py
    v10_phase_b_nb_audit.py
    v10_pull_external_training_data.py
    v10_pull_georoc_cpx.py
    v10_world_map_static.py
    v10_world_map_interactive.py
    v10_audit_notebook_markdown.py
    v10_figure_audit_checker.py
  src/
    __init__.py
    calibration.py
    cpx_features.py                    <- NEW cpx-specific structural formulas
    data.py                            <- expanded with cpx/twopx loaders
    evaluation.py                      <- adds LeaveOneRegionOut
    external_models.py                 <- adds Petrelli 2020, AutoGluon wrappers
    features.py
    geotherm.py
    io_utils.py
    models.py                          <- adds CatBoost, LightGBM, ElasticNet, MLP
    optuna_search.py                   <- extended for new models
    plot_style.py                      <- Okabe-Ito enforcement
    resampling.py
    shap_utils.py                      <- NEW unified SHAP across tree/linear/NN
    stacking.py                        <- fixed CV-predict per T09
    twopx_features.py                  <- NEW twopx-specific
    universal_features.py              <- NEW masking arch
    world_map.py                       <- NEW cartopy + folium
    ablations/                         <- NEW scaffold for failed-test code
```

---

## 10. Risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | Cleanup wipes a hard-to-regenerate file | Low | Medium | Git tag + archive before delete |
| 2 | Downstream NB hardcoded path breaks | Medium | Medium | Phase B audit catches |
| 3 | Cpx training data unavailable / licensing | Medium | High | Phase A audit first; fallback to our LEPR cpx |
| 4 | CatBoost / LightGBM adds compute time beyond estimate | Low | Low | User OK'd "comprehensiveness over runtime" |
| 5 | MLP doesn't converge on small training set | Medium | Low | Report null result per test T12, ablate |
| 6 | Universal model underperforms all specialized models | Medium | Low | User pre-approved this as side exploration |
| 7 | Natural sample cross-check disagrees across cpx models | Medium | Medium | Report honest divergence; framing opportunity |
| 8 | World map rendering at 100k samples too slow | Low | Low | Hex-bin density, not point-per-sample |
| 9 | Composition-conditional T correction overfits ArcPL | Medium | Medium | Fit on in-domain GroupKFold only |
| 10 | AutoGluon installation on Windows has issues | Medium | Low | Ship without if broken; test Ridge/two-level/greedy which are lightweight |
| 11 | Reviewer rejects opx paper because "why not cpx?" | Low | Medium | Answer: cpx paper in prep, framework generalization deferred. Honest scope. |
| 12 | Commissioning interrupts Phase G-I | Medium | Medium | Plan designed for interruptible overnight runs + multi-session chat |

---

## 11. Approval gate

Before Phase A execution, user checklist:

- [ ] Reviewed this master plan
- [ ] Reviewed `v10_cleanup_manifest.md`
- [ ] Reviewed `v10_notebooks_compatibility_audit.md`
- [ ] Reviewed `v10_nb03_test_protocol.md`
- [ ] Reviewed `v10_figure_audit.md` (key since user concerned about figures)
- [ ] Reviewed `v10_cpx_pipeline_plan.md`, `v10_twopx_pipeline_plan.md`
- [ ] Reviewed `v10_universal_model_exploration.md`
- [ ] Verified ~3 GB free disk for new artifacts + archive
- [ ] `git tag pre_v10_cleanup_$(date +%Y_%m_%d)` committed and pushed
- [ ] User confirms: "I have seen all plan documents and approve Phase A execution"

---

## 12. Post-submission vision

After opx paper submits (end of Phase J-K):

- Cpx paper Phase L: full manuscript draft
- Universal paper (if results warrant): separate submission to ML methods venue
- Reviewer revisions budget: 2-4 weeks per round

PhD integration: Phase A-I results become foundation for Year 1 of Stony Brook PhD. Dr. Lee / JJ Dong context: ML thermobarometry is the methodology work; DAC mineral physics is the application. Paper cites PhD advisor affiliation.

---

## 13. Where to look next

- Phase A details: `v10_cleanup_manifest.md`, `v10_external_models_audit.md`
- Phase C details: `v10_nb03_test_protocol.md`
- Phase D details: `v10_cpx_pipeline_plan.md`
- Phase E details: `v10_twopx_pipeline_plan.md`
- Phase F details: `v10_universal_model_exploration.md`
- Phase G figures: `v10_figure_audit.md`
- Phase H natural: `v10_natural_worldwide_plan.md`
- Ensemble methods (cross-phase): `v10_ensemble_methods_plan.md`
- Stacking everywhere (cross-phase): `v10_stacking_propagation_audit.md`
- Per-cell rigor: `v10_markdown_template.md`
