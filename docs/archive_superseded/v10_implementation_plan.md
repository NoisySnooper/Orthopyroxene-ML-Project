# v10 implementation plan

**Status:** proposed, pending user approval
**Author:** NQTa (drafted with Claude assistance 2026-04-16)
**Supersedes:** remaining-work section in the v9 handoff summary

This is the master plan for the post-v9 push to manuscript submission. Work companions:

- `v10_nb03_test_protocol.md` — test-first rebuild protocol for NB03
- `v10_cleanup_manifest.md` — explicit file/model delete list (fresh start)
- `v10_notebooks_compatibility_audit.md` — downstream NB compatibility matrix
- `stacking_strategy.md`, `resampling_strategy.md`, `optuna_strategy.md` — methodology docs updated with v9 empirical outcomes

---

## 1. Situation summary

### 1.1 Numbers on record (verified 2026-04-16)

**Test set, 20-seed means** (`nb03_multi_seed_summary.csv`):

| Track | Target | Best model | RMSE | R^2 |
|---|---|---|---|---|
| opx_liq | T_C | XGB+alr | 77.77 (+/-9.5) | 0.76 |
| opx_liq | P_kbar | XGB+raw | 4.81 (+/-0.8) | 0.74 |
| opx_only | T_C | RF+pwlr | 139.5 (+/-18.4) | 0.50 |
| opx_only | P_kbar | GB+pwlr | 10.65 (+/-2.8) | 0.28 |

**ArcPL full external n=197** (recomputed directly from `nb04_arcpl_opx_liq_predictions_*.csv`):

| Family | T bias | T RMSE | P bias | P RMSE |
|---|---|---|---|---|
| Forest (RF) | +48.51 C | 71.60 | +1.24 kbar | 2.79 |
| Boosted (XGB) | +20.95 C | 63.68 | -0.09 kbar | 2.70 |

**ArcPL Kd-equilibrated n=96** (`nb04_method_benchmark_arcpl_paired_kdEq.csv`):

| Method | T RMSE | T R^2 | P RMSE | P R^2 |
|---|---|---|---|---|
| Ours opx-liq stacked | 53.59 | 0.632 | 3.30 | 0.282 |
| Ours opx-liq boosted | 56.24 | 0.594 | 3.12 | 0.357 |
| Ours opx-liq forest | 64.06 | 0.474 | 2.98 | 0.413 |
| Putirka 2008 opx-liq | 54.94 | 0.618 | 4.11 | -0.111 |
| Agreda-Lopez cpx-liq | 44.97 | 0.741 | 1.94 | 0.750 |
| Jorgenson cpx-only | 69.30 | 0.384 | 1.51 | 0.849 |

### 1.2 Findings that correct the v9 handoff

1. **Stacking collapsed on test, won on ArcPL.** Alpha pegged at 100 (endpoint) for all 4 targets. Test RMSE: stacked T=89.4 vs best base RF=84.9 (worse by 4.5). ArcPL RMSE: stacked T=53.6 vs boosted T=56.2 (better by 2.7). Story is "stacking protects against distribution shift at cost of in-distribution performance." Handoff oversold stacking as a clean win.

2. **Bias correction P result is SIGNIFICANT, not null.** `nb07_bias_correction_null_result.csv` shows T piecewise within noise (null), P piecewise improves over raw RF by 0.27 kbar with CI not containing zero. Handoff says "NULL result on test" for both — wrong for P.

3. **Forest T bias on ArcPL is +48.5 C, boosted is +20.95 C.** Handoff cites "+37 C from v8" as carry-over. Forest is worse than v8; boosted is half that. Boosted is the correct primary model.

4. **Boosted T residual is wide, weakly bimodal, dominated by regression-to-mean** (r = -0.68 vs true T). Strong composition signal: liq_TiO2 r = -0.55, H2O_Liq r = +0.48, liq_K2O r = +0.45. Arc magmas (hot H2O-rich) get underpredicted in T.

5. **`nb10_two_pyroxene_benchmark.csv` is empty.** Handoff claims NB10 two-pyroxene executed. False.

6. **`CANONICAL_FIGURES` in `config.py` is stale.** Lists 23 figures, file stems do not match actual files in `figures/`. NBF regeneration needed.

### 1.3 v9 scorecard (helped / hurt / wash)

| Change | Verdict | Evidence |
|---|---|---|
| Optuna TPE replaces HalvingRandomSearchCV | Help (methodology) | 48 studies stable, frozen params defensible, no failed studies |
| Ridge stacking | Complicated | Hurt on test (4 of 4), helped on ArcPL T (1 of 4). Keep as OOD ablation, not canonical. |
| P-T tempered resampling | Hurt | 6/8 configs worse on test, hurt on ArcPL. Correctly ablated. |
| N_AUG=1 (augmentation disabled) | Help | NB03 sensitivity test justified drop. Keep. |
| Per-base-model feature_set in stacking | Wash | Right call but base corr still 0.96+, feature diversity gave almost nothing. |
| Option B Kd-equilibrated scope | Help (honesty) | Head-to-head with Putirka defensible. Keep. |

---

## 2. v10 goals

Primary: submit opx-only manuscript to JGR ML & Computation.

Secondary (per Dr. Lee 2026-04-16 whiteboard meeting):

- Worldwide / natural-sample generalization test (partial via NB08 already, extend to GEOROC)
- Parallel cpx pipeline benchmarked against Agreda-Lopez / Jorgenson / Wang
- Opx-cpx convergence on natural samples

v10 covers the primary in full. Secondary work spins off into v11.

---

## 3. Scope and sequencing

### 3.1 Phase A: fresh-start cleanup (est. 2 hours)

User approved fresh start. Everything v9 in `results/`, `figures/`, `models/` (except `models/external/`), `logs/`, executed notebooks, optuna studies — all archived under `archive/pre_v10_rebuild_YYYY_MM_DD/` and then deleted from the canonical tree.

See `v10_cleanup_manifest.md` for the explicit file list. Quick summary:

- `results/*` — archive then delete (79 files). Regeneratable.
- `results/optuna_studies/*` — archive then delete (48 studies, regenerate in Phase C)
- `figures/*` — archive then delete (64 figures). Regeneratable.
- `models/*.joblib` at root — archive then delete (37 joblibs). Retrain.
- `models/external/*` — **KEEP** (Agreda-Lopez onnx/joblib are vendor artifacts, not regeneratable)
- `notebooks/executed/*` — archive then delete. Regenerate via papermill.
- `logs/*` except `.gitkeep` — archive then delete
- `data/processed/*` parquet — **KEEP** (NB01 output, still valid)
- `data/splits/*.npy` — **KEEP** (canonical train/test split, frozen SEED_SPLIT=42)
- `data/raw/*` — **KEEP** (ExPetDB source)

Tag git before cleanup: `git tag pre_v10_cleanup_$(date)`.

### 3.2 Phase B: downstream compatibility audit (est. 1 hour)

**Before** rebuilding NB03, verify that every downstream notebook reads NB03 output through the documented public API (`canonical_model_filename`, `canonical_model_spec`, `load_canonical_model`) and does not hardcode feature_set names, model filenames, or test-set indices.

See `v10_notebooks_compatibility_audit.md` for the per-notebook matrix. Must pass before Phase C begins.

Expected outcome: NB04 through NBF pass unchanged if NB03 v10 writes the same JSON schema (`nb03_per_family_winners.json`, `nb03_stacked_members_*.json`, etc.) and the same joblib filenames. Test this empirically by swapping the JSON with a test fixture and confirming each NB can at least load the references.

### 3.3 Phase C: NB03 test-first rebuild (est. 4-6 hours)

The core scientific rebuild. Every design change gets a pre-registered test. If the test passes, the change ships to canonical. If it fails, the change is ablated with a one-line result table in this plan's Section 7.

See `v10_nb03_test_protocol.md` for the full protocol. Summary of tests to run:

| Test # | Hypothesis | Test design | Accept criterion |
|---|---|---|---|
| T01 | Boosted is the best primary model on ArcPL | Compare RF/ERT/XGB/GB T bias and RMSE on n=197 ArcPL | Lowest |T bias| AND RMSE within 5% of best |
| T02 | Stacking improves ArcPL T but hurts test T | Re-fit meta, report both metrics | ArcPL T better by >=1 C AND test T worse by <=5 C. Expected ship: stacking as OOD-robustness ablation only. |
| T03 | Resampling hurts | Re-run resampled variants, compare to non-resampled | 6/8 configs worse: confirm handoff claim. Expected ship: ablate. |
| T04 | N_AUG=1 beats N_AUG=5 | Fit both, compare test RMSE | N_AUG=1 test RMSE <= N_AUG=5 across both targets |
| T05 | pwlr wins T, alr/raw wins P (opx_liq) | Confirm feature-set winners stable vs v9 | 4 of 4 winners match v9 |
| T06 | opx_only ML is worse than Putirka 2-px eq36 | Compare on ArcPL Kd-eq subset | opx_only R^2 < 0; confirm supplementary-only framing |
| T07 | Composition-conditional bias correction improves ArcPL T | Fit correction on in-domain GroupKFold, apply to ArcPL | ArcPL T RMSE drops by >=5 C with CI excluding zero |
| T08 | P bias correction (piecewise) improves test P | Fit piecewise on train, eval on test | Test P RMSE drops with CI excluding zero (replicate nb07 v9 finding) |
| T09 | Fix stacking distributional mismatch: use CV-predict at test time | Compare legacy stacking vs CV-predict stacking | CV-predict test RMSE improves by >=1 C OR no worse |
| T10 | IsolationForest OOD flag correlates with residual magnitude | Compute OOD scores on ArcPL, bin by score, check residual trend | Monotonic trend with Spearman r >= 0.2 |

Each test is its own NB03 cell block. Result logged to `results/v10_nb03_test_log.csv` with columns (test_id, hypothesis, metric, observed, threshold, passed, shipped).

Tests whose results contradict prior handoff claims get an explicit paragraph in NB03 markdown: "We tested X. Hypothesis: Y. Observed: Z. Ship/ablate."

### 3.4 Phase D: downstream re-run (est. 3 hours)

If Phase B passes, the downstream NBs run in order under papermill:

1. NB04 (Putirka + cpx benchmark, Option B Kd-filter scope preserved)
2. NB05 (LOSO + Cluster-KFold + TargetBin)
3. NB06 (SHAP + robustness appendix)
4. NB07 (P bias correction — claim updated from NULL to SIGNIFICANT for P)
5. NB07b (ArcPL bias probe — run on v10 predictions, verify +48.5 / +20.95 numbers match)
6. NB08 (natural twopx — confirm opx-only bias +80 C reproduces, acknowledge in manuscript)
7. NB10 (ensure two_pyroxene_benchmark.csv is populated this time)
8. NB09 (manuscript compilation)
9. NBF (canonical figures — reconcile `CANONICAL_FIGURES` list with actual files first)

### 3.5 Phase E: manuscript polish (est. 6-10 hours)

Only after Phase D's pipeline_health.txt passes 23/23.

- Re-draft Methods to reflect actual shipped decisions (stacking OOD-only, P bias correction significant, composition-conditional T correction)
- Re-draft Results around the boosted-primary narrative
- Update every figure caption with verified n and RMSE
- Re-write figure style guide `docs/figure_style_guide.md` (new), apply to all figures (Okabe-Ito, JGR-MLC sizing)
- Write `docs/manuscript_methods_paragraph.md` (new) with the paragraph template agreed with Lee

---

## 4. Test-first protocol summary (see v10_nb03_test_protocol.md for detail)

Each NB03 change follows this pattern, in NB03 markdown and in `results/v10_nb03_test_log.csv`:

```
## Test T0X: [hypothesis in one sentence]

Design: [3-line description of how we test]
Accept criterion: [concrete numeric threshold]

### Result

Observed: [actual number from the run]
Passed: [YES/NO]
Shipped: [YES/NO/ABLATION-ONLY]
Commentary: [1-2 sentences on what this means, especially if result is unexpected]
```

When a test fails, the code path for the proposed change is still retained but moved to `src/ablations/` and the canonical path bypasses it. An `ABLATION-ONLY` note in the markdown explains why.

---

## 5. Documentation updates

### 5.1 New docs (written in Phase A alongside this plan)

- `v10_implementation_plan.md` (this file)
- `v10_nb03_test_protocol.md`
- `v10_cleanup_manifest.md`
- `v10_notebooks_compatibility_audit.md`
- `figure_style_guide.md` (Phase E)
- `manuscript_methods_paragraph.md` (Phase E)

### 5.2 Existing docs updated in Phase A

- `stacking_strategy.md` — append v9 outcome block showing empirical results (stacking collapsed on test, won on ArcPL). Add shipped-decision paragraph.
- `resampling_strategy.md` — append v9 outcome block (hurt 6/8 configs, ablated).
- `optuna_strategy.md` — append v9 outcome block (stable, no failed studies, expected runtime matched).
- `README.md` — update headline numbers to verified v9 values, add v10 path and status.
- `PROJECT_OVERVIEW.md` — update remaining-work section to point to this plan.

### 5.3 Naming and versioning convention

All v10 docs use the `v10_` prefix so they are greppable. Existing audit docs (`codebase_consistency_audit_optionB.md`, `putirka_inconsistency_audit.md`, `putirka_kd_filter_lookup.md`) keep current names — they are historical and do not need version bumps.

---

## 6. Risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | Cleanup wipes a file that is harder to regenerate than expected | Low | Medium | Git tag before cleanup; archive to `archive/pre_v10_rebuild_*` before delete |
| 2 | Downstream NB hardcodes a filename that v10 renames | Medium | Medium | Phase B audit catches this before Phase C runs |
| 3 | Test T07 (composition-conditional bias correction) overfits to ArcPL | Medium | Medium | Fit correction on in-domain GroupKFold only; apply to ArcPL as true external eval. Report both raw and corrected numbers per probe7 recommendation |
| 4 | Optuna search converges to a different point than v9 (non-reproducible) | Low | Low | Seed is pinned (42). If it diverges, investigate Optuna version bump |
| 5 | NB04 Thermobar pin drift | Low | High | Thermobar 1.0.70 pinned in config; re-audit K->C conversion if version changes |
| 6 | User runs low on disk during cleanup (archive + delete both present for a moment) | Low | Low | Cleanup script checks free space before archiving |
| 7 | NB08 opx-only bias (+80 C on natural samples) gets challenged by reviewer as disqualifying | Medium | High | Pre-empt in Discussion: "opx-only is fundamentally limited; framework paper posits cpx-liq as primary use case". Keep opx-only as supplementary only. |
| 8 | Boosted T bimodality reviewer questions | Medium | Medium | T07 composition-conditional correction should address; include residual-vs-composition scatter plots in SI |

---

## 7. v10 test log (populated during Phase C)

Empty at start of Phase C. Each row appended after test execution.

| Test ID | Hypothesis | Observed | Threshold | Passed | Shipped | Date |
|---|---|---|---|---|---|---|
| T01 | | | | | | |
| T02 | | | | | | |
| T03 | | | | | | |
| T04 | | | | | | |
| T05 | | | | | | |
| T06 | | | | | | |
| T07 | | | | | | |
| T08 | | | | | | |
| T09 | | | | | | |
| T10 | | | | | | |

---

## 8. Approval gate

User must explicitly approve this plan before Phase A execution. Checklist:

- [ ] Read `v10_cleanup_manifest.md` and confirm fresh-start list
- [ ] Read `v10_notebooks_compatibility_audit.md` and confirm NB compatibility expectations
- [ ] Read `v10_nb03_test_protocol.md` and confirm test designs (especially T07, T09 which are new)
- [ ] Verify `archive/pre_v10_rebuild_*` has disk space (~700 MB needed)
- [ ] Run `git tag pre_v10_cleanup_$(date +%Y_%m_%d)` and push

Once approved, Phase A through D run sequentially. Phase E begins only after the 23/23 pipeline_health.txt PASS.

---

## 9. Post-v10: framework paper prep (v11 scope, not this document)

The cpx parallel pipeline, worldwide-sample generalization, and opx-cpx convergence analysis are out of scope for v10. They inherit the v10 codebase cleanly: same `src/features.py`, same `src/models.py`, new `config.py` constants for cpx, new `nb03_cpx_*` notebooks. Planned as v11 once the opx manuscript is submitted.
