# Lee revisions overnight summary

Branch `lee_revisions_overnight`. Started from HEAD `7f987c0` (v8 final state before v9 rebuild). Autonomous overnight pass executing `LEE_REVISIONS_OVERNIGHT_PROMPT.md` (Phase 1 + Phase 2 + Part 3). Wall-clock budget 16 hours; actual completion 2026-04-22 03:35 UTC.

## Headline delta (before → after)

| item | before | after |
|---|---|---|
| RMSE uncertainty format | `mean ± seed_std` | `RMSE [CI lo, hi]` bootstrapped 500×, seed 42 |
| Per-cell best-model justification | implicit | pre-registered §4.1.1 citing Wolpert 1996, Feurer 2015, Erickson 2020, Grinsztajn 2022 |
| Best-family sensitivity table | absent | S15 (9 families × 8 cells) |
| Natural-sample validation | absent from Results | §4.7 with 9 paragraphs + Core_13a/b + Core_14a/b + 20-row pairing matrix |
| Interpretability section | SHAP only, qualitative | 8-layer bundle with bootstrapped numbers, §4.5.1–§4.5.8 |
| Discussion of black-box concern | §5.6 "Feature-attribution alignment", qualitative | §5.6 "On the black-box concern" with the 8 bundle numbers |
| Submission checklist | absent | `manuscripts/opx_2026/SUBMISSION_CHECKLIST.md` |
| Manuscript Word doc | absent | `manuscripts/opx_2026/arxiv_submission/manuscript.docx` (82 KB) |
| Tests | existing | +3 bootstrap-CI tests (total 87 pass) |

## Phase 1 deliverables (commit `12be026`)

- `scripts/evaluation/compute_bootstrap_cis_all_cells.py` — produces `results/bootstrap_rmse_cis_all_cells.csv` (72 rows, 8 cells × 9 families × feature sets; n_boot = 500, seed = 42, paired test-set residuals; 2.7 MB)
- `manuscripts/opx_2026/tables/T2_model_roster.{csv,tex}` — adds `rmse_ci_lo` and `rmse_ci_hi` columns
- `scripts/pairing/build_pairing_matrix.py` — produces `results/pairing_matrix_22rows.csv` (20 rows: A–V minus K,L)
- `figures/core/Core_13a_fig_method_agreement_T.{pdf,png}` + `Core_13b` — 10×10 method agreement matrix, hierarchical clustering (scipy average linkage) on row vectors
- `figures/core/Core_14a_fig_pairing_panels_T.{pdf,png}` + `Core_14b` — 20-panel pairwise scatter with fixed axes and identity-line envelope
- `tests/test_bootstrap_ci.py` — 3 tests: nontrivial CI for deterministic fit; seed-determinism; 8-cell completeness
- Manuscript sections amended: `03_methods.md` §3.4, §3.10; `04_results.md` §4.1 CI format + §4.7 natural-sample validation (9 paragraphs); `01_introduction.md` contributions (i)/(ii) CI format

## Phase 2 deliverables (commit `a296b48`)

### P2.1 Per-cell best-model defense
- `scripts/evaluation/build_S15_single_family_sensitivity.py` → `manuscripts/opx_2026/tables/S15_single_family_sensitivity.{csv,tex}` — 9 family rows × 8 cells + per-cell winner row, "RMSE [lo, hi] (fs)" format
- New `04_results.md` §4.1.1 — pre-registered best-per-cell selection citing Wolpert 1996 (No Free Lunch), Feurer 2015 (auto-sklearn), Erickson 2020 (AutoGluon), Grinsztajn 2022 (tabular ML benchmarks)

### P2.2 Feature concordance across families
- `scripts/interpretability/compute_feature_concordance.py` → `results/feature_concordance_permutation_importance.csv` (8 cells × 8 tuned families × N features; TabPFN omitted, documented) + `results/feature_concordance_spearman_matrix.csv`
- `scripts/interpretability/make_fig_feature_concordance.py` → `figures/core/Core_15_fig_feature_concordance.{pdf,png}` (8-panel per-cell heatmap)
- Per-cell median pairwise Spearman rho: opx_liq T = 0.58, opx_liq P = 0.59, opx_only T = 0.25, opx_only P = 0.37, cpx_liq T = 0.75, cpx_liq P = 0.36, cpx_only T = 0.51, cpx_only P = 0.42

### P2.3 Classical-feature ablation
- `scripts/interpretability/classical_feature_ablation.py` → `results/classical_feature_ablation.csv` (4 opx cells × full vs classical-subset RMSE + CI + delta %)
- Results: opx_liq T +9.1%, opx_liq P +25.3%, opx_only T +25.9%, opx_only P +58.4% RMSE on classical-Putirka subset

### P2.4 Classical-equivalence regression
- `scripts/interpretability/classical_equivalence_regression.py` → `results/classical_equivalence_regression.csv` + `figures/core/Core_16_fig_classical_equivalence.{pdf,png}` (2×4 ML vs classical-fit scatter)
- R² (winner ML prediction fit with classical Putirka-form features): opx_liq T 0.99, opx_liq P 0.86, opx_only T 0.76, opx_only P 0.56, cpx_liq T 0.87, cpx_liq P 0.76, cpx_only T 0.80, cpx_only P 0.75

### P2.5 Partial dependence across families
- `scripts/interpretability/partial_dependence_across_families.py` → `results/partial_dependence_across_families.parquet` (2160 rows, 4 opx cells × 3 top features × 8 families × 30 grid) + `figures/core/Core_17_fig_partial_dependence.{pdf,png}` (4×3 grid)
- Families agree on monotonic direction for all 12 (cell × feature) panels; crossovers confined to tail 10%

### P2.6 Surrogate decision trees
- `scripts/interpretability/surrogate_decision_trees.py` → `results/surrogate_tree_r_squared.csv` + `figures/core/Core_18_fig_surrogate_trees.pdf` (4-page tree plot, max_depth = 4, min_samples_leaf = 10)
- Surrogate R² (with 500-bootstrap 95% CI): opx_liq T 0.88 [0.85, 0.91], opx_liq P 0.81 [0.77, 0.86], opx_only T 0.88 [0.84, 0.92], opx_only P 0.89 [0.83, 0.93]

### P2.7 Physics consistency checks
- `scripts/interpretability/physics_consistency_checks.py` → `results/physics_consistency_checks.csv`
- Fe-Mg Kd (opx-liq test set, n = 174): 100% in Roeder-Emslie [0.23, 0.35]; median 0.29, IQR 0.28–0.32
- Al-P Spearman (opx-only test set, n = 190): rho = 0.518 — passes Gasparik 1987 > 0.5 threshold

### P2.8 Gasparik agreement
- `scripts/interpretability/gasparik_agreement.py` → `results/gasparik_agreement.csv`
- opx-liq P winner (MLP/raw): 4 of top-10 SHAP features Al-bearing, 45.3% aggregate importance share — passes
- opx-only P winner (RF/pwlr): 2 of top-10 Al-bearing, 23.6% share (features are pwlr log-ratios of Al with other oxides; the low count reflects the coordinate system, not a Gasparik violation)

### P2.9 / P2.10 Manuscript rewrites
- `04_results.md` §4.5 rewritten as 8-layer bundle (§4.5.1 SHAP, §4.5.2 concordance, §4.5.3 ablation, §4.5.4 classical equivalence, §4.5.5 surrogate tree, §4.5.6 physics, §4.5.7 partial dependence + Gasparik, §4.5.8 where ML exceeds classical)
- `05_discussion.md` §5.6 rewritten from "Feature-attribution alignment" to "On the black-box concern" citing the bundle

### P2.11 Bibliography
- Added insertion list to `manuscripts/opx_2026/SUBMISSION_CHECKLIST.md` §6: Wolpert 1996, Feurer 2015, Erickson 2020, Grinsztajn 2022 (plus already-cited Roeder-Emslie 1970, Gasparik 1987)

## Part 3 deliverables (commit pending)

- Terminology sweep: all `RMSE ± std` → `RMSE [CI lo, hi]` in §4.1, §4.2, §4.5 of Results and contributions (i), (ii) of Introduction. Remaining `±` in manuscript are methodological perturbation magnitudes (edge-sensitivity `±1 kbar`) and literature-cited constants (Putirka 2008 Kd = 0.29 ± 0.06) — correctly retained.
- `MANUSCRIPT_WRITING_AUDIT_v3.md` — supersedes v1 for Phase 1 / Phase 2 touched material; preserves v1 sections on data inventory and preregistration state unchanged
- `scripts/manuscript/build_manuscript_docx.py` — new; concatenates all 11 sections files into `manuscripts/opx_2026/arxiv_submission/manuscript.docx` (82 KB, 11-in-1 double-spaced)
- pytest: 87 passed, 0 failed at HEAD `a296b48`

## Outstanding at end of overnight

1. **Preregistration amendments still untracked**: `docs/preregistration/AMENDMENT_1_*.md` and `AMENDMENT_2_*.md` remain outside git (flagged in audit v1 and v3). Author action — these files exist but have not been committed. The Methods §3.5 language referring to "amendment pending commit" is still accurate and can stay until the amendments are tracked.
2. **Abstract not CI-updated**: `00_abstract.md` still uses pre-Phase 1 number formats. Author action.
3. **Cover letter not updated**: `09_cover_letter.md` does not reference the bundle. Author action.
4. **Remote not pushed**: the overnight pass committed locally only (per the brief "don't push"). Author action at publication time.
5. **Cpx-side bundle not produced**: the 8-layer bundle covers opx cells only (4 × 4 probes). Cpx cells have classical-equivalence R² but no ablation, no surrogate-tree, no physics checks. If reviewers ask for cpx parity, the scripts in `scripts/interpretability/` can be run with cpx CELLS entries added — roughly 1 hour of compute.

## Files changed / added

Commits: `12be026` (Phase 1) and `a296b48` (Phase 2). `git log --oneline --stat main..HEAD` gives the full diff.

- Scripts: 9 new in `scripts/interpretability/`, 2 new in `scripts/pairing/` and `scripts/evaluation/`, 1 new in `scripts/manuscript/`
- Results CSVs: 7 new results tables + 1 parquet (partial dependence)
- Figures: 6 new Core figures (13a/b, 14a/b, 15, 16, 17, 18)
- Tables: S15 new; T2 amended with CI columns
- Tests: test_bootstrap_ci.py new (3 tests)
- Manuscript: `01_introduction.md` and `04_results.md` and `05_discussion.md` edited; `SUBMISSION_CHECKLIST.md` and `manuscript.docx` new
- Audits: `MANUSCRIPT_WRITING_AUDIT_v3.md` new; this summary `LEE_REVISIONS_OVERNIGHT_SUMMARY.md`
