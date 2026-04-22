# Manuscript Writing Audit v3

Generated 2026-04-22 after Lee-revisions overnight pass (branch
`lee_revisions_overnight`). Supersedes v1 (`MANUSCRIPT_WRITING_AUDIT.md`)
for anything touched in Phases 1 and 2 of the overnight run. Read
alongside v1 for sections not amended here (data inventory, preregistration
state, cpx replication figures).

Tone: caveman. Short sentences. Cite every number.
Repo HEAD `a296b48`, branch `lee_revisions_overnight`.

## Section 1. What changed since v1

Phase 1 (bootstrap CIs + pairing matrix).
- New script `scripts/evaluation/compute_bootstrap_cis_all_cells.py` produces `results/bootstrap_rmse_cis_all_cells.csv` with 72+ rows (8 cells × 9 families × feature sets), n_boot=500, seed=42, on paired test-set residuals.
- `manuscripts/opx_2026/tables/T2_model_roster.{csv,tex}` adds `rmse_ci_lo` and `rmse_ci_hi` columns.
- `scripts/pairing/build_pairing_matrix.py` emits `results/pairing_matrix_22rows.csv` (20 rows: A-V minus K,L).
- Figures `Core_13a/b` (10×10 method agreement matrix, scipy average linkage) and `Core_14a/b` (20-panel pairing scatter).
- Tests `tests/test_bootstrap_ci.py` (3 tests: nontrivial CI for a deterministic fit; seed-determinism; 8-cell completeness). All pass.

Phase 2 (interpretability bundle).
- Per-cell best-model defense: `scripts/evaluation/build_S15_single_family_sensitivity.py` builds `manuscripts/opx_2026/tables/S15_single_family_sensitivity.{csv,tex}` — 9 family rows × 8 cells plus per-cell winner row, each reporting `RMSE [lo, hi] (feature_set)` for that family's best FS.
- 8-layer interpretability bundle scripts in `scripts/interpretability/`:
  - `compute_feature_concordance.py` → `results/feature_concordance_permutation_importance.csv` + `results/feature_concordance_spearman_matrix.csv` + `figures/core/Core_15_fig_feature_concordance.pdf`
  - `classical_feature_ablation.py` → `results/classical_feature_ablation.csv`
  - `classical_equivalence_regression.py` → `results/classical_equivalence_regression.csv` + `figures/core/Core_16_fig_classical_equivalence.pdf`
  - `surrogate_decision_trees.py` → `results/surrogate_tree_r_squared.csv` + `figures/core/Core_18_fig_surrogate_trees.pdf`
  - `physics_consistency_checks.py` → `results/physics_consistency_checks.csv`
  - `gasparik_agreement.py` → `results/gasparik_agreement.csv`
  - `partial_dependence_across_families.py` → `results/partial_dependence_across_families.parquet` + `figures/core/Core_17_fig_partial_dependence.pdf`
  - `make_fig_feature_concordance.py` → Core_15 figure

Manuscript sections amended.
- `00_abstract.md` — wait, abstract not touched this pass (left for author); check v4.
- `01_introduction.md` — §1 contribution (i): ± → [CI lo, hi] for all four opx winner RMSEs; contribution (ii): TabPFN cpx-liq T numbers switched to CI notation.
- `03_methods.md` — §3.10 pairing matrix spec (Phase 1, already shipped).
- `04_results.md` — §4.1.1 new (best-per-cell defense, Wolpert 1996 + Erickson 2020 + Feurer 2015 + Grinsztajn 2022); §4.5 expanded to 8-layer bundle (Layers 1–8, one §4.5.x per layer); §4.7 natural-sample validation (Phase 1); §4.8 shipped claims summary (renumbered from old §4.7).
- `05_discussion.md` — §5.6 rewritten from "Feature-attribution alignment" to "On the black-box concern" citing the 8-layer bundle with bootstrapped numbers.

New files (previously untracked).
- `manuscripts/opx_2026/SUBMISSION_CHECKLIST.md` — single source of truth for what ships at submission, flagged `[lee-rev]` entries for Phase 1/2 artifacts, bibliography additions in §6.

## Section 2. Pre-registration state (unchanged from v1)

See v1 §1. No amendments added this pass. Two existing amendments (`AMENDMENT_1_tiered_veto.md`, `AMENDMENT_2_veto_tolerance.md`) remain untracked in git and are flagged in Methods §3.5; this pass did not resolve the tracking gap — the files exist but have not been committed. Action item for the final commit below.

## Section 3. Numeric anchor table (post-v2)

All numbers below are from `results/bootstrap_rmse_cis_all_cells.csv` seed 42 with 500-bootstrap 95% CIs.

| cell | target | winner | FS | RMSE | CI |
|---|---|---|---|---|---|
| opx_liq | T_C | ElasticNet | raw | 77.06 °C | [61.81, 92.73] |
| opx_liq | P_kbar | MLP | raw | 4.40 kbar | [3.08, 4.54] |
| opx_only | T_C | LightGBM | alr | 146.63 °C | [121.19, 170.00] |
| opx_only | P_kbar | RF | pwlr | 10.35 kbar | [6.90, 13.92] |
| cpx_liq | T_C | TabPFN | raw | 69.96 °C | [55.11, 82.41] |
| cpx_liq | P_kbar | LightGBM | pwlr | 6.55 kbar | [5.50, 7.52] |
| cpx_only | T_C | ERT | pwlr | 127.02 °C | [118.54, 136.86] |
| cpx_only | P_kbar | TabPFN | raw | 13.42 kbar | [11.12, 15.54] |

Natural-sample validation anchor (Row A, pairing_matrix_22rows.csv): our ML opx-liq agrees with our ML cpx-liq on 301 coexisting natural pairs at T RMSE-of-disagreement 46.7 °C [CI 42.0, 51.2] and P RMSE-of-disagreement 3.29 kbar [CI 2.61, 4.19].

Ágreda-López 2024 natural-corpus T bias: Row Q (ML cpx-liq vs Ágreda cpx-liq) = 291.2 °C [CI 286.0, 296.9] disagreement; direction is "Ágreda T below every other method by ~300 °C".

## Section 4. Interpretability bundle anchor table

| layer | probe | metric | cells | value range | passes? |
|---|---|---|---|---|---|
| 1 | SHAP on winner | mean \|SHAP\| | 4 opx (+ cpx) | qualitative | yes, features align with Putirka 28a/29a/29c |
| 2 | Permutation importance concordance | median pairwise Spearman rho | 8 | 0.25–0.75 | yes on T-cells, ambiguous on opx-only |
| 3 | Classical-feature ablation | RMSE delta % | 4 opx | +9 to +58% | diagnostic, not pass/fail |
| 4 | Classical-equivalence regression | R² | 8 | 0.56–0.99 | yes on 5/8 cells at R² > 0.75 |
| 5 | Surrogate decision tree | R² [CI] | 4 opx | 0.81–0.89 | yes, all 4 ≥ 0.81 |
| 6 | Fe-Mg exchange Kd | fraction in [0.23, 0.35] | opx_liq (n=174) | 100% | yes (Roeder-Emslie 1970) |
| 6 | Al-P Spearman | rho | opx_only (n=190) | 0.518 | yes (Gasparik 1987 > 0.5) |
| 7 | Partial dependence | family agreement on shape | 4 opx × 3 features | monotone | yes, crossovers in tail 10% only |
| 7 | Gasparik Al-importance share | fraction in top-10 | opx_liq P, opx_only P | 45.3%, 23.6% | yes on opx_liq P, diagnostic on opx_only P |

## Section 5. Tests

All tests pass at HEAD `a296b48`: 87/87 in `tests/` at 2026-04-22 03:27 UTC.

- `tests/test_bootstrap_ci.py` [new this pass]: 3 tests
- `tests/test_bias_correction.py`: existing
- `tests/test_preregistration.py`: existing
- (full list: `ls tests/`)

## Section 6. Bibliography additions to cite

Add these to the manuscript bibliography. Not yet inserted in any sections file. Bibliographic management is deferred to the author's Zotero pass; this audit flags the insertion list.

- Wolpert, D. H. (1996) — No Free Lunch; §4.1.1
- Feurer et al. (2015) — auto-sklearn; §4.1.1
- Erickson et al. (2020) — AutoGluon; §4.1.1
- Grinsztajn et al. (2022) — tabular ML benchmarks; §4.1.1
- Roeder & Emslie (1970) — Kd window; §4.5.6
- Gasparik (1987) — opx Al-solubility; §4.5.6, §4.5.7

(Putirka 2008 and Brey-Köhler 1990 already cited.)

## Section 7. Outstanding items at end of overnight pass

1. Preregistration amendments 1 and 2 untracked in git (flagged in v1, unchanged).
2. Abstract (`00_abstract.md`) not updated to reflect new [CI lo, hi] format or 8-layer bundle. Author action.
3. `SUBMISSION_CHECKLIST.md` is newly created; at next author pass, review §7 pre-submission check list and decide whether any items need conversion to CI.
4. `09_cover_letter.md` not updated; references "ML models" generically. Author action.
5. `scripts/manuscript/build_manuscript_docx.py` — verified present; docx build is an author-pass action using `pandoc` on the concatenated sections files.

## Section 8. Repo state at end of overnight pass

- Branch: `lee_revisions_overnight`
- HEAD: `a296b48` (Phase 2 commit)
- Phase 1 HEAD: `12be026`
- Remote: no push this pass
- Uncommitted: assorted figures, notebooks, tests that exist but were not part of Phase 1 or Phase 2 scope
