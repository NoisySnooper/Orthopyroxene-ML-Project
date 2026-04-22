# Submission checklist — opx ML thermobarometer (JGR:ML&C)

This checklist is the single source of truth for what ships with the manuscript at submission. Items added under the Lee-revision overnight pass (2026-04-22) are tagged `[lee-rev]`.

## 1. Manuscript sections (draft)

- `00_abstract.md` — headline opx-only P, 54.7% RMSE reduction; TabPFN inclusion; honest nulls listed
- `00a_front_matter.md`
- `01_introduction.md` — 5 contributions; natural-sample external bias
- `02_data.md` — LEPR + holdouts
- `03_methods.md` — bootstrap CI (§3.10), pairing matrix spec, augmentation protocol
- `04_results.md` — §4.1 aggregate with [CI lo, hi] format; §4.1.1 best-per-cell defense [lee-rev]; §4.5 8-layer interpretability bundle [lee-rev]; §4.7 natural-sample validation [lee-rev]; §4.8 shipped claims summary
- `05_discussion.md` — §5.5 cpx replication; §5.6 "On the black-box concern" [lee-rev]; §5.7 limitations
- `06_conclusions.md`
- `07_data_and_code.md`
- `08_acknowledgments_and_credit.md`
- `09_cover_letter.md`

## 2. Tables

- `T2_model_roster.csv` / `.tex` — 8-cell roster with RMSE CI columns [lee-rev]
- `T4_bias_correction_shipped.tex`
- `S9_bias_correction_per_seed.tex`
- `S10_bias_correction_stability.tex`
- `S15_single_family_sensitivity.csv` / `.tex` [lee-rev] — per-family best-FS RMSE across 8 cells + per-cell winner row

## 3. Core figures

- `Core_13a_fig_method_agreement_T.pdf` / .png [lee-rev] — 10×10 method agreement matrix, T
- `Core_13b_fig_method_agreement_P.pdf` / .png [lee-rev] — 10×10 method agreement matrix, P
- `Core_14a_fig_pairing_panels_T.pdf` / .png [lee-rev] — 20-panel pairwise scatter, T
- `Core_14b_fig_pairing_panels_P.pdf` / .png [lee-rev] — 20-panel pairwise scatter, P
- `Core_15_fig_feature_concordance.pdf` / .png [lee-rev] — per-cell Spearman rho matrix across 8 families
- `Core_16_fig_classical_equivalence.pdf` / .png [lee-rev] — ML vs classical-fit 2×4 grid
- `Core_17_fig_partial_dependence.pdf` / .png [lee-rev] — PD across families, 4×3 grid
- `Core_18_fig_surrogate_trees.pdf` / .png [lee-rev] — 4 surrogate decision trees, max_depth=4

## 4. Results artifacts (machine-readable)

- `results/bootstrap_rmse_cis_all_cells.csv` [lee-rev] — 72+ rows, 8 cells × 9 families × FS, with CI lo/hi
- `results/pairing_matrix_22rows.csv` [lee-rev]
- `results/method_agreement_matrix_T.csv` / `_P.csv` [lee-rev]
- `results/feature_concordance_permutation_importance.csv` [lee-rev]
- `results/feature_concordance_spearman_matrix.csv` [lee-rev]
- `results/classical_feature_ablation.csv` [lee-rev]
- `results/classical_equivalence_regression.csv` [lee-rev]
- `results/surrogate_tree_r_squared.csv` [lee-rev]
- `results/physics_consistency_checks.csv` [lee-rev]
- `results/gasparik_agreement.csv` [lee-rev]
- `results/partial_dependence_across_families.parquet` [lee-rev]
- `results/shap_importance_winners.csv`

## 5. Tests (pytest)

- `tests/test_bootstrap_ci.py` [lee-rev] — bootstrap CI nontrivial; seed-determinism; 8-cell completeness

## 6. Bibliography additions [lee-rev]

The 8-layer interpretability bundle and per-cell best-model defense (§4.1.1, §4.5, §5.6) require the following references. Add to the master bibliography at submission time.

### AutoML / per-task best-model literature
- **Wolpert, D. H. (1996)** — "The lack of a priori distinctions between learning algorithms." *Neural Computation* 8(7):1341–1390. doi:10.1162/neco.1996.8.7.1341. Foundational No Free Lunch theorem cited in §4.1.1 for per-cell winner selection.
- **Feurer, M., Klein, A., Eggensperger, K., Springenberg, J., Blum, M., & Hutter, F. (2015)** — "Efficient and robust automated machine learning." *Advances in Neural Information Processing Systems* 28. Cited in §4.1.1 for auto-sklearn and per-task model selection being standard practice.
- **Erickson, N., Mueller, J., Shirkov, A., Zhang, H., Larroy, P., Li, M., & Smola, A. (2020)** — "AutoGluon-Tabular: Robust and accurate AutoML for structured data." *arXiv:2003.06505*. Cited in §4.1.1 for the AutoML state-of-the-art treating per-task best-model selection as the default.
- **Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022)** — "Why do tree-based models still outperform deep learning on typical tabular data?" *Advances in Neural Information Processing Systems* 35. Cited in §4.1.1 on tabular ML benchmarks showing no single family dominates across datasets.

### Interpretability bundle supporting literature (already in bibliography, re-verify citations)
- Roeder, P. L., & Emslie, R. F. (1970) — Kd equilibrium window, §4.5.6
- Gasparik, T. (1987) — opx Al-solubility barometer, §4.5.6, §4.5.7
- Brey, G. P., & Köhler, T. (1990) — two-pyroxene thermobarometry, §4.5.4
- Putirka, K. D. (2008) — classical calibrations referenced throughout §4.5

## 7. Pre-submission checks (run before upload)

- `pytest tests/` — all green
- `python scripts/evaluation/build_S15_single_family_sensitivity.py` — regenerate S15
- `python scripts/interpretability/compute_feature_concordance.py` — regenerate feature concordance
- `python scripts/interpretability/classical_feature_ablation.py` — regenerate ablation CSV
- `python scripts/interpretability/classical_equivalence_regression.py` — regenerate + Core_16
- `python scripts/interpretability/surrogate_decision_trees.py` — regenerate + Core_18
- `python scripts/interpretability/physics_consistency_checks.py` — regenerate
- `python scripts/interpretability/gasparik_agreement.py` — regenerate
- `python scripts/interpretability/partial_dependence_across_families.py` — regenerate + Core_17
- `python scripts/manuscript/build_manuscript_docx.py` — Word doc for copy-editing
- Terminology sweep: no raw `±` CI notation in final Results/Discussion; all CI should read `[CI lo, hi]` [lee-rev]

## 8. Archive release

- Data: processed parquet splits + LEPR xlsx SHA256 prefix c96ffea4
- Models: `models/canonical/{opx,cpx}/base_*.joblib`
- Reproduction: `scripts/evaluation/run_full_pipeline.sh`
