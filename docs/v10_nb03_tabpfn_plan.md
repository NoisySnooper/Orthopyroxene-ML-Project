# v10 nb03 supplementary: TabPFN v2 baseline plan

**Status.** Post-hoc supplementary benchmark. Not in the pre-registered model roster; the roster was locked 2026-04-17 and TabPFN is evaluated after the fact.

**Reference.** Hollmann, N., Müller, S., Purucker, L., Krishnakumar, A., Körfer, M., Hoo, S. B., Schirrmeister, R. T., and Hutter, F. (2025). Accurate predictions on small data with a tabular foundation model. *Nature* 637, 319-326. doi:10.1038/s41586-024-08328-6.

## What TabPFN is

TabPFN v2 is a pretrained transformer for small tabular regression and classification. It is an in-context learner: at inference time the full labeled training set is passed as context to a single forward pass, and predictions are produced for each test row without any gradient steps. The model was pretrained on millions of synthetic tabular priors drawn from structural causal models, and the published claim is state-of-the-art accuracy on small tabular regression (n less than 10000) without hyperparameter tuning. Internal ensembling via `n_estimators` composes multiple permutations and feature subsets.

## Why add it

Nature 637 established foundation-model baselines as a standard comparison for small-data tabular regression, and our data set sits squarely in TabPFN's published operating range (n between 600 and 2400 per track). Adding it lets us address, pre-emptively, a likely reviewer question: "Would a 2025-era foundation model have done this without any of your domain engineering?" A principled answer requires running it, not speculating.

## Minimal protocol, and why

1. **Raw features only.** TabPFN autoscales and handles missing values internally. An ALR or PWLR transform is not a meaningful axis for a foundation model whose priors already cover log-ratio and nonlinear compositions of continuous inputs. Running the feature-set sweep would dilute the comparison without a theoretical prior that it should help.

2. **No Optuna tuning.** TabPFN has essentially no user-tunable hyperparameters in the Hollmann protocol. `n_estimators` controls internal ensemble size, which we fix per pipeline (opx=8, cpx=4 for CPU runtime control; see Known Limitations below). No learning rate, no depth, no regularization to sweep.

3. **No stacking.** Stacking TabPFN onto the v10 base-model set would require refitting the Ridge meta-learner; the v10 stacking hyperparameters are pre-registered and modifying them is outside this task. TabPFN's internal ensemble also already captures the heterogeneity that stacking would target.

4. **No SHAP.** Attribution in an in-context learner is an open research problem; permutation importance on context examples or per-feature marginal effects are both misleading for this architecture. We would rather report no attribution than a plausibly wrong one.

5. **5 seeds, not 20.** The internal `n_estimators` ensemble already captures model-stochasticity variance; the remaining variance across runs comes from data-order permutations. 5 seeds is enough to report a seed-stability standard deviation without the 40-60% of wall-clock cost of the full 20-seed protocol. The 20-seed protocol is specifically pre-registered for tuned models to match the Optuna variance envelope; TabPFN has no Optuna variance.

## Constraints on execution

- Device: CPU. User does not have a GPU; `device='cpu'` is set explicitly.
- Per-pipeline ensemble: `n_estimators=8` for opx combinations (n between 600 and 1035), `n_estimators=4` for cpx combinations (n between 2400 and 2900) to keep wall-clock under the task envelope. Deviations logged per-combination.
- Seeds: 42, 43, 44, 45, 46.
- Venv: `.venv-tabpfn/`, separate from the main `.venv/`. The main venv is never touched.
- Checkpoints: `results/v10_tabpfn_checkpoints/{pipeline}_{track}_{target}_s{seed}.pkl`. Script resumes from existing checkpoints; `--force-rerun` clears them.

## Outputs and downstream reach

The 4 output CSVs match the schemas of the pre-registered multiseed and regime tables so no schema conversion is needed anywhere downstream:

- `results/v10_tabpfn_multiseed_results.csv` (40 rows, 8 combos x 5 seeds, superset schema: `test_rmse` alongside `rmse, mae, r2, n_train, n_test`)
- `results/v10_tabpfn_multiseed_summary.csv` (8 rows, aggregates RMSE across 5 seeds)
- `results/v10_tabpfn_regime_rmse.csv` (40 rows, 8 combos x 5 regimes, matches `v10_regime_allmodels.csv` schema)
- `results/v10_tabpfn_predictions.csv` (per-sample predictions on held-out test split, new schema)

These feed into:

- `nb04_v10_benchmark`: additive cell that joins `v10_tabpfn_multiseed_summary.csv` against v10 aggregate-best and writes `results/v10_tabpfn_head_to_head.csv`.
- `nbF_figures`: new `fig35_tabpfn_vs_v10` (registered as `CANONICAL_FIGURES` entry 35).
- `nb09_manuscript_compilation`: new `Table S12` (`tables/S8_12_tabpfn_benchmark.{csv,md,tex}`).
- `manuscripts/opx_2026/text/tabpfn_paragraph.md`: drop-in discussion paragraph.

## Known limitations

- **CPU inference only.** For cpx combinations the wall-clock is sensitive; we drop `n_estimators` to 4 to compensate. If future runs have GPU access, `n_estimators=8` with `device='cuda'` should be preferred for both pipelines.
- **TabPFN version scope.** This baseline uses TabPFN v2 (Hollmann et al. 2025, Nature 637, doi:10.1038/s41586-024-08328-6), not the more recent TabPFN-2.5 release (Grinsztajn et al. 2025, arXiv:2511.08667). The Nature v2 release is explicitly citable, has a permissive Apache 2.0 license with attribution, and is the model referenced in methods comparisons across the tabular-foundation-model literature as of Q2 2026. TabPFN-2.5 offers larger context support (50,000 samples vs 10,000) and marginal performance improvements, but is distributed under a more restrictive non-commercial license and is not yet peer-reviewed. Future work may evaluate TabPFN-2.5 on larger cpx databases where the v2 sample-size ceiling becomes binding; in the current opx/cpx training sets (n <= 2,385) this is not a concern. Pip pin `tabpfn>=2.0,<2.5` enforces this at install time, and the baseline script asserts the package version at the first fit.
- **No regime-stratified scorecard entry.** TabPFN is not in `V10_BASE_ORDER` or the pre-registered scorecard; the post-correction scorecard is a head-to-head of the v10 tuned pipeline vs Putirka, Agreda-Lopez, and Jorgenson baselines, and mixing in a foundation model would change what that scorecard tests (domain tuning, not raw predictive power).
- **Not included in `V10_BASE_ORDER`.** The pre-registered roster was locked 2026-04-17; amending it would violate the registration discipline. TabPFN's role here is explicitly supplementary.

## What ships

If TabPFN beats v10 on aggregate RMSE in at least one cell, we report the head-to-head table in Section 5 Discussion and note which cells tip each way. If v10 is at least tied on all cells, we still report the comparison -- a negative result for foundation models on domain-specific small data is independently informative. Either way, no primary-model decisions change: the pre-registered pipeline and scorecard remain the paper's main finding.
