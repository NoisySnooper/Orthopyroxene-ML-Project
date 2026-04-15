# Optuna Hyperparameter Search Strategy

**Status:** proposed for v9 Phase 2 (NB03 rebuild)
**Author:** NQTa
**Date:** 2026-04-15

---

## Background

The v7/v8 pipeline used `HalvingRandomSearchCV` for hyperparameter optimization. Halving is defensible in principle (allocate more CV budget to survivors of each round) but has a known failure mode: promising hyperparameter regions can be eliminated prematurely when initial trials happen to underperform on the small early-round budgets. For regression problems with noisy CV scores, this produces search paths that are sensitive to the ordering of initial trials.

Modern Bayesian optimization via Optuna's TPE (Tree-structured Parzen Estimator) sampler explores the hyperparameter space more efficiently by modeling the joint distribution of hyperparameter values and validation performance, then drawing new trials from the posterior over high-expected-improvement regions. Published cpx ML thermobarometry work (Ágreda-López 2024, Jorgenson 2022, Wang 2021) has not yet adopted Optuna; moving to it aligns this project with broader ML best practice and gives a cleaner convergence story for the manuscript Methods section.

---

## Approach: TPE with median pruning

For every (model, target, track, feature_set) combination in the search grid:

- 4 models: RF, ERT, XGB, GB
- 2 targets: T_C, P_kbar
- 2 tracks: opx_only, opx_liq
- 3 feature sets: raw, alr, pwlr
- **Total: 4 × 2 × 2 × 3 = 48 Optuna studies**

Each study runs 50 trials under:

- Sampler: `optuna.samplers.TPESampler(multivariate=True, seed=42)`
- Pruner: `optuna.pruners.MedianPruner(n_startup_trials=10)` (kills trials below the running median after the warm-up phase)
- Inner CV: 3-fold `GroupKFold` by Citation (same group structure as outer validation)
- Scoring: neg RMSE (sklearn convention), converted to positive for reporting
- Timeout per trial: 300 s (hard cap to prevent runaway boosting configurations)
- n_jobs per inner CV: 4 (leaves cores for Optuna's sampler overhead on the i7-1265U hybrid CPU)

Search spaces per model type are fully specified in `src/optuna_search.py` (see Phase 2.2 in the rebuild plan for the exact distributions).

Hyperparameter selection is per feature_set. Phase 3.5's existing canonical-winner selection rule then picks the best feature_set per (model, target, track) based on held-out validation RMSE, producing the canonical model set.

---

## Why 50 trials per config

Empirical guidance from the Optuna documentation and hyperparameter-search literature: TPE convergence typically plateaus between 30 and 80 trials for small parameter spaces (5 to 10 hyperparameters). 50 sits at the middle of this range and is also where the compute budget becomes tractable (48 studies × 50 trials × ~7 s per trial ≈ 5 hours on i7-1265U).

The convergence diagnostic figure (`fig_nb03_optuna_search_progress`) verifies this empirically per config. If a given study's best-so-far curve has not plateaued by trial 50, that means either the search space is too wide (rare with our fixed spaces) or the model architecture fits the data poorly at any settings (informative diagnostic on its own). Both outcomes are reportable.

If Phase 2 hits hour 10 of its compute budget and is behind schedule, we can reduce `N_TRIALS` from 50 to 30 per the risk register in the rebuild plan.

---

## Why Optuna over HalvingRandomSearchCV

| Criterion | HalvingRandomSearchCV | Optuna TPE |
|---|---|---|
| Budget allocation | Fixed halving schedule | Adaptive via expected-improvement model |
| Sensitivity to initial trials | High (weak early trials can be eliminated) | Lower (posterior learns from all trials) |
| Parallelism | Embarrassing within a round | Sequential within a study (TPE is stateful) |
| Interpretability of search | Round survivors | Full trial history + param importance |
| Reproducibility | Seeded random search within rounds | Seeded TPE sampler |
| Manuscript precedent in cpx ML | Some | None (novelty modest but real) |

The trade-off: Optuna is less parallel within a single study (TPE needs each trial's result before drawing the next), but we parallelize across the 48 studies trivially. The net compute is comparable; the search quality should be meaningfully better on noisy CV scores.

---

## Tune-once-then-freeze

**Critical discipline:** hyperparameter search runs ONCE, with `seed=42` for the TPE sampler, producing a single set of best params per (model, target, track, feature_set). These frozen best params are then used across all 20 outer random seeds (`SPLIT_SEEDS = range(42, 62)`) for the variance-quantification final training.

This is the same pattern as v7/v8; only the search algorithm changed. The alternative (re-running Optuna per seed) would cost 20x more compute, and the sampler's stochasticity would add variance that obscures the actual train/test-split variance we care about.

The frozen best params are saved to `results/nb03_optuna_best_params.json` immediately after search and are the source of truth for Phase 3.4 (final training) and Phase 3.9 (resampled final training).

---

## Study persistence

Each Optuna study is saved to disk as a joblib pickle under `results/optuna_studies/` (new subdir, created in Phase 2.2):

```
results/optuna_studies/
  study_{model}_{target}_{track}_{feature_set}.joblib
```

Persisting the full study objects (not just the best params) lets us:

1. Plot per-study convergence curves (`fig_nb03_optuna_search_progress`)
2. Compute hyperparameter importance via `optuna.importance.get_param_importances(study)` (`fig_nb03_optuna_hyperparameter_importance`)
3. Audit the search post-hoc if a reviewer asks "what did the sampler try for config X?"

Study files are archived alongside `.joblib` models in `models/` if a future re-analysis needs them; otherwise they live under `results/` and are archived on the next clean slate.

---

## Sanity checks during the run

The Phase 2.5 NB03 cell should log and flag:

1. **Plateau check:** for each study, compute the fractional improvement between trial 25 and trial 50. If improvement is > 5%, the search did not converge; flag the config.
2. **Trial failure rate:** count trials that raised exceptions (OOM, timeout, numerical). If > 20% of trials failed for a config, the search space is misconfigured for that model; investigate before trusting best params.
3. **Pruner kill rate:** log how many trials the MedianPruner killed early. A pruner kill rate near 0% means the pruner did nothing (probably fine); near 100% means most trials were worse than the starting median (search is struggling).
4. **Progress snapshots:** dump partial best_params every 10 studies completed so we don't lose the whole run if something crashes at study 35 of 48.

---

## Expected impact

Primary expected benefit: more stable best_params across minor perturbations of the search (seed, ordering), leading to lower variance in downstream multi-seed evaluation. Direct RMSE improvements vs HalvingRandomSearchCV are modest in typical ML benchmarks (low single-digit % relative improvement).

Secondary benefit: the search-progress and hyperparameter-importance figures give the manuscript a defensible "we searched carefully" story that HalvingRandomSearchCV does not support.

---

## Implementation location

`src/optuna_search.py` (new module, to be written in Phase 2.2).

Exposed function:
- `optuna_search(model_name, X_train, y_train, groups, n_trials=50, seed=42, timeout_per_trial=300)` → returns `{'best_params', 'best_score', 'best_trial', 'study'}` dict

Called from NB03 Phase 3.3b (modified cell) in a 48-iteration loop. Intermediate progress saved every 10 studies to `results/nb03_optuna_best_params_partial.json`, finalized to `results/nb03_optuna_best_params.json` at the end.

---

## References

- Akiba et al. (2019) KDD — Optuna: A Next-generation Hyperparameter Optimization Framework
- Bergstra et al. (2011) NeurIPS — Algorithms for Hyper-Parameter Optimization (TPE original)
- Bergstra & Bengio (2012) JMLR — Random Search for Hyper-Parameter Optimization (baseline comparison)
- Li et al. (2018) JMLR — Hyperband / Successive Halving (the family HalvingRandomSearchCV belongs to)
