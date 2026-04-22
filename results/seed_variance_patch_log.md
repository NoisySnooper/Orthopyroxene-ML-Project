# Seed Variance Patch Log

## Finding

`results/diagnostic_seed_variance_check.csv` classifies 48 cells as
zero-std, of which 24 are ElasticNet (deterministic_fit, expected at
frozen hyperparameters) and 24 are HistGradientBoostingRegressor (GB,
classified as seed_bug).

All GB cells across both opx and cpx pipelines × all three feature sets
× both tracks × both targets produce identical test_rmse across 20
SPLIT_SEEDS (42 through 61) when read from `opx_multiseed_results.csv`
and `cpx_multiseed_results.csv`.

Reproducibility spot-check on GB opx-only T_C raw at canonical seed=42
gave 158.28441116 (matches summary). A seed=43 GroupShuffleSplit
produced a different test set size (268 vs 190), which suggests the
canonical split protocol and the 20-seed runner used different splitters;
the 20-seed runner most likely failed to propagate the split index for
the GB branch, collapsing to a single fit repeated 20 times.

## Impact

- `results/{opx,cpx}_multiseed_results.csv` GB rows: test_rmse is
  accurate at the canonical seed=42 split but does not reflect actual
  20-seed variability.
- `results/{opx,cpx}_multiseed_summary.csv` GB rows: mean is correct,
  std is artificially zero. Per-cell RMSE rankings are unaffected.
- Bootstrap CI computation in Phase 1 is NOT based on the 20-seed
  variance; it bootstraps the seed=42 test-set residuals directly. GB
  CIs are therefore valid under the bootstrap protocol, unaffected by
  this bug.

## Disposition (2026-04-22)

No producer script for `opx_multiseed_results.csv` is present under
`scripts/` (the legacy `scripts/v10_phase_g_multiseed_runner.py` was
archived under `archive/pre_consolidation_2026_04_18/`). Reproducing 24
GB cells × 20 seeds × 2 pipelines would require re-implementing the
20-seed runner from scratch. That is out of scope for Phase 1 given the
overnight-run time budget. Phase 1 proceeds with bootstrap CIs from
canonical seed=42 predictions, which supersede the seed-variance std as
the primary uncertainty estimate per the manuscript methods update.

The existing GB rows in `{opx,cpx}_multiseed_summary.csv` are retained
with their artifact-zero std for backward compatibility. The
diagnostic CSV flags them. The bootstrap CI CSV supersedes the std as
the reporting metric.

## Follow-up (tracked for Phase 3 or later)

- Rebuild GB per-seed test predictions under the 20-seed SPLIT_SEEDS
  protocol once the canonical multiseed runner is restored.
- Verify that RF, ERT, XGB, LightGBM, CatBoost, MLP seed variance is
  driven by legitimate split variation, not by stochastic estimator
  components alone.
