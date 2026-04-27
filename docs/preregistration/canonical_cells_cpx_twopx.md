# Canonical cell roster — cpx and twopx (Phase H.0b lock)

Lock date: 2026-04-26 UTC
Source: results/cpx_multiseed_summary.csv (cpx, 100 (model, feature_set, track, target) rows × 20 seeds)
Source: results/v10_twopx_multiseed_summary.csv (twopx, 64 rows × 20 seeds)

This file locks the canonical (model_family, feature_set) winner for every (track, target) cell on the cpx and twopx pipelines, mirroring the Phase G Chunk B canonical-cell selection that was performed for opx_liq. It is required by Phase H.0b before any natural-sample inference run that uses cpx or twopx canonical models.

## Selection rule

For each (track, target) cell, the winner is the (model_family, feature_set) combination with minimum mean test-set RMSE across the 20 seeds in the canonical multiseed run. Ties broken by lower std. This is the same rule used for opx_liq in Phase G.

## Canonical cells: cpx pipeline

| track | target | winner family | feature set | mean RMSE | std | n_seeds | source row |
|---|---|---|---|---|---|---|---|
| cpx_liq  | T_C    | TabPFN     | raw  | 69.960 °C   | 0.982 | 20 | results/cpx_multiseed_summary.csv |
| cpx_liq  | P_kbar | LightGBM   | pwlr | 6.546 kbar  | 0.050 | 20 | results/cpx_multiseed_summary.csv |
| cpx_only | T_C    | ERT        | pwlr | 127.022 °C  | 0.305 | 20 | results/cpx_multiseed_summary.csv |
| cpx_only | P_kbar | TabPFN     | raw  | 13.419 kbar | 0.418 | 20 | results/cpx_multiseed_summary.csv |

## Canonical cells: twopx pipeline

| track | target | winner family | feature set | mean RMSE | std | n_seeds | source row |
|---|---|---|---|---|---|---|---|
| twopx | T_C    | ElasticNet | raw | 79.536 °C  | 0.000 | 20 | results/v10_twopx_multiseed_summary.csv |
| twopx | P_kbar | XGB        | alr | 4.261 kbar | 0.029 | 20 | results/v10_twopx_multiseed_summary.csv |

ElasticNet std = 0 because the regularized linear estimator is deterministic; per-seed variance comes only from train-side stochasticity (citation-grouped fold reshuffling), and ElasticNet's coordinate descent converges to the same solution when the fold geometry is preserved. Treat the std=0 row as a numerical artifact of the deterministic estimator, not a data-quality flag.

## Pre-registration boundary enforcement

Per docs/natural_worldwide_plan.md Section 0 collisions 3-4:

- These cells lock the model identity for natural-sample inference (H.3b/c) and curated-locality validation (H.6a). They do not enable per-regime numeric RMSE claims on natural samples — that boundary stays scoped to the calibration domain (collision 1).
- The 20-seed dual-robustness honesty bar (test-set bootstrap CI + per-seed spread) only applies to opx_liq from Phase G. Cpx and twopx natural-sample claims are descriptive with bootstrap CI only; per-seed-spread axis is not claimed for cpx or twopx unless the optional H.0c Chunk-C replication is run later.

## Companion artifacts

- Tuned parameter JSONs:
  - results/optuna_best_params_cpx.json
  - results/optuna_best_params_twopx.json
- Phase G opx_liq lock (already canonical, not duplicated here):
  - docs/preregistration/p_regime_preregistration.md (regime cuts)
  - docs/preregistration/canonical_cells_opx_liq.md (if present; otherwise see results/opx_multiseed_summary.csv winners)

This file is part of the pre-registration set and is not edited after the lock date except to record the optional Chunk-C extension (H.0c) outcome if that work is later performed.
