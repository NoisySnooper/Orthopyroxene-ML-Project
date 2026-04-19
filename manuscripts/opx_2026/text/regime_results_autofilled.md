# Regime-stratified results (auto-filled from pre-registered pipeline)

*Generated: 2026-04-18 (Chunk B), revised with two-axis honesty bar 2026-04-18 (Chunk C). Source tables: `tables/S8_5_{1,2,3,4}_regime_*.md` (S8.5.4 is the canonical robust audit). Source figures: `figures/fig24_per_regime_rmse_opx_liq.{pdf,png}`, `figures/fig25_per_regime_residual_violins_opx_liq.{pdf,png}`. Source audits: `results/opx_per_regime_claims_audit.csv` (axis-1 only, Chunk B) and `results/opx_per_regime_claims_audit_robust.csv` (two-axis, Chunk C). Pre-registration doc: `docs/preregistration/p_regime_preregistration.md` (registered 2026-04-17).*

## Framing

This section characterizes calibration-domain performance of the v10 opx-liq thermobarometer across the pre-registered four-bin P partition `[0, 5, 15, 30, 100]` kbar (labels: shallow_crustal, deep_crustal_MASH, lithospheric_mantle, deeper_mantle). It is **not** a per-regime performance ranking. A claim of the form "method A outperforms method B in regime X" must pass a **two-axis honesty bar**:

1. **Axis 1 — test-set sampling noise.** Bootstrap 95% CI on RMSE over paired (y_true, y_pred) rows (500 resamples, seed = SEED_BOOTSTRAP). v10 CI upper bound must lie below Putirka CI lower bound.
2. **Axis 2 — model-fit stochasticity.** Empirical 97.5th percentile of per-seed RMSE across 20 seeds (42..61, Optuna `best_params` fixed; only model internal stochasticity varies) must lie below Putirka CI lower bound. Putirka equations are deterministic, so their axis-2 collapses to a point.
3. **n ≥ P_REGIME_MIN_N_FOR_CLAIMS (= 20).**

Failing axis 1 or 2: "competitive with." Failing n: "insufficient data."

## Headline findings (test split, n = 174 opx-liq samples)

- **Exactly one bin meets the two-axis honesty bar for a directional claim:** in the **shallow crustal** regime (0–5 kbar, n = 47), v10 (ElasticNet / raw) outperforms Putirka 29a for pressure with non-overlapping bootstrap CIs **and** zero seed variance (the ElasticNet cell is deterministic given fixed hyperparameters, so all 20 seeds produce RMSE = 2.66 kbar). v10 RMSE = 2.66 kbar [2.25, 3.13] vs Putirka 29a RMSE = 3.89 kbar [3.24, 4.47]. See Table S8.5.4 (robust) and Figure 24 right panel.
- **All other non-limited bins are "competitive with":** in the deep_crustal_MASH (n = 61) and lithospheric_mantle (n = 58) regimes, v10 and Putirka 95% CIs overlap for both T and P. Deep_crustal_MASH P_kbar uses MLP/raw, whose seed spread [2.01, 3.07] kbar also overlaps Putirka 29a's lower CI — the Chunk B point estimate of 2.08 was on the favorable tail of the seed distribution; the 20-seed mean is 2.54 kbar. This is transparency about training stochasticity, not a downgrade of the verdict.
- **The deeper mantle bin is sample-size-limited:** n = 8 for both T and P; per pre-registration this bin is reported with CIs for transparency but no directional claim is made.
- **Seed-axis transparency note:** the per-regime best cell for shallow_crustal pressure (ElasticNet/raw) is deterministic. Had the same regime been anchored on MLP/raw (a natural alternative choice since MLP/raw is the aggregate test-set winner), the claim would have collapsed — shallow_crustal MLP/raw seed RMSE ranges [2.03, 5.79] kbar across the same 20 seeds. The robust audit guards against this kind of unreported across-seed variance.

## Per-target summary

### Pressure (kbar)

| Regime                | n  | v10 best cell    | v10 RMSE [95% CI]  | Seed RMSE mean [min, max] | Putirka best eq | Putirka RMSE [95% CI] | Robust verdict                         |
| --------------------- | -- | ---------------- | ------------------ | ------------------------- | --------------- | --------------------- | -------------------------------------- |
| shallow_crustal       | 47 | ElasticNet / raw | 2.66 [2.25, 3.13]  | 2.66 [2.66, 2.66] (det.)  | Putirka 29a     | 3.89 [3.24, 4.47]     | **v10 outperforms Putirka (robust)**   |
| deep_crustal_MASH     | 61 | MLP / raw        | 2.08 [1.62, 2.54]  | 2.54 [2.01, 3.07]         | Putirka 29a     | 2.31 [1.72, 2.85]     | competitive with Putirka               |
| lithospheric_mantle   | 58 | CatBoost / raw   | 4.54 [3.49, 5.56]  | 4.57 [4.38, 4.73]         | Putirka 29a     | 4.66 [3.25, 5.96]     | competitive with Putirka               |
| deeper_mantle         | 8  | MLP / alr        | 4.56 [3.19, 5.75]  | 7.06 [3.78, 24.02]        | Putirka 29b     | 8.52 [4.44, 11.14]    | insufficient data (n < 20)             |

### Temperature (°C)

| Regime                | n  | v10 best cell     | v10 RMSE [95% CI]      | Seed RMSE mean [min, max] | Putirka best eq | Putirka RMSE [95% CI]   | Robust verdict           |
| --------------------- | -- | ----------------- | ---------------------- | ------------------------- | --------------- | ----------------------- | ------------------------ |
| shallow_crustal       | 47 | ERT / pwlr        | 23.47 [15.89, 30.40]   | 23.03 [22.19, 23.87]      | Putirka 28a     | 25.52 [16.32, 33.87]    | competitive with Putirka |
| deep_crustal_MASH     | 61 | ElasticNet / raw  | 78.46 [29.85, 113.07]  | 78.46 [78.46, 78.46] (det.)| Putirka 28a     | 87.48 [34.61, 120.83]   | competitive with Putirka |
| lithospheric_mantle   | 58 | ElasticNet / raw  | 93.70 [77.31, 111.17]  | 93.70 [93.70, 93.70] (det.)| Putirka 28a     | 79.65 [54.89, 107.41]   | competitive with Putirka |
| deeper_mantle         | 8  | ElasticNet / pwlr | 105.60 [75.14, 127.73] | 105.60 [105.60, 105.60] (det.) | Putirka 28a | 85.50 [14.78, 123.50] | insufficient data (n < 20) |

## Residual-distribution characterization (Fig 25)

Figure 25 plots per-regime residual distributions (predicted minus observed) for the v10 canonical base models on the opx_liq test split (T: ElasticNet/raw; P: MLP/raw). For both targets, the residual distribution narrows and centers on zero in the shallow_crustal and deep_crustal_MASH bins, consistent with the RMSE-based finding that v10 is most accurate in those regimes. Residual medians drift mildly away from zero in the lithospheric_mantle bin (underprediction of T, modest bias in P) and the deeper_mantle bin is too sparse to characterize.

## Manuscript-ready claim list

1. *"On the opx-liq test split, the v10 thermobarometer outperforms Putirka (2008) for pressure in the shallow crustal regime (0–5 kbar, n = 47; v10 ElasticNet/raw RMSE 2.66 kbar, 95% CI [2.25, 3.13]; Putirka 29a RMSE 3.89 kbar, 95% CI [3.24, 4.47]; non-overlapping bootstrap CIs). The v10 cell is deterministic given fixed hyperparameters (20-seed RMSE spread of zero), so the claim is robust to model-fit stochasticity."* — **meets two-axis honesty bar.**
2. *"Across the deep crustal / MASH and lithospheric mantle regimes, v10 and Putirka are competitive with one another (95% CIs overlap for both T and P) — we make no directional claim in these regimes. In deep_crustal_MASH pressure, the v10 cell is an MLP and its 20-seed RMSE spread is [2.01, 3.07] kbar; the single-seed point estimate (2.08) is on the favorable tail of this spread, which is why we explicitly report 'competitive' rather than 'outperforms.'"* — **meets two-axis honesty bar.**
3. *"The deeper mantle regime (> 30 kbar) is sample-size-limited in our test split (n = 8) and we report its results for transparency without a directional claim. The across-seed MLP/alr RMSE spread [3.78, 24.02] kbar further underscores the instability of this regime at n = 8."* — **meets two-axis honesty bar.**

Any stronger phrasing elsewhere in the manuscript (e.g. "v10 beats Putirka in the deep crust") would fail the honesty bar and must be rewritten.

## Related artifacts

- `docs/preregistration/p_regime_preregistration.md` — bin edges and honesty bar, registered 2026-04-17.
- `results/opx_per_regime_benchmark.csv` — long-format headline per (regime, target).
- `results/opx_per_regime_claims_audit.csv` — axis-1 verdict table (source of Table S8.5.3, Chunk B).
- `results/opx_per_regime_claims_audit_robust.csv` — **two-axis robust verdict table (Chunk C, canonical), source of Table S8.5.4.**
- `results/v10_chunkC_perseed_regime_rmse.csv` — 20-seed x per-regime RMSE for the 7 per-regime-best opx_liq cells.
- `results/v10_chunkC_perseed_predictions.csv` — per-sample per-seed predictions underlying the above.
- `results/v10_chunkC_perseed_aggregate_rmse.csv` — per-seed aggregate RMSE (cross-check vs existing opx_multiseed_summary.csv; confirmed match to 4 decimals).
- `results/v10_regime_allmodels.csv` — every (method_family, method, regime) row for opx_liq + other tracks.
- `results/opx_liq_canonical_residuals_by_regime.csv` — per-sample residuals used in Figure 25.
- `tables/S8_5_{1,2,3}_regime_*.md` — Chunk B supplementary tables (axis-1 only).
- `tables/S8_5_4_regime_claims_audit_robust_opx_liq.md` — Chunk C robust audit (axes 1 + 2). **Canonical.**
- `figures/fig24_per_regime_rmse_opx_liq.{pdf,png}` — pre-registered 4-bin RMSE with CIs.
- `figures/fig25_per_regime_residual_violins_opx_liq.{pdf,png}` — per-regime residual violins.
