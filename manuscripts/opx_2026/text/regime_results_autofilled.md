# Regime-stratified results (auto-filled from pre-registered pipeline)

*Generated: 2026-04-18. Source tables: `tables/S8_5_{1,2,3}_regime_*.md`. Source figures: `figures/fig24_per_regime_rmse_opx_liq.{pdf,png}`, `figures/fig25_per_regime_residual_violins_opx_liq.{pdf,png}`. Source audit: `results/v10_opx_per_regime_claims_audit.csv`. Pre-registration doc: `docs/v10_p_regime_preregistration.md` (registered 2026-04-17).*

## Framing

This section characterizes calibration-domain performance of the v10 opx-liq thermobarometer across the pre-registered four-bin P partition `[0, 5, 15, 30, 100]` kbar (labels: shallow_crustal, deep_crustal_MASH, lithospheric_mantle, deeper_mantle). It is **not** a per-regime performance ranking. Per pre-registration, a claim of the form "method A outperforms method B in regime X" requires BOTH (i) non-overlapping bootstrap 95% CIs AND (ii) n ≥ 20 in the regime. Otherwise, results are reported as "competitive with" or flagged as "insufficient data."

## Headline findings (test split, n = 174 opx-liq samples)

- **Only one bin meets the pre-registered honesty bar for a directional claim:** in the **shallow crustal** regime (0–5 kbar, n = 47), v10 outperforms Putirka 2008 for pressure with non-overlapping 95% CIs (v10 RMSE = 2.66 kbar [2.25, 3.13] vs Putirka 29a RMSE = 3.89 kbar [3.24, 4.47]). See Table S8.5.3 and Figure 24 right panel.
- **All other non-limited bins are "competitive with":** in the deep_crustal_MASH (n = 61) and lithospheric_mantle (n = 58) regimes, v10 and Putirka 95% CIs overlap for both T and P. These are characterizations of calibration-domain capability, not directional claims.
- **The deeper mantle bin is sample-size-limited:** n = 8 for both T and P; per pre-registration this bin is reported with CIs for transparency but no directional claim is made.

## Per-target summary

### Pressure (kbar)

| Regime                | n  | v10 best cell         | v10 RMSE [95% CI]    | Putirka best eq  | Putirka RMSE [95% CI] | Verdict                                |
| --------------------- | -- | --------------------- | -------------------- | ---------------- | --------------------- | -------------------------------------- |
| shallow_crustal       | 47 | ElasticNet / raw      | 2.66 [2.25, 3.13]    | Putirka 29a      | 3.89 [3.24, 4.47]     | **v10 outperforms Putirka**            |
| deep_crustal_MASH     | 61 | MLP / raw             | 2.08 [1.62, 2.54]    | Putirka 29a      | 2.31 [1.72, 2.85]     | competitive with Putirka (CIs overlap) |
| lithospheric_mantle   | 58 | CatBoost / raw        | 4.54 [3.49, 5.56]    | Putirka 29a      | 4.66 [3.25, 5.96]     | competitive with Putirka (CIs overlap) |
| deeper_mantle         | 8  | MLP / alr             | 4.56 [3.19, 5.75]    | Putirka 29b      | 8.52 [4.44, 11.14]    | insufficient data (n < 20)             |

### Temperature (°C)

| Regime                | n  | v10 best cell         | v10 RMSE [95% CI]       | Putirka best eq  | Putirka RMSE [95% CI]   | Verdict                                |
| --------------------- | -- | --------------------- | ----------------------- | ---------------- | ----------------------- | -------------------------------------- |
| shallow_crustal       | 47 | ERT / pwlr            | 23.47 [15.89, 30.40]    | Putirka 28a      | 25.52 [16.32, 33.87]    | competitive with Putirka (CIs overlap) |
| deep_crustal_MASH     | 61 | ElasticNet / raw      | 78.46 [29.85, 113.07]   | Putirka 28a      | 87.48 [34.61, 120.83]   | competitive with Putirka (CIs overlap) |
| lithospheric_mantle   | 58 | ElasticNet / raw      | 93.70 [77.31, 111.17]   | Putirka 28a      | 79.65 [54.89, 107.41]   | competitive with Putirka (CIs overlap) |
| deeper_mantle         | 8  | ElasticNet / pwlr     | 105.60 [75.14, 127.73]  | Putirka 28a      | 85.50 [14.78, 123.50]   | insufficient data (n < 20)             |

## Residual-distribution characterization (Fig 25)

Figure 25 plots per-regime residual distributions (predicted minus observed) for the v10 canonical base models on the opx_liq test split (T: ElasticNet/raw; P: MLP/raw). For both targets, the residual distribution narrows and centers on zero in the shallow_crustal and deep_crustal_MASH bins, consistent with the RMSE-based finding that v10 is most accurate in those regimes. Residual medians drift mildly away from zero in the lithospheric_mantle bin (underprediction of T, modest bias in P) and the deeper_mantle bin is too sparse to characterize.

## Manuscript-ready claim list

1. *"On the opx-liq test split, the v10 thermobarometer outperforms Putirka (2008) for pressure in the shallow crustal regime (0–5 kbar, n = 47; v10 RMSE 2.66 kbar, 95% CI [2.25, 3.13]; Putirka 29a RMSE 3.89 kbar, 95% CI [3.24, 4.47]; non-overlapping CIs)."* — **meets honesty bar.**
2. *"Across the deep crustal / MASH and lithospheric mantle regimes, v10 and Putirka are competitive with one another (95% CIs overlap for both T and P) — we make no directional claim in these regimes."* — **meets honesty bar.**
3. *"The deeper mantle regime (> 30 kbar) is sample-size-limited in our test split (n = 8) and we report its results for transparency without a directional claim."* — **meets honesty bar.**

Any stronger phrasing elsewhere in the manuscript (e.g. "v10 beats Putirka in the deep crust") would fail the honesty bar and must be rewritten.

## Related artifacts

- `docs/v10_p_regime_preregistration.md` — bin edges and honesty bar, registered 2026-04-17.
- `results/v10_opx_per_regime_benchmark.csv` — long-format headline per (regime, target).
- `results/v10_opx_per_regime_claims_audit.csv` — verdict table (source of Table S8.5.3).
- `results/v10_regime_allmodels.csv` — every (method_family, method, regime) row for opx_liq + other tracks.
- `results/v10_opx_liq_canonical_residuals_by_regime.csv` — per-sample residuals used in Figure 25.
- `tables/S8_5_{1,2,3}_regime_*.md` — supplementary tables.
- `figures/fig24_per_regime_rmse_opx_liq.{pdf,png}` — pre-registered 4-bin RMSE with CIs.
- `figures/fig25_per_regime_residual_violins_opx_liq.{pdf,png}` — per-regime residual violins.
