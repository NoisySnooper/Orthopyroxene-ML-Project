# Bias-correction results (auto-filled)

*Generated: 2026-04-18 (Phase G.7). Source tables: `results/v10_bias_correction_{per_seed,summary,shipped}.csv`, `results/v10_bias_correction_edge_sensitivity.csv`, `results/v10_bias_correction_form_b_stability.csv`. Figures: `figures/fig30-34_*`.*

## Framing

Phase G.7 quantifies whether a post-hoc bias correction -- fit on citation-grouped out-of-fold residuals of the training split and applied at inference -- reduces test-set error without inflating any of the four pre-registered P regimes. Two forms are tested:

* **Form A**: per-regime ordinary least squares, `y_corr = a_r * y_pred + b_r` for r in {shallow_crustal, deep_crustal_MASH, lithospheric_mantle, deeper_mantle}.
* **Form B**: quantile-thresholded piecewise, `bias(y) = s_L * (y - alpha_L)` if `y < alpha_L`, `0` in the middle, `s_R * (y - alpha_R)` if `y > alpha_R`, with alphas chosen from a quantile grid under a minimum middle-width constraint and a slope cap `|s| <= 2`.

A correction **ships** for a cell only if, on the held-out test set, overall RMSE improves by more than `SHIP_TOL = 1e-6` and no regime RMSE degrades by more than `SHIP_TOL`. When both Form A and Form B ship, the larger overall `Delta`-RMSE wins; ties go to Form A (simpler).

## Per-cell ship decisions (canonical seed = 42)

| track | target | model / feat. | winner | $\Delta$RMSE (mean) | max reg. deg. | ships (maj. of 20) |
|---|---|---|---|---|---|---|
| cpx-liq | P_kbar | LightGBM / pwlr | none | -- | -- | no (0/0) |
| cpx-only | P_kbar | MLP / alr | A | 2.334 | -0.676 | yes (19/20) |
| opx-liq | P_kbar | MLP / raw | none | -- | -- | no (0/0) |
| opx-only | P_kbar | RF / pwlr | A | 4.337 | -4.228 | yes (20/20) |
| cpx-liq | T_C | ERT / pwlr | none | -- | -- | no (0/0) |
| cpx-only | T_C | ERT / pwlr | none | -- | -- | no (0/0) |
| opx-liq | T_C | ElasticNet / raw | none | -- | -- | no (0/0) |
| opx-only | T_C | LightGBM / alr | none | -- | -- | no (0/0) |

## Form A bin-edge sensitivity (+/-1 kbar)

| track | target | $\Delta$RMSE base | max $|\Delta|$ swing |
|---|---|---|---|
| cpx-liq | P_kbar | 0.809 | 0.280 |
| cpx-liq | T_C | 3.549 | 1.409 |
| cpx-only | P_kbar | 2.411 | 0.405 |
| cpx-only | T_C | -12.023 | 5.916 |
| opx-liq | P_kbar | 1.233 | 0.251 |
| opx-liq | T_C | -0.107 | 0.251 |
| opx-only | P_kbar | 4.296 | 0.878 |
| opx-only | T_C | 24.564 | 3.789 |

## Form B parameter stability across 5 CV reseeds

| track | target | alpha_L mean | alpha_L std | alpha_R mean | alpha_R std |
|---|---|---|---|---|---|
| cpx-liq | P_kbar | 0.450 | 0.000 | 0.950 | 0.000 |
| cpx-liq | T_C | 0.450 | 0.000 | 0.806 | 0.060 |
| cpx-only | P_kbar | 0.138 | 0.016 | 0.950 | 0.000 |
| cpx-only | T_C | 0.258 | 0.106 | 0.566 | 0.020 |
| opx-liq | P_kbar | 0.202 | 0.125 | 0.630 | 0.160 |
| opx-liq | T_C | 0.050 | 0.000 | 0.846 | 0.020 |
| opx-only | P_kbar | 0.306 | 0.041 | 0.854 | 0.048 |
| opx-only | T_C | 0.130 | 0.000 | 0.550 | 0.000 |

## Caveats

* Per-regime pre/post rows (non-ALL) feed Table S9.
* `ships (maj. of 20)` counts seeds where, at the winning form, both the overall-improve and no-regime-degrade conditions hold on the held-out test split.
* `GEOROC natural inference` post-correction (cell opx-only) is in `results/v10_natural_opx_post_correction_inference.csv`; regime is assigned from the _predicted_ P (no ground truth on natural samples), following the convention in Agreda-Lopez 2024.

