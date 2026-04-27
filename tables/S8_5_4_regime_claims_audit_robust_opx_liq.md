# Table S8.5.4 . Per-regime claims audit, two-axis honesty bar (opx_liq)

Axis 1 (test-set sampling noise): bootstrap 95% CI on RMSE over paired (y_true, y_pred), 500 resamples, seed = SEED_BOOTSTRAP. Axis 2 (model-fit stochasticity): 20-seed refit of the same cell at seeds 42..61, fixed Citation-grouped train/test split. Putirka equations have no seed axis (deterministic closed-form).

Verdict rule: "outperforms" requires BOTH our_rmse_hi < putirka_rmse_lo (axis 1) AND seed_rmse_hi < putirka_rmse_lo (axis 2) AND n >= 20.

| Regime              | Target   |   n | our cell        | our RMSE [95% CI]      | Seed RMSE mean [min, max]   | Putirka eq   | Putirka RMSE [95% CI]   | Axis-1 (residual) non-overlap   | Axis-2 (seed) non-overlap   | Robust verdict                   |
|:--------------------|:---------|----:|:----------------|:-----------------------|:----------------------------|:-------------|:------------------------|:--------------------------------|:----------------------------|:---------------------------------|
| shallow_crustal     | T_C      |  47 | ERT/pwlr        | 23.47 [15.89, 30.40]   | 23.03 [22.19, 23.87]        | Putirka 28a  | 25.52 [16.32, 33.87]    | False                           | False                       | competitive with Putirka         |
| shallow_crustal     | P_kbar   |  47 | ElasticNet/raw  | 2.66 [2.25, 3.13]      | 2.66 [2.66, 2.66]           | Putirka 29a  | 3.89 [3.24, 4.47]       | True                            | True                        | our method outperforms Putirka (robust) |
| deep_crustal_MASH   | T_C      |  61 | ElasticNet/raw  | 78.46 [29.85, 113.07]  | 78.46 [78.46, 78.46]        | Putirka 28a  | 87.48 [34.61, 120.83]   | False                           | False                       | competitive with Putirka         |
| deep_crustal_MASH   | P_kbar   |  61 | MLP/raw         | 2.08 [1.62, 2.54]      | 2.54 [2.01, 3.07]           | Putirka 29a  | 2.31 [1.72, 2.85]       | False                           | False                       | competitive with Putirka         |
| lithospheric_mantle | T_C      |  58 | ElasticNet/raw  | 93.70 [77.31, 111.17]  | 93.70 [93.70, 93.70]        | Putirka 28a  | 79.65 [54.89, 107.41]   | False                           | False                       | competitive with Putirka         |
| lithospheric_mantle | P_kbar   |  58 | CatBoost/raw    | 4.54 [3.49, 5.56]      | 4.57 [4.38, 4.74]           | Putirka 29a  | 4.66 [3.25, 5.96]       | False                           | False                       | competitive with Putirka         |
| deeper_mantle       | T_C      |   8 | ElasticNet/pwlr | 105.60 [75.14, 127.73] | 105.60 [105.60, 105.60]     | Putirka 28a  | 85.50 [14.78, 123.50]   | False                           | False                       | insufficient data (n < 20)       |
| deeper_mantle       | P_kbar   |   8 | MLP/alr         | 4.56 [3.19, 5.75]      | 7.06 [3.78, 24.02]          | Putirka 29b  | 8.52 [4.44, 11.14]      | False                           | False                       | insufficient data (n < 20)       |
