# Regime-stratified RMSE · scope=opx · bin=P · target=T_C

Cells: `RMSE [bootstrap 95% CI]`. Winner per regime bold.

### opx_liq · T_C (binned by P)

| Regime | n | our_best | Putirka |
| --- | --- | --- | --- |
| shallow_crustal | 47 | **23.47 [15.89, 30.40] (ERT/pwlr)** | 25.52 [16.32, 33.87] (Putirka 28a) |
| deep_crustal_MASH | 61 | **78.46 [29.85, 113.07] (ElasticNet/raw)** | 87.48 [34.61, 120.83] (Putirka 28a) |
| lithospheric_mantle | 58 | 93.70 [77.31, 111.17] (ElasticNet/raw) | **79.65 [54.89, 107.41] (Putirka 28a)** |
| deeper_mantle | 8 | 105.60 [75.14, 127.73] (ElasticNet/pwlr) | **85.50 [14.78, 123.50] (Putirka 28a)** |
| P<5 | 47 | **23.47 [15.89, 30.40] (ERT/pwlr)** | 25.52 [16.32, 33.87] (Putirka 28a) |
| 5<=P<10 | 11 | **27.00 [18.19, 33.91] (ElasticNet/raw)** | 43.05 [25.11, 61.83] (Putirka 28a) |
| 10<=P<20 | 81 | **88.40 [59.90, 114.05] (ElasticNet/raw)** | 97.54 [66.44, 124.69] (Putirka 28a) |
| 20<=P<40 | 32 | 92.14 [75.43, 108.58] (ElasticNet/raw) | **41.76 [25.17, 59.66] (Putirka 28a)** |
| P>=40 | 3 | **20.43 (MLP/raw)** | 138.78 [61.86, 173.83] (Putirka 28a) |
| P<20 (Agreda range) | 139 | **70.95 [50.29, 88.99] (ElasticNet/raw)** | 75.04 [53.39, 96.25] (Putirka 28a) |
| ALL | 174 | 77.06 [62.56, 91.65] (ElasticNet/raw) | **71.67 [53.75, 88.74] (Putirka 28a)** |

### opx_only · T_C (binned by P)

| Regime | n | our_best |
| --- | --- | --- |
| shallow_crustal | 48 | **108.26 [90.90, 122.97] (GB/raw)** |
| deep_crustal_MASH | 76 | **109.47 [86.56, 132.73] (LightGBM/pwlr)** |
| lithospheric_mantle | 37 | **70.43 [57.66, 84.52] (ERT/raw)** |
| deeper_mantle | 29 | **247.61 [167.18, 313.44] (ElasticNet/raw)** |
| P<5 | 48 | **108.26 [90.90, 122.97] (GB/raw)** |
| 5<=P<10 | 20 | **136.73 [95.58, 174.79] (MLP/alr)** |
| 10<=P<20 | 84 | **87.91 [74.08, 105.38] (LightGBM/pwlr)** |
| 20<=P<40 | 23 | **108.01 [69.23, 152.09] (LightGBM/alr)** |
| P>=40 | 15 | **321.18 [205.15, 406.97] (ElasticNet/raw)** |
| P<20 (Agreda range) | 152 | **112.02 [100.43, 124.72] (XGB/raw)** |
| ALL | 190 | **147.98 [122.71, 172.97] (LightGBM/alr)** |
