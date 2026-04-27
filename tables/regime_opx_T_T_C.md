# Regime-stratified RMSE · scope=opx · bin=T · target=T_C

Cells: `RMSE [bootstrap 95% CI]`. Winner per regime bold.

### opx_liq · T_C (binned by T)

| Regime | n | our_best | Putirka |
| --- | --- | --- | --- |
| 800<=T<1000 | 4 | **188.99 [57.44, 261.03] (MLP/pwlr)** | 259.23 [86.77, 356.19] (Putirka 28a) |
| 1000<=T<1200 | 31 | **89.57 [48.44, 124.07] (ElasticNet/alr)** | 98.28 [57.51, 132.27] (Putirka 28a) |
| 1200<=T<1400 | 109 | 52.06 [41.85, 63.28] (ElasticNet/raw) | **39.40 [29.94, 48.07] (Putirka 28a)** |
| T>=1400 | 30 | 88.20 [73.85, 103.35] (ElasticNet/raw) | **61.08 [34.03, 82.71] (Putirka 28a)** |
| ALL | 174 | 77.06 [62.56, 91.65] (ElasticNet/raw) | **71.67 [53.75, 88.74] (Putirka 28a)** |

### opx_only · T_C (binned by T)

| Regime | n | our_best |
| --- | --- | --- |
| 800<=T<1000 | 22 | **165.56 [128.60, 203.92] (GB/raw)** |
| 1000<=T<1200 | 78 | **107.34 [92.17, 121.21] (XGB/raw)** |
| 1200<=T<1400 | 63 | **71.77 [62.62, 80.88] (ElasticNet/alr)** |
| T>=1400 | 27 | **249.94 [168.45, 323.85] (ElasticNet/raw)** |
| ALL | 190 | **147.98 [122.71, 172.97] (LightGBM/alr)** |
