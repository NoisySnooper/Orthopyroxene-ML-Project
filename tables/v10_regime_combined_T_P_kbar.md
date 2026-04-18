# Regime-stratified RMSE · scope=combined · bin=T · target=P_kbar

Cells: `RMSE [bootstrap 95% CI]`. Winner per regime bold.

### twopx · P_kbar (binned by T)

| Regime | n | v10 best |
| --- | --- | --- |
| 800<=T<1000 | 1 | **0.86 (MLP/twopx_components)** |
| 1000<=T<1200 | 30 | **3.56 [2.34, 4.83] (CatBoost/pwlr)** |
| 1200<=T<1400 | 15 | **2.47 [1.50, 3.31] (MLP/twopx_components)** |
| T>=1400 | 2 | **8.94 (MLP/raw)** |
| ALL | 48 | **4.27 [3.00, 5.30] (XGB/alr)** |
