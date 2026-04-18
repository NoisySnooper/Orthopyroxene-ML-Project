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

### universal · P_kbar (binned by T)

| Regime | n | v10 best |
| --- | --- | --- |
| T<800 | 7 | **27.77 [22.29, 34.25] (ERT/universal_raw)** |
| 800<=T<1000 | 97 | **7.79 [5.53, 9.61] (RF/universal_raw)** |
| 1000<=T<1200 | 443 | **5.58 [4.93, 6.22] (RF/universal_raw)** |
| 1200<=T<1400 | 317 | **7.51 [6.31, 8.73] (ERT/universal_raw)** |
| T>=1400 | 141 | **34.47 [26.23, 42.76] (MLP/universal_raw)** |
| ALL | 1005 | **14.64 [11.92, 17.54] (MLP/universal_raw)** |
