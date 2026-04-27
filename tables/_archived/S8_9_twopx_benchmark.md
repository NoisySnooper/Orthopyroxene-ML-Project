# Table S8.9: two-pyroxene (twopx) benchmark

Best base model and best stacked ensemble per target on the twopx Citation-grouped held-out test split. `seed_rmse_{mean,std,min,max}` are the 20-seed spread from 20-seed multi-seed refit. The Putirka two-pyroxene equations (eq36/37/38/39) are DEFERRED because the call requires paired opx-cpx Thermobar invocation; the row is retained with NaN so downstream consumers flag the gap.

| Target | Method class | Model / feature set | Single-seed RMSE | Seed mean [min, max] | Notes |
|---|---|---|---|---|---|
| T_C | our_best_base | ElasticNet/raw | 79.54 °C | 79.54 [79.54, 79.54] °C | 20-seed spread from opx_multiseed_summary.csv |
| T_C | our_best_ensemble | greedy/pwlr | 81.04 °C | — | stacked ensemble, single fit per training pipeline |
| T_C | putirka_twopx | Putirka 2008 eq36/37/38/39/n/a | deferred | — | DEFERRED: requires paired opx-cpx Thermobar call, not executed in this study |
| P_kbar | our_best_base | XGB/alr | 4.26 kbar | 4.26 [4.22, 4.32] kbar | 20-seed spread from opx_multiseed_summary.csv |
| P_kbar | our_best_ensemble | greedy/pwlr | 4.51 kbar | — | stacked ensemble, single fit per training pipeline |
| P_kbar | putirka_twopx | Putirka 2008 eq36/37/38/39/n/a | deferred | — | DEFERRED: requires paired opx-cpx Thermobar call, not executed in this study |