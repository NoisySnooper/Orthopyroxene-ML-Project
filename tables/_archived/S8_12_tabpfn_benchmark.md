# Table S12. TabPFN v2 baseline comparison

| pipeline   | track    | target   | our_best       | our_best_model   | tabpfn         | external_best   | external_method   | verdict     |
|:-----------|:---------|:---------|:---------------|:-----------------|:---------------|:----------------|:------------------|:------------|
| cpx        | cpx_liq  | P_kbar   | 6.55 +/- 0.05  | LightGBM         | 7.51 +/- 0.13  | --              |                   | our_method_better    |
| cpx        | cpx_liq  | T_C      | 72.5 +/- 0.2   | ERT              | 70.1 +/- 1.0   | --              |                   | tabpfn_better |
| cpx        | cpx_only | P_kbar   | 13.66 +/- 0.28 | MLP              | 13.41 +/- 0.58 | --              |                   | competitive |
| cpx        | cpx_only | T_C      | 127.0 +/- 0.3  | ERT              | 130.6 +/- 3.0  | --              |                   | our_method_better    |
| opx        | opx_liq  | P_kbar   | 4.40 +/- 0.46  | MLP              | 5.46 +/- 0.14  | --              |                   | our_method_better    |
| opx        | opx_liq  | T_C      | 77.1           | ElasticNet       | 84.5 +/- 0.8   | --              |                   | our_method_better    |
| opx        | opx_only | P_kbar   | 10.35 +/- 0.04 | RF               | 12.52 +/- 0.16 | --              |                   | our_method_better    |
| opx        | opx_only | T_C      | 146.6 +/- 0.6  | LightGBM         | 150.6 +/- 1.5  | --              |                   | our_method_better    |

*TabPFN fit with default hyperparameters (n_estimators=8 opx / 4 cpx, device=cpu, 5 seeds). Reference: Hollmann et al. 2025, Nature 637.*
