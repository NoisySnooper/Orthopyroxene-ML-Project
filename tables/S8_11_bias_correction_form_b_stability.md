# Table S8.11: Form B parameter stability across 5 CV reseeds

Form B (quantile-thresholded piecewise) fit parameters for each of the 8 Phase-G.7 cells at canonical model seed 42 under five alternative StratifiedGroupKFold split seeds. `Form B OK` is True when the fit passed all validity checks (slope cap, minimum middle-width, alpha ordering).

| pipeline | track | target | model | feat. | model seed | CV seed | alpha_L | alpha_R | s_L | s_R | OOF MSE | Form B OK |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| opx | opx-liq | T_C | ElasticNet | raw | 42 | 42 | 0.050 | 0.830 | 0.409 | -0.177 | 4406.70 | True |
| opx | opx-liq | T_C | ElasticNet | raw | 42 | 43 | 0.050 | 0.870 | 0.410 | -0.223 | 4385.55 | True |
| opx | opx-liq | T_C | ElasticNet | raw | 42 | 44 | 0.050 | 0.870 | 0.526 | -0.242 | 4409.25 | True |
| opx | opx-liq | T_C | ElasticNet | raw | 42 | 45 | 0.050 | 0.830 | 0.636 | -0.195 | 4455.77 | True |
| opx | opx-liq | T_C | ElasticNet | raw | 42 | 46 | 0.050 | 0.830 | 0.395 | -0.182 | 4280.08 | True |
| opx | opx-liq | P_kbar | MLP | raw | 42 | 42 | 0.130 | 0.550 | 0.549 | 0.106 | 27.18 | True |
| opx | opx-liq | P_kbar | MLP | raw | 42 | 43 | 0.130 | 0.550 | 0.575 | 0.184 | 33.18 | True |
| opx | opx-liq | P_kbar | MLP | raw | 42 | 44 | 0.450 | 0.950 | 0.192 | -0.507 | 24.85 | True |
| opx | opx-liq | P_kbar | MLP | raw | 42 | 45 | 0.130 | 0.550 | 0.461 | 0.091 | 26.44 | True |
| opx | opx-liq | P_kbar | MLP | raw | 42 | 46 | 0.170 | 0.550 | 0.316 | 0.113 | 27.06 | True |
| opx | opx-only | T_C | LightGBM | alr | 42 | 42 | 0.130 | 0.550 | -0.085 | 0.125 | 17807.05 | True |
| opx | opx-only | T_C | LightGBM | alr | 42 | 43 | 0.130 | 0.550 | -0.094 | 0.120 | 18151.45 | True |
| opx | opx-only | T_C | LightGBM | alr | 42 | 44 | 0.130 | 0.550 | -0.119 | 0.099 | 18168.32 | True |
| opx | opx-only | T_C | LightGBM | alr | 42 | 45 | 0.130 | 0.550 | -0.096 | 0.134 | 17969.56 | True |
| opx | opx-only | T_C | LightGBM | alr | 42 | 46 | 0.130 | 0.550 | -0.104 | 0.133 | 18093.17 | True |
| opx | opx-only | P_kbar | RF | pwlr | 42 | 42 | 0.290 | 0.910 | -0.157 | 0.462 | 114.15 | True |
| opx | opx-only | P_kbar | RF | pwlr | 42 | 43 | 0.370 | 0.910 | -0.138 | 0.629 | 116.00 | True |
| opx | opx-only | P_kbar | RF | pwlr | 42 | 44 | 0.290 | 0.830 | -0.105 | 0.649 | 122.10 | True |
| opx | opx-only | P_kbar | RF | pwlr | 42 | 45 | 0.250 | 0.830 | -0.146 | 0.677 | 121.42 | True |
| opx | opx-only | P_kbar | RF | pwlr | 42 | 46 | 0.330 | 0.790 | -0.084 | 0.635 | 124.93 | True |
| cpx | cpx-liq | T_C | ERT | pwlr | 42 | 42 | 0.450 | 0.750 | -0.080 | -0.073 | 6097.99 | True |
| cpx | cpx-liq | T_C | ERT | pwlr | 42 | 43 | 0.450 | 0.790 | -0.060 | -0.170 | 5854.16 | True |
| cpx | cpx-liq | T_C | ERT | pwlr | 42 | 44 | 0.450 | 0.750 | -0.081 | -0.120 | 6046.89 | True |
| cpx | cpx-liq | T_C | ERT | pwlr | 42 | 45 | 0.450 | 0.910 | -0.057 | -0.239 | 6037.47 | True |
| cpx | cpx-liq | T_C | ERT | pwlr | 42 | 46 | 0.450 | 0.830 | -0.081 | -0.143 | 5836.30 | True |
| cpx | cpx-liq | P_kbar | LightGBM | pwlr | 42 | 42 | 0.450 | 0.950 | -0.074 | 0.178 | 70.96 | True |
| cpx | cpx-liq | P_kbar | LightGBM | pwlr | 42 | 43 | 0.450 | 0.950 | -0.049 | 0.257 | 71.11 | True |
| cpx | cpx-liq | P_kbar | LightGBM | pwlr | 42 | 44 | 0.450 | 0.950 | -0.056 | 0.190 | 69.79 | True |
| cpx | cpx-liq | P_kbar | LightGBM | pwlr | 42 | 45 | 0.450 | 0.950 | -0.070 | 0.034 | 64.06 | True |
| cpx | cpx-liq | P_kbar | LightGBM | pwlr | 42 | 46 | 0.450 | 0.950 | -0.069 | 0.270 | 76.19 | True |
| cpx | cpx-only | T_C | ERT | pwlr | 42 | 42 | 0.290 | 0.550 | 0.064 | 0.406 | 24979.61 | True |
| cpx | cpx-only | T_C | ERT | pwlr | 42 | 43 | 0.330 | 0.590 | 0.049 | 0.444 | 24742.11 | True |
| cpx | cpx-only | T_C | ERT | pwlr | 42 | 44 | 0.050 | 0.550 | -0.168 | 0.342 | 24288.52 | True |
| cpx | cpx-only | T_C | ERT | pwlr | 42 | 45 | 0.290 | 0.550 | 0.058 | 0.409 | 24857.99 | True |
| cpx | cpx-only | T_C | ERT | pwlr | 42 | 46 | 0.330 | 0.590 | 0.039 | 0.490 | 25159.42 | True |
| cpx | cpx-only | P_kbar | MLP | alr | 42 | 42 | 0.130 | 0.950 | 1.679 | 0.286 | 285.71 | True |
| cpx | cpx-only | P_kbar | MLP | alr | 42 | 43 | 0.170 | 0.950 | 1.009 | 0.299 | 289.24 | True |
| cpx | cpx-only | P_kbar | MLP | alr | 42 | 44 | 0.130 | 0.950 | 1.405 | 0.206 | 290.80 | True |
| cpx | cpx-only | P_kbar | MLP | alr | 42 | 45 | 0.130 | 0.950 | 1.530 | 0.115 | 303.25 | True |
| cpx | cpx-only | P_kbar | MLP | alr | 42 | 46 | 0.130 | 0.950 | 1.641 | 0.317 | 295.67 | True |
