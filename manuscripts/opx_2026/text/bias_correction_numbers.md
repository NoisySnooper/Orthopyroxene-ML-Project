# Bias correction numbers for draft_v1 -> v2 update

Auto-generated from results/bias_correction_*.csv on 2026-04-19. Canonical seed 42. 20-seed mean used at regime=ALL where applicable. All RMSE in native units (kbar for P, C for T). % reduction defined as 100*(pre - post)/pre.

## For Section 1 (Introduction / headline claim)

**Strongest single finding (opx-only P):**
- Aggregate pre RMSE (raw (uncorrected) tuned baseline): 10.35 kbar
- Aggregate post RMSE: 6.01 kbar
- % reduction: 41.9%
- vs Putirka 29c at aggregate: 13.34 kbar
- % reduction vs Putirka 29c: 54.9%
- Per-seed ship vote: 20 of 20 seeds ship Form A, 0 ship Form B, 0 ship none
- Wins per regime (post-correction): the tuned ML baseline wins 4/4 non-ALL regimes

**Second finding (cpx-only P):**
- Aggregate pre RMSE (raw (uncorrected) tuned baseline): 13.66 kbar
- Aggregate post RMSE: 11.33 kbar
- % reduction: 17.1%
- vs Putirka 32a at aggregate: 11.83 kbar
- % reduction vs Putirka 32a: 4.2%
- Per-seed ship vote: 19 of 20 seeds ship Form A, 0 ship Form B, 1 ship none
- Wins per regime (post-correction): the tuned ML baseline wins 2/4 non-ALL regimes

**Ablation result (Form B ship count):**
- Form B ships on 0/8 combinations at canonical seed 42
- Form B ships on 0/160 per-seed cell decisions
- Interpretation: regime-based correction dominates value-based thresholding for this data -- residual structure is aligned with pre-registered P bins, not with extreme values of the prediction distribution.

## For Section 3.5 (Methods -- bias correction)

**Form A parameters for shipped cells (canonical seed 42):**
- opx-only P_kbar (RF/pwlr):
  - shallow_crustal: a=-0.050, b=1.51
  - deep_crustal_MASH: a=0.107, b=8.49
  - lithospheric_mantle: a=0.204, b=15.34
  - deeper_mantle: a=0.241, b=34.53
- cpx-only P_kbar (MLP/alr):
  - shallow_crustal: a=-0.009, b=0.61
  - deep_crustal_MASH: a=0.107, b=8.45
  - lithospheric_mantle: a=0.071, b=17.71
  - deeper_mantle: a=0.592, b=36.43

**Form B parameters at canonical seed 42 (diagnostic, none ship):**
- opx-liq T_C (ElasticNet/raw): alpha_L=0.050, alpha_R=0.830, s_L=0.409, s_R=-0.177
- opx-liq P_kbar (MLP/raw): alpha_L=0.130, alpha_R=0.550, s_L=0.549, s_R=0.106
- opx-only T_C (LightGBM/alr): alpha_L=0.130, alpha_R=0.550, s_L=-0.085, s_R=0.125
- opx-only P_kbar (RF/pwlr): alpha_L=0.290, alpha_R=0.910, s_L=-0.157, s_R=0.462
- cpx-liq T_C (ERT/pwlr): alpha_L=0.450, alpha_R=0.750, s_L=-0.080, s_R=-0.073
- cpx-liq P_kbar (LightGBM/pwlr): alpha_L=0.450, alpha_R=0.950, s_L=-0.074, s_R=0.178
- cpx-only T_C (ERT/pwlr): alpha_L=0.290, alpha_R=0.550, s_L=0.064, s_R=0.406
- cpx-only P_kbar (MLP/alr): alpha_L=0.130, alpha_R=0.950, s_L=1.679, s_R=0.286

**Form B stability across 5 CV reseeds (mean +/- std; see Table S8.11 for full list):**
- cpx-liq P_kbar: alpha_L=0.450+/-0.000, alpha_R=0.950+/-0.000, s_L=-0.064+/-0.010, s_R=0.186+/-0.084
- cpx-liq T_C: alpha_L=0.450+/-0.000, alpha_R=0.806+/-0.060, s_L=-0.072+/-0.011, s_R=-0.149+/-0.055
- cpx-only P_kbar: alpha_L=0.138+/-0.016, alpha_R=0.950+/-0.000, s_L=1.453+/-0.242, s_R=0.245+/-0.075
- cpx-only T_C: alpha_L=0.258+/-0.106, alpha_R=0.566+/-0.020, s_L=0.008+/-0.089, s_R=0.418+/-0.049
- opx-liq P_kbar: alpha_L=0.202+/-0.125, alpha_R=0.630+/-0.160, s_L=0.418+/-0.145, s_R=-0.003+/-0.254
- opx-liq T_C: alpha_L=0.050+/-0.000, alpha_R=0.846+/-0.020, s_L=0.475+/-0.093, s_R=-0.204+/-0.025
- opx-only P_kbar: alpha_L=0.306+/-0.041, alpha_R=0.854+/-0.048, s_L=-0.126+/-0.027, s_R=0.611+/-0.076
- opx-only T_C: alpha_L=0.130+/-0.000, alpha_R=0.550+/-0.000, s_L=-0.100+/-0.011, s_R=0.122+/-0.013

## For Section 4.3 (Results -- bias correction)

**Per-cell shipping decisions (canonical seed 42, with reasons):**

| track | target | model / feat. | winner | form A reason | form B reason |
|---|---|---|---|---|---|
| opx-liq | T_C | ElasticNet/raw | none | overall delta -0.1071 <= tol 1e-06 | regime worst degradation +1.614 > tol 1e-06 |
| opx-liq | P_kbar | MLP/raw | none | regime worst degradation +3.151 > tol 1e-06 | overall delta -0.1211 <= tol 1e-06 |
| opx-only | T_C | LightGBM/alr | none | regime worst degradation +3.903 > tol 1e-06 | regime worst degradation +0.4643 > tol 1e-06 |
| opx-only | P_kbar | RF/pwlr | A | ships: overall improves, no regime degrades | regime worst degradation +1.093 > tol 1e-06 |
| cpx-liq | T_C | ERT/pwlr | none | regime worst degradation +0.8988 > tol 1e-06 | regime worst degradation +1.566 > tol 1e-06 |
| cpx-liq | P_kbar | LightGBM/pwlr | none | regime worst degradation +0.9168 > tol 1e-06 | overall delta -0.1739 <= tol 1e-06 |
| cpx-only | T_C | ERT/pwlr | none | overall delta -12.02 <= tol 1e-06 | overall delta -7.589 <= tol 1e-06 |
| cpx-only | P_kbar | MLP/alr | A | ships: overall improves, no regime degrades | regime worst degradation +0.2531 > tol 1e-06 |

**opx-only P per-regime pre/post (ships A, unanimous over 20 seeds):**
- shallow_crustal n=48: pre=7.19 kbar, post=1.13 kbar, external best=2.20 kbar (Putirka 29c), winner=post-correction
- deep_crustal_MASH n=76: pre=5.21 kbar, post=1.97 kbar, external best=4.68 kbar (Putirka 29c), winner=post-correction
- lithospheric_mantle n=37: pre=4.03 kbar, post=3.34 kbar, external best=5.07 kbar (Putirka 29c), winner=post-correction
- deeper_mantle n=29: pre=19.65 kbar, post=14.55 kbar, external best=30.92 kbar (Putirka 29c), winner=post-correction
- ALL n=190: pre=10.33 kbar, post=6.03 kbar, external best=13.34 kbar (Putirka 29c), winner=post-correction

**cpx-only P per-regime pre/post (ships A at canonical, 19/20 seeds):**
- shallow_crustal n=236: pre=8.33 kbar, post=1.16 kbar, external best=2.84 kbar (Putirka 32b), winner=post-correction
- deep_crustal_MASH n=193: pre=3.28 kbar, post=1.70 kbar, external best=1.39 kbar (Jorgenson 2022), winner=external
- lithospheric_mantle n=213: pre=8.19 kbar, post=3.88 kbar, external best=3.83 kbar (Jorgenson 2022), winner=external
- deeper_mantle n=142: pre=18.43 kbar, post=25.41 kbar, external best=24.71 kbar (Putirka 32a), winner=pre-correction
- ALL n=784: pre=13.39 kbar, post=11.05 kbar, external best=11.83 kbar (Putirka 32a), winner=post-correction

**T correction failures (strict policy blocks shipping):**
- opx-liq T: aggregate Form A delta=-0.11 C at canonical seed; worst regime shallow_crustal degrades +18.68 C (post-pre).
- opx-only T: aggregate Form A delta=+24.56 C at canonical seed; worst regime lithospheric_mantle degrades +3.90 C (post-pre).
- cpx-liq T: aggregate Form A delta=+3.55 C at canonical seed; worst regime lithospheric_mantle degrades +0.90 C (post-pre).
- cpx-only T: aggregate Form A delta=-12.02 C at canonical seed; worst regime lithospheric_mantle degrades +31.18 C (post-pre).
- Interpretation: T residual structure is approximately symmetric across regimes, so per-regime OLS offsets average out. P residual structure is strongly regime-dependent, so Form A captures real bias.

## For Section 4 edge sensitivity

**opx-only P edge perturbation (+/-1 kbar on interior boundaries):**
- perturbation=base, edges=[0.0, 5.0, 15.0, 30.0, 100.0]: ships=True, overall_delta=4.30 kbar, max_regime_degradation=-4.15 kbar
- perturbation=inner_m1, edges=[0.0, 4.0, 14.0, 29.0, 100.0]: ships=True, overall_delta=4.16 kbar, max_regime_degradation=-3.76 kbar
- perturbation=inner_p1, edges=[0.0, 6.0, 16.0, 31.0, 100.0]: ships=True, overall_delta=5.17 kbar, max_regime_degradation=-2.24 kbar

**opx-liq P edge perturbation (marginal combination):**
- perturbation=base, edges=[0.0, 5.0, 15.0, 30.0, 100.0]: ships=False, overall_delta=1.23 kbar, max_regime_degradation=3.15 kbar
- perturbation=inner_m1, edges=[0.0, 4.0, 14.0, 29.0, 100.0]: ships=False, overall_delta=1.23 kbar, max_regime_degradation=3.15 kbar
- perturbation=inner_p1, edges=[0.0, 6.0, 16.0, 31.0, 100.0]: ships=True, overall_delta=1.48 kbar, max_regime_degradation=-0.87 kbar

- Conclusion (opx-only P): correction robust under +/-1 kbar edge perturbation; ships under both alternative edge placements.
- Conclusion (opx-liq P): sits at the stability edge; a +1 kbar shift of interior boundaries flips the ship decision from "no" to "yes". Supports 15 kbar boundary being borderline for this cell.

## For Section 4.4 cpx replication update

**Post-correction head-to-head vs external cpx models (full scorecard):**

| track | target | regime | n | post-correction | external best | method | winner |
|---|---|---|---|---|---|---|---|
| cpx-liq | T_C | shallow_crustal | 268 | 44.2 C | 30.1 C | Agreda 2024 | external |
| cpx-liq | T_C | deep_crustal_MASH | 187 | 51.1 C | 47.5 C | Agreda 2024 | external |
| cpx-liq | T_C | lithospheric_mantle | 143 | 77.9 C | 42.6 C | Agreda 2024 | external |
| cpx-liq | T_C | deeper_mantle | 56 | 166.8 C | 66.1 C | Putirka 33 | external |
| cpx-liq | T_C | ALL | 654 | 72.5 C | 63.4 C | Putirka 33 | external |
| cpx-liq | P_kbar | shallow_crustal | 268 | 4.97 kbar | 4.18 kbar | Putirka 30 | external |
| cpx-liq | P_kbar | deep_crustal_MASH | 187 | 2.78 kbar | 1.99 kbar | Agreda 2024 | external |
| cpx-liq | P_kbar | lithospheric_mantle | 143 | 4.37 kbar | 3.08 kbar | Jorgenson 2022 | external |
| cpx-liq | P_kbar | deeper_mantle | 56 | 17.50 kbar | 10.23 kbar | Putirka 30 | external |
| cpx-liq | P_kbar | ALL | 654 | 6.54 kbar | 5.48 kbar | Putirka 30 | external |
| cpx-only | T_C | shallow_crustal | 236 | 89.9 C | 64.8 C | Jorgenson 2022 | external |
| cpx-only | T_C | deep_crustal_MASH | 193 | 91.1 C | 56.5 C | Jorgenson 2022 | external |
| cpx-only | T_C | lithospheric_mantle | 213 | 140.0 C | 100.6 C | Agreda 2024 | external |
| cpx-only | T_C | deeper_mantle | 142 | 188.2 C | 197.9 C | Jorgenson 2022 | pre-correction |
| cpx-only | T_C | ALL | 784 | 127.3 C | 109.9 C | Jorgenson 2022 | external |
| cpx-only | P_kbar | shallow_crustal | 236 | 1.16 kbar | 2.84 kbar | Putirka 32b | post-correction |
| cpx-only | P_kbar | deep_crustal_MASH | 193 | 1.70 kbar | 1.39 kbar | Jorgenson 2022 | external |
| cpx-only | P_kbar | lithospheric_mantle | 213 | 3.88 kbar | 3.83 kbar | Jorgenson 2022 | external |
| cpx-only | P_kbar | deeper_mantle | 142 | 25.41 kbar | 24.71 kbar | Putirka 32a | pre-correction |
| cpx-only | P_kbar | ALL | 784 | 11.05 kbar | 11.83 kbar | Putirka 32a | post-correction |

