# Manuscript Writing Audit

Generated 2026-04-20. Read-only inventory for drafting. No writes to
results/, src/, scripts/, figures/, docs/, manuscripts/.
Target audience: authors assembling JGR:MLC draft.
Tone: caveman. Short sentences. Cite every number.
Repo SHA a9f102f844682e24e7bfbe9001d766dc6d0ec7ec, branch main, 356 dirty files.

## Section 1. Pre-registration state

Three docs define the locked protocol.

- [docs/preregistration/p_regime_preregistration.md](docs/preregistration/p_regime_preregistration.md). Locked 2026-04-17. Bin edges [0, 5, 15, 30, 100] kbar. Labels shallow_crustal, deep_crustal_MASH, lithospheric_mantle, deeper_mantle. Regime assignment uses predicted pressure at inference. Honesty bar P_REGIME_MIN_N_FOR_CLAIMS = 20.
- [docs/preregistration/nb03_test_protocol.md](docs/preregistration/nb03_test_protocol.md). 20-seed multiseed, seeds 42..61. Canonical seed 42. Optuna tune on train only. Citation-grouped 10-fold CV.
- [docs/preregistration/AMENDMENT_1_tiered_veto.md](docs/preregistration/AMENDMENT_1_tiered_veto.md). N_MIN_FOR_VETO = 20. Low-n regimes cannot veto ship. Amendment UNTRACKED in git. Flag as "amendment pending commit" in Methods §3.5.
- [docs/preregistration/AMENDMENT_2_veto_tolerance.md](docs/preregistration/AMENDMENT_2_veto_tolerance.md). Tolerance band veto_tol_r = max(SHIP_TOL, T_ABS, T_REL * pre_r). T_ABS_T = 10 C. T_ABS_P = 1 kbar. T_REL = 0.10. Amendment UNTRACKED in git. Flag as "amendment pending commit" in Methods §3.5.

Source for the ship-if-better rule: [src/bias_correction/__init__.py](src/bias_correction/__init__.py). SHIP_TOL = 1e-6. Constants in [scripts/bias_correction/rescore_under_v3_rule.py](scripts/bias_correction/rescore_under_v3_rule.py) verified by [tests/test_preregistration.py::test_T21_v3_constants_match_amendment](tests/test_preregistration.py).

Honest-disclosure note in AMENDMENT_2 §7 dated 2026-04-20. Observed 4 v3 ship flips vs 1 predicted. Three are tolerance-band rescues where deeper_mantle (n<=8) no longer vetoes under Amendment 1. Must appear in Methods rather than Discussion.

## Section 2. Dataset inventory

Sheet reads via pandas.read_parquet on the split files in data/splits/.

| pipeline | n_total | n_train | n_test | citations | P range kbar [1,99] | T range C [1,99] |
|---|---|---|---|---|---|---|
| opx_liq  |  600 |  426 | 174 | 93  | [0.00, 40.05] | [850.0, 1660.1] |
| opx_only | 1035 |  845 | 190 | 123 | [0.00, 60.00] | [850.0, 1656.6] |
| cpx_liq  | 2385 | 1731 | 654 | 230 | [0.00, 70.00] | [850.0, 1725.0] |
| cpx_only | 2897 | 2113 | 784 | 249 | [0.00, 135.0] | [800.0, 1750.0] |

opx per-regime test counts from [results/opx_per_regime_pivot_n.csv](results/opx_per_regime_pivot_n.csv). opx_liq: shallow 47, MASH 61, litho 58, deeper 8. opx_only: shallow 48, MASH 76, litho 37, deeper 29. Only opx_only deeper has n >= 20.

SHA256 prefixes from [data/hashes.json](data/hashes.json), date 2026-04-16.

| file | sha256[:8] |
|---|---|
| opx_liq.parquet     | 0c15cc4f |
| opx_only.parquet    | c38fa83e |
| ExPetDB.xlsx        | 8aebe073 |
| LEPR.xlsx           | c96ffea4 |

Use these hashes as a reproducibility pin in a Data Availability statement.

## Section 3. Model roster (20-seed multiseed)

Source: [results/opx_multiseed_summary.csv](results/opx_multiseed_summary.csv) (100 rows = 4 cells x 25 family+fs entries, count=20 seeds). [results/cpx_multiseed_summary.csv](results/cpx_multiseed_summary.csv) same shape. Top 3 per cell, ranked by mean test RMSE.

### opx

| cell | rank | family / fs | RMSE mean +/- std |
|---|---|---|---|
| opx_liq T_C    | 1 | ElasticNet / raw | 77.062 +/- 0.000 |
|                | 2 | RF / pwlr        | 84.883 |
|                | 3 | TabPFN / raw     | 84.885 |
| opx_liq P_kbar | 1 | MLP / raw        |  4.404 +/- 0.460 |
|                | 2 | GB / raw         |  4.757 |
|                | 3 | XGB / raw        |  4.788 |
| opx_only T_C   | 1 | LightGBM / alr   | 146.627 +/- 0.588 |
|                | 2 | XGB / alr        | 147.351 |
|                | 3 | GB / alr         | 148.442 |
| opx_only P_kbar| 1 | RF / pwlr        | 10.347 +/- 0.043 |
|                | 2 | XGB / pwlr       | 10.418 |
|                | 3 | XGB / alr        | 10.797 |

TabPFN ranks in opx: T=3, P=16 (opx_liq); T=4, P=21 (opx_only).

### cpx

| cell | rank | family / fs | RMSE mean +/- std |
|---|---|---|---|
| cpx_liq T_C    | 1 | TabPFN / raw    |  69.960 +/- 1.043 |
|                | 2 | ERT / pwlr      |  72.476 |
|                | 3 | CatBoost / pwlr |  72.775 |
| cpx_liq P_kbar | 1 | LightGBM / pwlr |   6.546 +/- 0.050 |
|                | 2 | XGB / pwlr      |   6.933 |
|                | 3 | GB / pwlr       |   7.052 |
| cpx_only T_C   | 1 | ERT / pwlr      | 127.022 +/- 0.305 |
|                | 2 | TabPFN / raw    | 131.395 |
|                | 3 | GB / pwlr       | 131.590 |
| cpx_only P_kbar| 1 | TabPFN / raw    |  13.419 +/- 0.580 |
|                | 2 | MLP / alr       |  13.664 |
|                | 3 | ERT / pwlr      |  14.083 |

TabPFN ranks in cpx: T=1, P=15 (cpx_liq); T=2, P=1 (cpx_only).

## Section 4. External benchmarks and TabPFN head-to-head

Source: [results/external_benchmark.csv](results/external_benchmark.csv) (59 rows). [results/tabpfn_head_to_head.csv](results/tabpfn_head_to_head.csv) (8 rows, rewritten 2026-04-20 with `tuned_best_*` prefix). Legend at [results/tabpfn_head_to_head_legend.md](results/tabpfn_head_to_head_legend.md).

Verdict rule: delta = tabpfn - tuned. tol = 0.5 * max(std). abs(delta) <= tol is "competitive". Else delta > 0 means v10_wins. Enum `v10_wins` kept for code compat; renders as "Tuned wins" in prose.

### Head-to-head table

| track | target | tuned best | TabPFN | external best | verdict |
|---|---|---|---|---|---|
| cpx_liq  | P_kbar | LightGBM/pwlr 6.55 +/- 0.05 | 7.51 +/- 0.13 | Agreda 2024 [P<20] 3.45 | v10_wins |
| cpx_liq  | T_C    | ERT/pwlr 72.48 +/- 0.21   | 70.06 +/- 1.04 | Agreda 2024 [P<20] 37.57 | tabpfn_wins |
| cpx_only | P_kbar | MLP/alr 13.66 +/- 0.28    | 13.41 +/- 0.58 | Agreda 2024 [P<20] 3.30 | competitive |
| cpx_only | T_C    | ERT/pwlr 127.02 +/- 0.30  | 130.55 +/- 2.96 | Agreda 2024 [P<20] 70.28 | v10_wins |
| opx_liq  | P_kbar | MLP/raw 4.40 +/- 0.46     | 5.46 +/- 0.14 | Putirka 29a filt 4.75 | v10_wins |
| opx_liq  | T_C    | ElasticNet/raw 77.06      | 84.55 +/- 0.84 | Putirka 28a filt 71.67 | v10_wins |
| opx_only | P_kbar | RF/pwlr 10.35 +/- 0.04    | 12.52 +/- 0.16 | Putirka 29c 13.34 | v10_wins |
| opx_only | T_C    | LightGBM/alr 146.63 +/- 0.59 | 150.65 +/- 1.51 | NOT AVAILABLE | v10_wins |

Counts: v10_wins 6, tabpfn_wins 1, competitive 1. One cell (opx_only T) has no Putirka equation available in Thermobar.

Agreda-Lopez "restricted to P<20 kbar" numbers are NOT the fair comparator for manuscript headline claims. They are calibration-window-restricted (n=524 of 654 for cpx_liq, n=575 of 784 for cpx_only) and drop the high-pressure tail where ML pipelines compete hardest. Manuscript should quote the full-range Agreda numbers (cpx_liq T 80.6 C, cpx_liq P 9.31 kbar, cpx_only T 112.1 C, cpx_only P 18.22 kbar) alongside P<20 numbers, not substitute one for the other.

Putirka 28a unfiltered on opx_liq T returns RMSE 367.97 C (r2 = -6.04) on 174 rows (external_benchmark.csv line 49). Filtered to 161 plausible rows it is 71.67 C (r2 = 0.748). Manuscript must footnote the filter. Unfiltered Putirka 29a on opx_liq P is catastrophically unstable (RMSE 876.4, r2 = -8295.7, line 45) and the filtered 126-row number (4.75 kbar) is the only defensible citation.

## Section 5. Bias correction three-rule comparison

Sources:
- [results/bias_correction_shipped.csv](results/bias_correction_shipped.csv) = v1 strict rule.
- [results/bias_correction_shipped_v2.csv](results/bias_correction_shipped_v2.csv) = v1 + Amendment 1 (N_MIN_FOR_VETO=20).
- [results/bias_correction_shipped_v3.csv](results/bias_correction_shipped_v3.csv) = v2 + Amendment 2 (veto tolerance).
- Form A params: per-regime OLS slope a_r, intercept b_r.
- Form B params: piecewise sigmoid with (alpha_L, alpha_R, a_L, a_R, s_L, s_R).

### Ship counts (non-TabPFN, 8 cells)

| rule | ships | which cells | flips vs prior |
|---|---|---|---|
| v1 | 2 | opx_only P (RF/A), cpx_only P (MLP/A) | baseline |
| v2 | 3 | + opx_liq P (MLP/A) | +1 (low-n deeper_mantle rescue) |
| v3 | 7 | + opx_liq T (ElasticNet/B), opx_only T (LightGBM/A), cpx_liq T (ERT/A), cpx_liq P (LightGBM/A) | +4 |

cpx_only T never ships under any rule. Degradation too large for tolerance.

### TabPFN ship decisions

Only opx_only T and opx_only P ship TabPFN under Form A (all three rules). opx_liq T and opx_liq P never ship TabPFN. cpx cells are "excluded" because TabPFN has no OOF residuals to fit a correction on. Note: the TabPFN ship decisions were computed using residuals from a separate TabPFN-internal leave-one-citation-out pseudo-OOF, not the same OOF as the tuned-family cells. Must footnote this in Methods §3.5 because the pre-registration assumes a common OOF pipeline.

### Honest-disclosure ratio

Predicted flips vs observed flips under v3: AMENDMENT_2 §7 dated 2026-04-20 says "predicted 1 flip, observed 4". Three of four are tolerance-band rescues where deeper_mantle (n=8 for opx_liq, n=29 for opx_only) would have vetoed at strict tol 1e-6. The rescues are:
- opx_liq T ElasticNet/B: max regime degradation +1.61 C <= tol 10 C.
- opx_only T LightGBM/A: max regime degradation +3.90 C <= tol 10 C.
- cpx_liq T ERT/A: max regime degradation +0.90 C <= tol 10 C.
- cpx_liq P LightGBM/A: max regime degradation +0.92 kbar <= tol max(1, 0.1 * pre).

Manuscript must be honest about the ex-post character of T_REL = 0.10 and T_ABS_{T=10 C, P=1 kbar}. These thresholds were set by inspecting residuals, not prior theory.

### Aggregate delta-RMSE at canonical seed 42 (cells that ship under v3)

| track | target | form | pre (v10 base) | post | delta | worst regime deg |
|---|---|---|---|---|---|---|
| opx_liq  | T_C    | B | 77.06 | 76.42 | -0.11 | +1.61 C (litho) |
| opx_liq  | P_kbar | A |  4.40 |  3.17 | +1.23 | -0.13 kbar (MASH) |
| opx_only | T_C    | A |147.98 |123.42 | +24.56| +3.90 C (litho) |
| opx_only | P_kbar | A | 10.35 |  6.05 | +4.30 | -4.15 kbar (deeper) |
| cpx_liq  | T_C    | A | 71.57 | 68.02 | +3.55 | +0.90 C (litho) |
| cpx_liq  | P_kbar | A |  6.54 |  5.73 | +0.81 | +0.92 kbar (MASH) |
| cpx_only | P_kbar | A | 13.66 | 11.25 | +2.41 | -0.85 kbar (MASH) |

Largest absolute gain opx_only T +24.56 C. Largest relative gain opx_only P 41.9%.

## Section 6. Per-regime results (post-correction)

Source: [results/regime_allmodels_postcorrection.csv](results/regime_allmodels_postcorrection.csv) (4839 rows, 19 columns). `method_family` enum: v10, v10_corrected, tabpfn, tabpfn_corrected, Putirka, Jorgenson, Agreda, Wang. `correction_form` in {A, none}.

### ALL-regime winner per cell (on reconstructed opx test split and cpx external corpora)

| cell | winner | RMSE | runner-up | RMSE |
|---|---|---|---|---|
| opx_liq T_C    | Putirka 28a         |  71.67 | v10_corrected ElasticNet/raw |  77.06 |
| opx_liq P_kbar | v10_corrected MLP/raw |  3.82 | (Form A did not improve)     |  --    |
| opx_only T_C   | tabpfn_corrected TabPFN/raw | 125.58 | v10 LightGBM/alr | 147.98 |
| opx_only P_kbar| tabpfn_corrected TabPFN/raw |  5.94 | v10_corrected RF/pwlr |  6.03 |
| cpx_liq T_C    | Putirka 33          |  63.43 | v10 CatBoost/pwlr |  71.57 |
| cpx_liq P_kbar | Putirka 30          |   5.48 | v10 LightGBM/pwlr |   6.54 |
| cpx_only T_C   | Jorgenson 2022      | 109.91 | Agreda 2024       | 112.12 |
| cpx_only P_kbar| v10_corrected MLP/alr |  11.05 | Putirka 32a     |  11.83 |

Clean ML wins at ALL regime: opx_liq P, opx_only P (TabPFN), cpx_only P. ML loses cleanly: cpx_liq T, cpx_liq P, cpx_only T. Mixed: opx_liq T (Putirka beats ML on aggregate but ML wins shallow and MASH), opx_only T (TabPFN post wins aggregate but no Putirka opx-only T exists).

### Per-regime headline for opx_only P (the shipped headline finding)

| regime | n | raw RF/pwlr | corrected RF/pwlr | TabPFN corrected | Putirka 29c |
|---|---|---|---|---|---|
| shallow_crustal      | 48 |  7.19 |  1.13 |  1.22 | 2.20 |
| deep_crustal_MASH    | 76 |  5.21 |  1.97 |  2.03 | 4.68 |
| lithospheric_mantle  | 37 |  4.03 |  3.34 |  3.14 | (not computed per-regime in CSV) |
| deeper_mantle        | 29 | 19.65 | 14.55 | 14.33 | (not computed per-regime in CSV) |
| ALL                  |190 | 10.33 |  6.03 |  5.94 |13.34 |

Correction improves every regime. Tabpfn_corrected slightly edges RF/pwlr_corrected on lithospheric (3.14 vs 3.34) and deeper (14.33 vs 14.55) and ALL. Manuscript headline should report RF/pwlr_corrected as the shipped tuned family and TabPFN_corrected as the scorecard winner, with a joint citation.

### TabPFN ships A on opx_only T: per-regime numbers

| regime | n | tuned LightGBM/alr | tabpfn_corrected TabPFN/raw |
|---|---|---|---|
| shallow_crustal     | 48 | 108.88 | 106.20 |
| deep_crustal_MASH   | 76 | 109.47 |  94.40 |
| lithospheric_mantle | 37 |  not best | not best (best = v10 ERT/raw 70.43) |
| deeper_mantle       | 29 | 265.19 | 223.69 |
| ALL                 |190 | 147.98 | 125.58 |

Every regime tracked shows TabPFN_corrected ahead. Only blocker for shipping tuned-family LightGBM was the +3.90 C deeper_mantle degradation at strict tol; tol 10 C rescues it under v3.

## Section 7. TabPFN verdict

Counts from Section 4: 1 win, 6 losses, 1 competitive across 8 cells. Augmented by bias-correction outcome: TabPFN ships A on 2 cells (opx_only T, opx_only P) and is the scorecard winner at ALL regime in those two cells. On the remaining 6 cells TabPFN is excluded from bias correction (cpx, all 4) or ships no correction (opx_liq, both).

### One-paragraph verdict text (caveman, drop-in)

"TabPFN v2 (Hollmann et al. 2025) was promoted to the 9th BASE_ORDER family on 2026-04-19. Evaluated on 20 seeds with raw oxide features, no Optuna tuning, no stacking, no SHAP, no bias correction fit. Won 1 of 8 cells (cpx_liq T 70.1 C vs tuned ERT/pwlr 72.5 C), competitive on 1 (cpx_only P 13.41 kbar vs tuned MLP/alr 13.66 kbar), lost 6. After bias correction applied separately to TabPFN's leave-one-citation-out pseudo-OOF, TabPFN_corrected is the scorecard winner at ALL regime for opx_only T (125.58 C) and opx_only P (5.94 kbar). On both opx_only cells TabPFN_corrected narrowly beats the shipped tuned family (LightGBM/alr 147.98 C, RF/pwlr_corrected 6.03 kbar). Manuscript should cite both TabPFN and the tuned family as co-primary for opx_only, not pick one. Do not cite TabPFN as the primary opx_liq model: TabPFN_corrected did not ship on opx_liq T or P."

## Section 8. SHAP inventory and gap report

Sources:
- [results/shap_importance_winners.csv](results/shap_importance_winners.csv). 310 rows, 8 cells x ~10-72 features each. Columns: pipeline, track, target, model, feature_set, feature, mean_abs_shap, rank.
- [results/opx_liq_shap_importance.csv](results/opx_liq_shap_importance.csv). 123 rows, older file, two opx_liq cells. Use shap_importance_winners as canonical.
- [results/shap_values_winners.npz](results/shap_values_winners.npz) and [results/opx_liq_shap_values.npz](results/opx_liq_shap_values.npz) hold full per-sample SHAP arrays.
- Figure producer [scripts/figures/make_fig_shap_winners.py](scripts/figures/make_fig_shap_winners.py) (Core_12).

### Coverage of the 8 pipeline x track x target cells

| cell | tuned model | explainer |
|---|---|---|
| opx_liq T_C     | ElasticNet/raw  | LinearExplainer exact |
| opx_liq P_kbar  | MLP/raw         | permutation importance (20 repeats, seed=42) |
| opx_only T_C    | LightGBM/alr    | TreeExplainer exact |
| opx_only P_kbar | RF/pwlr         | TreeExplainer exact |
| cpx_liq T_C     | ERT/pwlr        | TreeExplainer exact |
| cpx_liq P_kbar  | LightGBM/pwlr   | TreeExplainer exact |
| cpx_only T_C    | ERT/pwlr        | TreeExplainer exact |
| cpx_only P_kbar | MLP/alr         | permutation importance (surrogate) |

### Gaps

1. TabPFN has NO SHAP pathway. Panels (c) and (d) of Core_12 (opx_only T and P) show tuned runner-up. Scorecard winner TabPFN is flagged in subtitle + caption. This is a real explainability gap for the two cells where TabPFN_corrected is the scorecard winner.
2. KernelExplainer for MLP cells (opx_liq P, cpx_only P) was deferred 2026-04-20. Permutation importance used as surrogate. Runtime budget estimate ~30 min per cell.
3. Absolute mean |SHAP| values are NOT comparable across cells with different feature_sets (raw vs alr vs pwlr). The log-ratio transforms rescale the input. Manuscript must not plot SHAP on a shared axis across cells.
4. opx_liq SHAP file is older and has partial feature coverage (25 features per cell vs 72 for cpx_liq tree-family). Manuscript should cite shap_importance_winners.csv only.

## Section 9. Natural sample validation

Sources:
- [results/nb08_natural_predictions.csv](results/nb08_natural_predictions.csv). 327 rows, 12 columns. Two-pyroxene natural samples with ML opx-only, Jorgenson cpx-only, Putirka 2-px eq36/39 predictions. `equilibrium_pair_flag` is Kd-based equilibrium test.
- [results/nb08_cross_mineral_agreement.csv](results/nb08_cross_mineral_agreement.csv). 3 rows: ML opx-only, Jorgenson cpx-only, Putirka 2-px.
- [results/core11_extended_predictions.csv](results/core11_extended_predictions.csv). 301 rows, 17 columns. Extended external-benchmark predictions including Agreda, Jorgenson, Wang, Putirka eq28a/29c for natural samples.
- [results/nb07_conformal_qhat.json](results/nb07_conformal_qhat.json). Conformal half-widths at alpha = 0.1 from 43 calibration samples.

### Cross-mineral agreement table (n=327 two-pyroxene pairs)

| method | T RMSE C | T bias | P RMSE kbar | P bias |
|---|---|---|---|---|
| ML opx-only (ours)    | 114.94 | +80.21 | 5.53 | +2.46 |
| Jorgenson cpx-only    |  74.92 | +42.13 | 1.84 | +0.38 |
| Putirka 2-px eq36/39  |  77.57 (n=323) | +36.03 | 5.75 (n=323) | -2.33 |

Our opx-only is the worst of the three on this natural corpus. ML opx-only has a large positive T bias (+80 C) and positive P bias (+2.5 kbar), meaning it reads hotter and deeper than the reference methods. Manuscript must NOT frame this as "ML beats classical". It is the opposite for natural samples. Frame as "ML opx-only is calibration-competitive and deeper-range-usable but has a systematic hot/deep offset on natural two-pyroxene pairs that classical methods do not have". Possible mechanism: the training-domain ExPetDB+LEPR sample is denser in experiment-space than in natural-rock space, so the model extrapolates warmer/deeper on natural tails.

### Conformal intervals

alpha = 0.1. n_calibration = 43. q_hat_T = 92.58 C, q_hat_P = 10.0 kbar. Empirical coverage T = 0.793 (undercovered, nominal 0.90). Empirical coverage P = 0.960 (overcovered). The P_kbar q_hat is suspiciously round (exactly 10.0); nb07_conformal_qhat.json header says "Reconstructed 2026-04-20 to unblock nb08_natural_twopx". Flag as reconstructed-not-native. Manuscript should cite alpha, n_calibration, and empirical coverage, not just q_hat.

### Core_11 extended natural-sample file

301 rows with all 4 classical families + all 4 ML tracks on the same Experiment IDs. This is the substrate for Core_11 figure (two-pyroxene 1:1 plots). Two IDs (RP62A, Z-342-xx) span 900-980 C and 2-12 kbar. `P_putirka_opx_only_eq29c` NaN on 1 of 3 head rows.

## Section 10. Augmentation ablation

Source: [results/augmentation_ablation_opx_headline.csv](results/augmentation_ablation_opx_headline.csv) (4 rows, one per opx cell). 15x Gaussian composition-noise protocol per Agreda-Lopez 2024. Multiplicative 3% rel-std, non-negative clipping, citation-groups preserved across copies.

| cell | non-aug winner + RMSE | aug winner + RMSE | aug FormA ship | aug FormB ship | FormA post RMSE |
|---|---|---|---|---|---|
| opx_liq T_C   | ElasticNet/raw 77.06 | ElasticNet/raw 77.19 | 0/20 | 0/20 | 77.69 |
| opx_liq P_kbar| MLP/raw 4.40         | XGB/raw 4.76         |17/20 | 0/20 |  3.06 |
| opx_only T_C  | LightGBM/alr 146.63  | XGB/raw 153.78       |20/20 | 4/20 |146.68 |
| opx_only P_kbar| RF/pwlr 10.35       | ERT/pwlr 11.12       |20/20 | 0/20 |  5.95 |

Two-line verdict: augmentation degrades aggregate non-aug RMSE on 4/4 cells. Form B under augmentation ships 4/80 cell-seeds, Form A ships 57/80. Augmentation does not fix the Form B null. Manuscript has this drop-in at [manuscripts/opx_2026/text/aug_discussion_5_3.md](manuscripts/opx_2026/text/aug_discussion_5_3.md); text says "augmentation hypothesis is not supported".

## Section 11. Figure traceability

15 Core figures in [figures/core/](figures/core/) (pdf+png+txt each).

Proposed mapping to JGR:MLC 6-section structure (Intro, Methods, Results, Discussion, Conclusions, Data Availability). Figure numbers preliminary.

| Core | content | proposed section | producer script |
|---|---|---|---|
| 01   | dataset map (all pipelines) | Methods §3.1 | make_fig_dataset_map.py |
| 01b  | holdout dataset map         | Methods §3.1 | make_fig_dataset_map_holdout.py |
| 02   | citation-group split        | Methods §3.2 | make_fig_citation_split.py |
| 03   | methods flowchart           | Methods §3.3 | make_fig_methods_flowchart.py |
| 04   | nb04 cross-pipeline heatmap | Methods §3.4 | make_fig_nb04_cross_pipeline_heatmap.py |
| 05   | bias correction per-regime RMSE | Results §4.3 | make_fig30_... |
| 06   | bias correction residuals   | Results §4.3 | make_fig31_... |
| 07   | bias correction scorecard delta | Results §4.3 | make_fig34_... |
| 08   | opx headline                 | Results §4.1 | make_fig45_opx_headline.py |
| 09a  | opx per-regime family bars (pre-correction) | Results §4.2 | make_fig_opx_regime_families.py |
| 09b  | opx overall family bars (pre-correction)    | Results §4.2 | make_fig_opx_overall_families.py |
| 10   | best vs Putirka (pre-correction)            | Results §4.2 | make_fig_best_vs_putirka.py |
| 10b  | ArcPL bias-corrected vs Putirka (post)      | Results §4.3 | make_fig_arcpl_bias_corrected_vs_putirka.py |
| 11   | two-pyroxene 1:1 (ML pre-correction)        | Results §4.4 | make_fig_twopx_extended.py |
| 12   | SHAP winners (top 10 per cell)              | Results §4.5 | make_fig_shap_winners.py |

Caveats printed in suptitles:
- Core_09a/09b suptitle ends "(pre-correction)".
- Core_10b suptitle contrasts ML pre vs ML post vs Putirka.
- Core_11 suptitle ends "(ML bars = pre-correction)".
- Core_12 panels (c)(d) subtitle flags scorecard winner is TabPFN.

## Section 12. Manuscript autofill inventory

6 files in [manuscripts/opx_2026/text/](manuscripts/opx_2026/text/). Total 5435 words. Section and target location:

| file | words | destination | autofill date |
|---|---|---|---|
| aug_methods_3_9.md            | 225  | Methods §3.9 (Augmentation) | 2026-04-19 |
| aug_discussion_5_3.md         | 220  | Discussion §5.3             | 2026-04-19 |
| bias_correction_autofilled.md | 1473 | Methods §3.5 + Results §4.3 | 2026-04-19 |
| bias_correction_numbers.md    | 1512 | companion numbers file      | 2026-04-19 |
| regime_results_autofilled.md  | 1299 | Results §4.2                | 2026-04-18 |
| tabpfn_paragraph.md           | 706  | Discussion §5 (TabPFN)      | 2026-04-20 |

### Terminology audit

Autofill prose uses several terms that need consolidation before the draft lands:
- "v10" vs "tuned ML baseline" vs "tuned family". Legend for tabpfn_head_to_head renamed `v10_best_*` to `tuned_best_*` on 2026-04-20 but the enum string `v10_wins` is kept for code backcompat. Manuscript should use "tuned" everywhere, never "v10". "v10" is internal version slang.
- "SHIP_TOL = 1e-6" is a numerical tolerance, not a degradation tolerance. v3 degradation tolerance is (T_ABS, T_REL). Prose must not conflate.
- "post-correction" vs "corrected" vs "form A applied". Pick one.
- "regime" bin labels: shallow_crustal, deep_crustal_MASH, lithospheric_mantle, deeper_mantle are the locked preregistered labels. Do not rename in prose.

### Gaps in autofilled prose vs target draft

1. No Introduction prose exists. Must be hand-written.
2. No Methods §3.1 (datasets) or §3.2 (citation split) prose. Methods §3.5 (bias correction) and §3.9 (augmentation) are filled. §3.3 (ML pipeline), §3.4 (feature sets), §3.6 (stacking), §3.7 (SHAP), §3.8 (bias correction Form B math) are NOT filled.
3. No Discussion overarching prose. §5.3 (augmentation) and TabPFN paragraph are filled; sections about ML-vs-classical tradeoff and natural-sample bias are NOT filled.
4. No Conclusions prose.
5. No Data Availability or Code Availability statement.
6. No Abstract.

### Tables directory

6 files in [manuscripts/opx_2026/tables/](manuscripts/opx_2026/tables/). T4 (shipped bias correction), S9 (per-seed bias correction), S10 (bias correction stability). Each exists as csv + tex. No other tables have been autofilled. Expect T1 (dataset inventory), T2 (model roster), T3 (head-to-head) to be authored by hand from Sections 2, 3, 4 above.

## Section 13. Open issues

1. Amendments 1 and 2 UNTRACKED. Must commit before Methods §3.5 ships.
2. TabPFN not installed in the live .venv (pip-check shows no tabpfn or tabpfn_client). The 20-seed TabPFN numbers in results/ are from an earlier environment. Methods §3.3 must pin TabPFN v2 version used. Data provenance gap.
3. Conformal q_hat_P_kbar = 10.0 is suspiciously round; nb07_conformal_qhat.json header says "Reconstructed 2026-04-20". Re-derive or cite the archive source (archive/pre_v10_rebuild_2026_04_16/results/nb07_conformal_qhat.json).
4. ML opx-only has a +80 C / +2.5 kbar positive bias on natural two-pyroxene pairs (Section 9). No accompanying prose exists. Needs a Discussion paragraph.
5. cpx_only T never ships bias correction under any v3 tolerance. Prose not written.
6. opx_liq T ElasticNet/B v3-promoted after AMENDMENT_2 (degradation +1.6 C within tol 10 C). Tolerance set after inspecting residuals. Must be disclosed as post-hoc tolerance calibration.
7. Per-regime deeper_mantle n=8 for opx_liq. Any opx_liq deeper_mantle claim fails the honesty bar (n<20).
8. 356 dirty files in git as of SHA a9f102f. Commit/branch before drafting or hashes drift.

## Section 14. Compute provenance

From `python -m pip show` inside .venv:
- Python 3.13.13
- pandas 3.0.2
- scikit-learn 1.8.0
- xgboost 3.2.0
- lightgbm 4.6.0
- catboost 1.2.10
- shap 0.51.0
- Thermobar 1.0.70
- NumPy 2.x (locked by pandas 3.0.2)
- tabpfn NOT INSTALLED in live .venv. tabpfn_client NOT INSTALLED. Flag.

OS: Windows 11 Pro 10.0.26200. Shell: bash (MINGW64/git-bash). No GPU. TabPFN ran on CPU per tabpfn_paragraph.md.

Repository: branch main, SHA a9f102f844682e24e7bfbe9001d766dc6d0ec7ec. 356 dirty files at audit time (most are results csv, optuna study joblib, nb*.ipynb regenerations).

Execution logs in [logs/](logs/). Most recent: Phase 3.3b Optuna completion 2026-04-18.

## Section 15. Gap checklist (for draft assembly)

Before the first pass draft compiles, the following must exist:

- [ ] Commit AMENDMENT_1_tiered_veto.md and AMENDMENT_2_veto_tolerance.md.
- [ ] Pin TabPFN v2 version in Methods §3.3 (install tabpfn or cite tag). Re-run or cite archived TabPFN numbers.
- [ ] Re-derive or cite archive source for nb07_conformal_qhat.json.
- [ ] Write Introduction prose.
- [ ] Write Methods §3.1 (datasets), §3.2 (citation split), §3.3 (ML pipeline), §3.4 (feature sets), §3.6 (stacking), §3.7 (SHAP), §3.8 (bias correction Form B math).
- [ ] Write Results §4.1 (headline), §4.4 (two-pyroxene), §4.5 (SHAP).
- [ ] Write Discussion paragraphs on ML-vs-classical tradeoff, natural-sample bias, cpx_only T ship failure.
- [ ] Write Conclusions.
- [ ] Write Abstract.
- [ ] Write Data Availability statement citing the 4 SHA256 hashes.
- [ ] Write Code Availability statement citing scripts/ and src/.
- [ ] Build T1 (dataset inventory), T2 (model roster), T3 (head-to-head) tables from Sections 2, 3, 4.
- [ ] Consolidate terminology ("tuned" not "v10"; "post-correction" not "corrected"; pick one).
- [ ] Honest-disclosure note on AMENDMENT_2 ex-post tolerance calibration must appear in Methods §3.5, not Discussion.
- [ ] Natural-sample opx-only +80 C / +2.5 kbar bias paragraph for Discussion.
- [ ] Commit current dirty state or reset to clean SHA so Data Availability hashes match shipped CSVs.

End of audit.
