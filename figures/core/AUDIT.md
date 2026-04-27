# Core figure audit (Tier 1 + Tier 2)

Entry template:

```
Core_NN_<stem>
  Caption claim: "..."
  Data source:   results/XYZ.csv (or data/processed/*.parquet)
  Models shown:  [list]
  TabPFN:        yes | no | N/A (reason)
  Regression:    matches caption; RMSE within 0.01 of CSV; winner verdict matches scoreboard
  Visual check:  yes | no
  Notes:         (any caveats)
```

Status codes: `[new]` built in Tier 1, `[rebuilt]` rebuilt in Tier 2,
`[copied]` copied from flat in Tier 2, `[deferred]` TabPFN rebuild
deferred (data not available at the needed granularity).

---

## Core_01_fig_dataset_map  [new]

- Caption claim: "P-T distribution of all experiments in ExPetDB 2025-07-21 across 4 pyroxene tracks, colored by pre-registered P regime"
- Data source:   data/processed/{opx,cpx}_clean_{opx_liq,opx_only,cpx_liq,cpx_only}.parquet; data/splits/*_indices_*.npy
- Models shown:  N/A (pure data figure)
- TabPFN:        N/A
- Regression:    n per track matches parquet row counts; regime edges [0,5,15,30,100] kbar match docs/preregistration/p_regime_preregistration.md
- Visual check:  pending
- Notes:         train/test marker distinction via alpha + edge; per-panel 2-column stats block below scatter

## Core_02_fig_citation_split  [new]

- Caption claim: "Citation-grouped 10-fold split vs naive row-random split"
- Data source:   schematic (no CSV)
- TabPFN:        N/A
- Visual check:  pending
- Notes:         30x6 visualization grid; actual opx corpus has ~93 publications

## Core_03_fig_methods_flowchart  [new]

- Caption claim: "10-stage pipeline ExPetDB raw -> per-regime test evaluation; 9 model families; 8 tuned + TabPFN untuned"
- Data source:   schematic (no CSV)
- Models shown:  9 families
- TabPFN:        yes (stage 4, stage-5 alt path, stage 7 5-seed)
- Visual check:  pending
- Notes:         V-branch 4->5; box 6 centered between both #5 boxes for visual balance

## Core_04_fig_nb04_cross_pipeline_heatmap  [copied]

- Caption claim: "Per-family per-track RMSE heatmap"
- Data source:   produced by nb04 ensemble construction
- TabPFN:        yes (9th row)
- Notes:         only .png available in flat; caption added manually

## Core_05_fig30_bias_correction_per_regime_rmse  [copied, TabPFN rebuild deferred]

- Data source:   results/bias_correction_summary.csv + checkpoints
- TabPFN:        no (per-regime TabPFN RMSE not in results/ at needed granularity)
- Notes:         defer rebuild; TabPFN per-regime coverage is instead shown in Core_09a (all families + Putirka per regime) and Core_10 (best-of-ours vs Putirka per regime)

## Core_06_fig31_bias_correction_residuals  [copied, TabPFN rebuild deferred]

- Data source:   results/bias_correction/checkpoints/
- TabPFN:        no (per-sample TabPFN residuals not cached at canonical seed)
- Notes:         TabPFN residual behavior summarized in fig44 (kept in SI as fig44_tabpfn_bias_scoreboard_opx)

## Core_07_fig34_bias_correction_scorecard_delta  [copied, TabPFN rebuild deferred]

- Data source:   results/preregistered_scorecard_postcorrection.csv
- TabPFN:        partial (scorecard has tabpfn_post_rmse columns but the v9 script only visualizes v10_post); the alternative TabPFN view is Core_09a per regime
- Notes:         full TabPFN heatmap rebuild deferred; readers get TabPFN regime-level info via Core_09a

## Core_08_fig45_opx_only_P_headline  [copied]

- Caption claim: "opx-only P is the ship-positive combo; 4-panel headline"
- Data source:   results/preregistered_scorecard_postcorrection.csv, bias_correction_shipped.csv
- TabPFN:        yes (TabPFN opx_only P ships with Form A)
- Notes:         produced by scripts/figures/make_opx_only_P_headline_fig.py

## Core_09a_fig_opx_regime_families  [new]

- Caption claim: "All 9 model families + Putirka per pre-registered pressure regime, 4 panels"
- Data source:   results/regime_allmodels.csv (pre-computed; includes tabpfn method_family)
- Models shown:  9 families + Putirka
- TabPFN:        yes (colored bar, 9th family)
- Regression:    per-regime feature_set winner; 95% CI whiskers from rmse_lo/rmse_hi
- Notes:         per-regime winner annotation under x-axis; TabPFN (vermillion) frequently wins deep regimes

## Core_09b_fig_opx_overall_families  [new]

- Caption claim: "All 9 model families + Putirka overall test-set RMSE, 4 panels"
- Data source:   results/opx_multiseed_summary.csv + tabpfn_multiseed_summary.csv + scorecard (ALL-row Putirka)
- TabPFN:        yes (10th bar with Putirka as 11th? actually TabPFN as 9th in MODEL_ORDER, Putirka as 10th)
- Regression:    family winner = feature_set with min mean; CI from mean +/- 1.96 * std
- Notes:         per-panel winner annotation in upper-right

## Core_10_fig_best_vs_putirka  [new]

- Caption claim: "Best of our 9 model families vs Putirka/Agreda external benchmark, per regime"
- Data source:   results/preregistered_scorecard_postcorrection.csv
- TabPFN:        included in "best of ours" pool
- Notes:         winning family name labeled below each "ours" bar; tight ylim; panel (c) no Putirka opx-only T

## Core_11_fig_nb08_twopx_1to1  [copied]

- Caption claim: "Two-pyroxene natural sample 1:1 check"
- Data source:   nb08 cross-mineral agreement
- TabPFN:        N/A (uses shipped model)

## Core_14a_fig_pairing_opx  [new]

- Caption claim: "Opx-perspective pairing matrix (13 rows, LEPR two-pyroxene corpus); three sections: vs baseline regression, vs own ML, vs external ML"
- Data source:   results/nb08_natural_predictions.csv + results/core11_extended_predictions.csv merged on Experiment; row stats from results/pairing_matrix_22rows.csv
- Models shown:  rows C, G, H, O (baseline); A, F, I (own ML); B, D, E, M, N, P (external ML)
- TabPFN:        no (cpx-only own-ML row I uses shipped TabPFN raw P, but no TabPFN-specific axis)
- Regression:    RMSE of disagreement + 95% bootstrap CI (n_boot=500, seed=42) per panel title; matches pairing_matrix_22rows.csv
- Visual check:  pending
- Notes:         Replaces former single Core_14_fig_pairing_panels.pdf. Row C (Putirka 2008 28a/29a) revived: 28a T real, 29a P unavailable -> P panel renders 'no valid pairs' placeholder. Rows H and O share data with G because corpus has no separate Brey-Kohler / Putirka cpx-only natural predictions. Section headers rendered as gray strips.

## Core_20_fig_locality_performance  [new]

- Caption claim: "Per-locality fraction of predicted T inside literature-bracketed expected range, stratified by pipeline (opx_only, cpx_only, twopx); n>=20 honesty bar enforced"
- Data source:   results/nb08_locality_stratified.csv (38 rows from H.6 spatial-join)
- Models shown:  canonical opx_only (LightGBM/alr T, RF/pwlr P), cpx_only (ERT/pwlr T, MLP/alr P substituted for TabPFN/raw), twopx (ElasticNet/raw T, XGB/alr P)
- TabPFN:        substituted for cpx_only P (head-to-head verdict 'competitive', §3.11)
- Regression:    no per-regime numeric RMSE on natural samples (pre-registration boundary); fraction-in-range only
- Visual check:  pending
- Notes:         3 horizontal-bar panels share [0,1] x-axis; cratonic xenolith Kaapvaal/Siberia at 87-96% on cpx_only T but 0% on twopx P (cratonic shallow-bias finding); Iceland at 0% on opx_only T (rift cool-bias finding). Tightened cratonic T ranges (Kaapvaal 950-1300, Siberia 900-1300) used in this figure per the H follow-up Task 4.

## Core_21_fig_opx_showcase  [new]

- Caption claim: "opx_only canonical pipeline performance on 11 claims-eligible curated localities"
- Data source:   results/nb08_locality_stratified.csv (per-locality medians via H.6 spatial join)
- Models shown:  opx_only canonical: LightGBM/alr T, RF/pwlr P
- TabPFN:        N/A (opx_only canonical is not TabPFN)
- Regression:    1:1 scatter against literature midpoint with +/-100 °C / +/-3 kbar envelope; residual = ML T_med - lit midpoint
- Visual check:  pending
- Notes:         4-panel layout (1:1 T, 1:1 P, T residuals, P residuals); markers sized by log(n_samples); colored by tectonic-setting on Okabe-Ito; Spitsbergen +240 °C and Iceland -244 °C are largest opposite-sign T residuals

## Core_22_fig_cpx_showcase  [new]

- Caption claim: "cpx_only canonical pipeline performance on 14 claims-eligible curated localities"
- Data source:   results/nb08_locality_stratified.csv
- Models shown:  cpx_only canonical: ERT/pwlr T, MLP/alr P (TabPFN/raw substituted per §3.11 disclosure)
- TabPFN:        substituted for cpx_only P; head-to-head 'competitive' verdict
- Regression:    1:1 scatter against literature midpoint with +/-100 °C / +/-5 kbar envelope (P envelope wider than Core_21 because cpx_only P has wider scatter on natural samples)
- Visual check:  pending
- Notes:         Same 4-panel layout as Core_21; Kaapvaal P_med 57.5 kbar (64.7% in [20,70]) and Siberia 59.6 kbar (45.5% in [20,60]) validate the cpx_only barometer on deep-mantle natural samples

## Core_23_fig_cross_pipeline_disagreement  [new]

- Caption claim: "Cross-pipeline T (panel a) and P (panel b) disagreement per curated locality where two or more pipelines have n>=20"
- Data source:   results/nb08_locality_stratified.csv
- Models shown:  paired comparisons among opx_only, cpx_only, twopx; preferred pair opx_only vs cpx_only
- TabPFN:        N/A
- Regression:    |delta| sorted; 50 °C and 5 kbar deployment-threshold red dashed lines
- Visual check:  pending
- Notes:         Iceland T disagreement 192 °C is the largest single-locality cross-pipeline gap (matches §4.8 prose 193 °C within rounding tolerance); Kaapvaal twopx-vs-cpx_only P disagreement ~44 kbar is the cratonic shallow-bias finding

## Core_14b_fig_pairing_cpx  [new]

- Caption claim: "Cpx-perspective pairing matrix (8 rows, LEPR two-pyroxene corpus); four sections: vs baseline regression, vs own ML (cross-mineral, duplicated from 14a), vs external ML, external vs external"
- Data source:   results/nb08_natural_predictions.csv + results/core11_extended_predictions.csv merged on Experiment; row stats from results/pairing_matrix_22rows.csv
- Models shown:  row W (NEW, baseline); A, I (own ML, cross-mineral); Q, R, S, T (external ML); J (external vs external)
- TabPFN:        no (own-ML row I uses shipped TabPFN raw P, no TabPFN-specific axis)
- Regression:    RMSE of disagreement + 95% bootstrap CI (n_boot=500, seed=42) per panel title; row W new in pairing_matrix_22rows.csv (T_rmse=66.8 C, P_rmse=7.74 kbar)
- Visual check:  pending
- Notes:         Replaces former single Core_14_fig_pairing_panels.pdf. Section 1 has only one row (W: cpx-liq vs Putirka 2-px) because corpus has no real cpx-only natural-sample predictions and no separate Putirka cpx-liq baseline columns. Rows S, T and the cpx-only legs of J use cpx-liq data labelled as cpx-only (known proxy from compute_pairing_matrix.py:111-122). Row J isolated to its own external-vs-external section.

---

## Deferred TabPFN rebuilds

fig28, fig30, fig31, fig34 were copied into Core_04..07 as-is from their
pre-TabPFN state. Full TabPFN-aware rebuilds require either per-regime
TabPFN RMSE (not cached in results/) or per-sample TabPFN residuals at
the canonical seed (also not cached). Running those compute passes was
out of scope for this audit pass. The TabPFN story is instead carried
by:

- Core_09a (all 9 + Putirka per regime)
- Core_09b (all 9 + Putirka overall)
- Core_10  (best-of-ours vs Putirka per regime)
- Core_08  (opx-only P ship headline, includes TabPFN)
- SI fig44 (TabPFN-only bias scoreboard)
- SI fig35 (TabPFN vs opx_tb head-to-head)

If a future pass needs full TabPFN-aware fig28/30/31/34, it should
re-run the TabPFN bias correction with per-regime RMSE accumulation
and residual checkpoint retention.

---

## SI audit sweep

All remaining pre-core flats have been moved to `figures/SI/`.

SI set (as of Tier 2 close):

- fig24_per_regime_rmse_opx_liq (predates TabPFN; opx_liq only)
- fig25_per_regime_residual_violins_opx_liq
- fig26_generalization_opx_liq
- fig27_shap_summary_opx_liq
- fig28_bias_correction_opx_liq
- fig29_twopx_benchmark (same data as Core_11, wider scope)
- fig30_bias_correction_per_regime_rmse (Core_05 copy lives in core)
- fig31_bias_correction_residuals (Core_06 copy lives in core)
- fig32_bias_correction_form_comparison
- fig33_bias_correction_per_seed_stability
- fig34_bias_correction_scorecard_delta (Core_07 copy lives in core)
- fig35_tabpfn_vs_opx_tb
- fig44_tabpfn_bias_scoreboard_opx
- fig45_opx_only_P_headline (Core_08 copy lives in core)
- fig_aug01_ship_verdict_comparison
- fig_aug02_aggregate_rmse_delta
- fig_aug03_residual_structure_per_regime
- fig_aug04_form_b_breakpoint_stability
- fig_h5ac_opx_world_map
- fig_h5b_opx_interactive (html)
- fig_nb04_ensemble_lift
- fig_nb04_winning_base_histogram
- fig_nb08_twopx_1to1 (Core_11 copy lives in core)

SI audit verdict: the SI set is internally consistent and each figure
is referenced by a matching .txt caption (where available). No
regeneration needed in this pass.
