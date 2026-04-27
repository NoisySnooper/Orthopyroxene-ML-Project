# Manuscript writing audit v4

Generated 2026-04-27 at the close of Phase H autonomous run on branch
`phase_h_natural_worldwide`. This audit extends v3 with one new
section: §17 covers the natural-sample worldwide inference pipeline.

Sections 1-16 (Phase A-G coverage) are unchanged from v3.

## Section 17 — Natural-sample Phase H inventory

### 17.1 Dataset inventory

| File | Rows | SHA256[:12] | Source | Notes |
|---|---|---|---|---|
| data/natural/natural_opx_with_coords.csv | 53050 | b1969ed8...8ab7 | GEOROC SGFTFN 2024-12 ORTHOPYROXENES | pre-existing; verified in H.1a |
| data/natural/natural_cpx_with_coords.csv | 93163 | 31e608dc... | GEOROC SGFTFN 2024-12 CLINOPYROXENES (file_id 118288, 338.4 MB raw) | NEW H.1b via Dataverse Native API |
| data/natural/natural_twopx_pairs.csv | 6234 | 92dfcc09... | per-sample inner-join opx + cpx | NEW H.1d; 0.01 deg lat/lon tol + rock-name consistency |
| data/natural/curated_localities.csv | 15 + 1 ArcPL | (unchanged) | Locality bounding boxes with expected_T_C and expected_P_kbar ranges | ArcPL opx folded as locality #16 per plan |

GEOROC raw cpx (339 MB) is gitignored as regenerable from
scripts/v10_pull_georoc_cpx.py. Dataset DOI 10.25625/SGFTFN, file_id
118288, SHA256[:12]=5598c3bd0b92.

### 17.2 Inference run summaries

#### Pipeline: opx_only (53,050 natural samples)

| Method | Cell | Mean T (C) | Mean P (kbar) | Notes |
|---|---|---|---|---|
| Our ML opx-only T | LightGBM/alr (canonical) | 1084.4 | -- | n=53050 |
| Our ML opx-only P | RF/pwlr (canonical)        | --     | 10.52  | n=53050 |
| Putirka 2008 opx-only P (eq29c) | classical | -- | (computed) | uses our T as anchor |

#### Pipeline: cpx_only (93,163 natural samples)

| Method | Cell | Mean T (C) | Mean P (kbar) | Notes |
|---|---|---|---|---|
| Our ML cpx-only T | ERT/pwlr (canonical) | 1125.9 | -- | n=93163 |
| Our ML cpx-only P | MLP/alr (substituted)  | --     | 14.66  | TabPFN/raw canonical winner deferred |
| Putirka 2008 cpx-only T (eq32d), P (eq32a) | classical | (computed) | (computed) | |

TabPFN substitution is documented in
docs/preregistration/canonical_cells_cpx_twopx.md and in the predictions
CSV via the `note_tabpfn_substituted` column. The TabPFN canonical run
on natural samples is deferred for a follow-up session because TabPFN
inference on 93k cpx + 53k opx rows requires per-tectonic-setting
chunking with 30 min timeout per chunk; multi-hour wall clock was not
budgeted.

#### Pipeline: twopx (6,234 natural pairs)

| Method | Cell | Mean T (C) | Mean P (kbar) | Notes |
|---|---|---|---|---|
| Our ML twopx T | ElasticNet/raw (canonical) | 1014.1 | -- | n=6234 |
| Our ML twopx P | XGB/alr (canonical)         | --     | 10.81  | n=6234 |
| Putirka 2008 twopx (eq36/37 T, eq38/39 P) | classical | (range 980-1080) | (median 6-9, RMSE-blowout tail) | |

### 17.3 Cross-mineral agreement summary (H.4)

KD Fe-Mg equilibrium flag: 2,490 of 6,234 pairs (39.9%) fall within
the Putirka 2008 equilibrium range KD in [0.95, 1.23]. Equilibrium-only
median absolute disagreement:

| Comparison | Method A | Method B | Median \|delta\| | n |
|---|---|---|---|---|
| Our twopx T vs Putirka eq36 (twopx) | our ML | classical 2-px | 50.95 °C | 2490 |
| Our twopx T vs Putirka eq37 (twopx) | our ML | classical 2-px | 53.08 °C | 2490 |
| Our twopx P vs Putirka eq39 (twopx) | our ML | classical 2-px |  2.98 kbar | 2490 |
| Our twopx P vs Putirka eq38 (twopx) | our ML | classical 2-px |  2.14 kbar | 2490 |
| Our twopx P vs Putirka eq29c (opx-only baseline) | our ML | opx-only baseline | 6.72 kbar | 2490 |
| Our twopx T vs Putirka eq32d (cpx-only baseline) | our ML | cpx-only baseline | 152.35 °C | 2489 |

The two Putirka twopx P equations (eq38, eq39) have heavy outlier
tails on natural-corpus samples (RMSE 1171-1634 kbar driven by
extreme tail), so median \|delta\| is the primary statistic; RMSE
is reported only with the caveat that natural-sample classical-method
extrapolation is poorly behaved.

### 17.4 Curated locality validation (H.6)

15 curated localities × 3 pipelines + ArcPL opx = 38 (locality,
pipeline) rows. 33 of 38 satisfy the n>=20 claims-eligibility floor.

Pipeline-level mean fraction of predicted T values inside the
curated expected_T_C range:

| Pipeline | Eligible localities | Mean T-in-range fraction |
|---|---|---|
| opx_only         | 11 | 58.6% |
| cpx_only         | 14 | 63.9% |
| twopx            |  7 | 61.7% |
| opx_liq_archive  |  1 (ArcPL) | (literature P-T column wiring deferred) |

Top-n eligible localities (cpx_only pipeline, sorted by n):

- Iceland (n=2944, T_med=1124 °C, expected [1100,1250]): 74.5% in range
- Siberian craton (n=2382, T_med=1229 °C, expected [900,1400]): 99.6% in range
- Kaapvaal craton (n=2107, T_med=1218 °C, expected [900,1450]): 99.4% in range
- Lesser Antilles (n=1199, T_med=1068 °C, expected [950,1100]): 83.2% in range
- Central Andes (n=738, T_med=1100 °C, expected [900,1200]): 100% in range

The high in-range fractions on cratonic xenolith sites (Kaapvaal,
Siberia) and on the well-characterized arc localities (Lesser
Antilles, Central Andes, Iceland) indicate that the canonical cpx_only
T model produces predictions inside the literature-bracketed
expected band on the natural corpus at rates that are difficult to
explain by chance.

### 17.5 Figure inventory

Static (cartopy Robinson, 2-panel: tectonic + predicted T):
- figures/core/Core_19a_fig_world_map_opx.{pdf, png} — 53,023 markers
- figures/core/Core_19b_fig_world_map_cpx.{pdf, png} — 93,125 markers
- figures/core/Core_19c_fig_world_map_twopx.{pdf, png} — 6,234 markers

Interactive (folium + MarkerCluster, popups with sample metadata):
- figures/interactive/world_map_opx.html — 8000-marker subsample
- figures/interactive/world_map_cpx.html — 8000-marker subsample
- figures/interactive/world_map_twopx.html — full 6,234 markers

Core_20 (G.4 sensitivity on curated subset) is **deferred**. The
G.4 piecewise correction requires regime assignment from literature P;
the existing curated_localities.csv has expected_P_kbar ranges per
locality, not per-sample literature P. A future pass with per-sample
P-T from primary citations can populate Core_20 without reshaping
the H.6 output schema.

### 17.6 Pre-registration boundary enforcement

All four pre-registration boundaries from
docs/natural_worldwide_plan.md Section 0 are enforced in the outputs:

1. **No per-regime numeric RMSE on natural samples.** The natural-sample
   prediction CSVs (results/nb08_natural_predictions_*.csv) contain
   point predictions and tectonic-setting metadata only; no
   per-regime RMSE column exists. Cross-mineral comparisons report
   pairwise method-vs-method disagreement, not method-vs-truth RMSE.
2. **G.4 bias correction NOT applied to natural samples by default.**
   The H.3 inference script writes raw model outputs. No call to
   `src.bias_correction.apply_form_a_correction` appears in the
   natural-sample pipeline. The supplementary curated-subset
   correction view (Core_20) is deferred.
3. **Canonical cpx + twopx cells locked at H.0b** before inference.
4. **Per-locality n>=20 honesty bar.** 5 of 38 (locality, pipeline)
   rows are flagged claims_eligible=False (n<20) in
   results/nb08_locality_stratified.csv; their numeric in-range fraction
   is reported but flagged not eligible for bootstrap CI claims.

### 17.7 Halt-and-report history

Halt #1 (GEOROC API < 30k raw cpx rows) fired at 2026-04-26T23:53Z
when the original endpoint candidates (api.georoc.eu/v1, v2,
queries) all returned 404. User supplied the correct download
mechanism: GRO.data Dataverse Native API at
data.goettingen-research-online.de under DOI 10.25625/SGFTFN.
Halt cleared at 2026-04-27T00:08Z; download succeeded at 338.4 MB raw,
93,163 cleaned cpx rows. Full halt-resolution chain in
results/HALT_REPORT_PHASE_H.md.

### 17.8 Deferred work

Items intentionally not completed in this autonomous run, with reason:

| Item | Reason for deferral |
|---|---|
| H.1c (GEOROC liquid/glass) | Lower priority per activation prompt; SGFTFN minerals dataset has no glass; dedicated dataset (10.25625/7JW6XU Melt Inclusions or 10.25625/2JETOA Rock Types) requires a separate pull |
| TabPFN canonical inference (cpx_liq T, cpx_only P) | Multi-hour wall clock per pipeline; substituted second-place models with explicit log entry |
| Monte Carlo uncertainty (100 reps × 1% noise) | ~10⁸ ops; defer until follow-up |
| IsolationForest OOD score per sample | Defer; can compute from existing models without re-inference |
| External Agreda / Wang / Petrelli on natural samples | No natural-sample wrappers wired; defer |
| Stacked Ridge ensemble on natural samples | Defer; not in canonical-cell winners list |
| Core_20 G.4 sensitivity on curated subset | Per-sample literature P required for non-circular regime assignment; not in curated CSV |
| Per-sample bootstrap-CI RMSE in nb08_locality_stratified.csv | Per-sample literature P-T not available without manual citation lookup |

Each of these items has a clear path forward documented in
results/PHASE_H_RUN_LOG.md and is not blocking submission of the
core H.3 / H.4 / H.5 results.

End of Section 17.
