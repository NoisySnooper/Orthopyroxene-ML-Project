# v10 natural worldwide plan

**Status:** canonical spec for Phase H + Phase I natural-sample work and world map figures
**Author:** NQTa
**Date:** 2026-04-16 (revised 2026-04-18 for Phase G collision integration)
**Companion to:** `v10_master_plan.md`, `v10_figure_audit.md`

User priority: world map is visually impressive centerpiece. Natural samples give cross-mineral convergence check. Both mineral-specific maps required (opx, cpx, twopx).

---

## 0. Phase G boundary conditions (added 2026-04-18)

Phase G established a pre-registered per-regime claims framework
(`docs/preregistration/p_regime_preregistration.md`) and a two-axis honesty bar
(test-set bootstrap CI + 20-seed spread) for opx_liq. Phase H inference
on natural samples collides with that framework in six ways; resolutions
are embedded in sections 4, 5, 6 below:

1. **Self-referential regime bins.** Phase G bins use *true* P from
   ExPetDB. Natural samples have no ground truth, so per-regime RMSE
   claims on natural predictions are circular. **Resolution:** the
   two-axis honesty bar stays scoped to the calibration domain.
   Natural-sample visuals may stratify by predicted P regime (coloring,
   panel splits) but produce no RMSE claim per regime.
2. **G.4 bias correction is not applied by default.** Corrections were
   fit on train-OOF per true-P regime, validated on test. Applying them
   to natural samples requires regime labels from predicted P (circular)
   and extrapolates beyond the fit domain. **Resolution:** raw
   predictions are the primary artifact. A supplementary panel on
   curated localities (where literature P is known, so regime is
   assignable) shows the correction effect as a sensitivity.
3. **Canonical cell roster incomplete for cpx and twopx.** Phase G
   Chunks A-C ran for opx_liq only. `v10_optuna_best_params_cpx.json`
   and `v10_optuna_best_params_twopx.json` exist from Phase 3.3b but the
   Chunk A pre-registration and canonical-cell selection have not. H.0
   locks those before inference.
4. **Two-axis honesty bar only exists for opx_liq.** Manuscript claims
   about cpx or twopx natural-sample predictions cannot cite the
   Chunk C framework. **Resolution:** opx carries full Chunks A-C rigor;
   cpx and twopx are descriptive with test-set bootstrap CI only (no
   20-seed axis unless H.0c opts in).
5. **Per-locality min_n=20 floor.** Phase G registered n>=20 for
   per-regime claims. Curated localities with n<20 (likely Erta Ale,
   Spitsbergen, some xenolith sites) cannot carry bootstrap CI claims.
   **Resolution:** per-locality RMSE reported where n>=20; pooled by
   tectonic setting otherwise.
6. **ArcPL opx G.4 deferral may unblock.** Thermobar wiring for H.0a
   (classical P-T on natural samples) is the same wiring that was
   missing in G.4. **Resolution (2026-04-18):** ArcPL opx has n=197
   paired opx+liq compositions with literature P-T in
   `archive/pre_v10_rebuild_2026_04_16/results/nb04_arcpl_opx_liq_predictions_forest.csv`.
   Rather than a standalone G.4 probe, fold it into H.6 as a curated
   "ArcPL-opx" locality (scope = inference only, per user directive).
   The same Thermobar+canonical-model+G.4-correction pipeline that runs
   on the 15 curated sites runs here too. This closes H.0d without a
   separate script.

---

## 1. Scope

Three deliverable categories from Phase H + I:

1. **Natural sample inference** — run every trained model on large natural-sample datasets, produce per-sample P-T predictions with uncertainty
2. **Cross-mineral convergence** — where a sample has both opx and cpx analyzed, compare what opx-only, cpx-only, twopx, and universal models predict
3. **World map figures** — static (publication) + interactive (supplementary/repo), two panels each (tectonic setting + predicted T), separate per mineral

---

## 1.5 Revised Phase H execution plan (2026-04-18)

```
H.0  Prereqs (NEW, collision-driven)
  H.0a  Wire Thermobar for classical P-T inference on natural samples
  H.0b  Run Phase C canonical-cell selection for cpx_only / cpx_liq / twopx
  H.0c  (Optional) replicate Phase G Chunks A-C per-regime audit for cpx_liq
  H.0d  [RESOLVED 2026-04-18] ArcPL opx folded into H.6 as curated locality,
        no separate probe needed (see Section 0 collision 6)

H.1  Data pulls
  H.1a  Re-merge lat/lon into natural_opx_cleaned
  H.1b  Pull GEOROC cpx 2024-12
  H.1c  Pull GEOROC liquid/glass
  H.1d  Build twopx pairs by sample+citation match

H.2  Curated localities
  H.2a  Populate curated_localities.csv (15 localities)
  H.2b  Note per-locality n; flag any <20 for claims framework

H.3  Inference (raw predictions only, no G.4 corrections)
  H.3a  Run canonical opx_liq + opx_only on natural_opx
  H.3b  Run canonical cpx_liq + cpx_only on natural_cpx
  H.3c  Run canonical twopx on pairs
  H.3d  OOD via IsolationForest, MC uncertainty

H.4  Cross-mineral convergence
  H.4a  Per-pair delta_T, delta_P across models
  H.4b  Fe-Mg equilibrium flag (KD 0.95-1.23)
  H.4c  twopx vs opx-only+cpx-only averaged, benchmarked on curated

H.5  World maps
  H.5a  Static (Robinson, tectonic + predicted T panels)
  H.5b  Interactive (folium clusters)
  H.5c  Stratify colorbar visually by predicted-P regime (NO RMSE claims)

H.6  Validation vs curated localities
  H.6a  Per-locality RMSE where n>=20; pooled-setting RMSE otherwise
  H.6b  Supplementary panel: G.4-corrected vs raw on curated subset

H.7  Phase H self-audit + commit
```

Sections 2-12 below describe WHAT (data sources, map spec, deliverables);
this section describes WHEN (the ordered execution plan respecting Phase G).

---

## 2. Natural sample data sources

### 2.1 Opx: GEOROC SGFTFN 2024-12

Already staged. `data/natural/2024-12-SGFTFN_ORTHOPYROXENES.csv` (78,532 samples, lat/lon, full metadata).

**Phase H task 1:** re-merge lat/lon and metadata into cleaned `natural_opx_cleaned.csv` (53,050 samples). Produce `data/natural/natural_opx_with_coords.csv` with training-schema oxides + lat/lon + tectonic setting + citation + sample_name.

Tectonic setting counts (verified):

| Setting | n |
|---|---|
| Convergent margin | 31,662 |
| Intraplate volcanics | 13,934 |
| Rift volcanics | 2,311 |
| Continental flood basalt | 2,071 |
| Archean craton (+ greenstone belts) | 1,200 |
| Ocean island | 1,109 |
| Seamount | 504 |
| Oceanic plateau | 179 |
| Complex volcanic settings | 51 |
| Submarine ridge | 2 |

### 2.2 Cpx: GEOROC monthly release

Phase A pulls the equivalent 2024-12 CPX release from GEOROC. Expected filename pattern: `2024-12-{code}_CLINOPYROXENES.csv` or `2024-12-{code}_CLINOPYROXENES.txt`.

If GEOROC restructured their release format by 2026, `scripts/v10_pull_georoc_cpx.py` queries their current API.

Expected cpx n: 80,000-150,000 (cpx is more abundant in terrestrial volcanic rocks than opx).

Same metadata schema: lat/lon, tectonic setting, citation, sample name, rock name, oxides.

Apply same cleaning:
- Wo_cpx > 20 (pyroxene quad cut)
- Mg_num in [0.5, 0.95]
- Oxide total 99-101%
- Drop samples missing any of 9 required cpx oxides

Expected cleaned n: 40,000-80,000.

Output: `data/natural/natural_cpx_with_coords.csv`.

### 2.3 TwoPX: join by sample name + citation

Samples where both opx and cpx are analyzed from the same specimen. Match on:

- Same CITATION
- Same SAMPLE NAME
- Within reasonable lat/lon tolerance (0.01 deg)
- Same tectonic setting
- Rock name consistent

Expected n: 5,000-20,000 natural twopx pairs.

Output: `data/natural/natural_twopx_pairs.csv` with both mineral compositions side-by-side.

### 2.4 Liquid/glass compositions

Per user Q33. Pull GEOROC volcanic glass compositions for sample context. Used for:

- Identifying which natural opx or cpx samples have paired liquid from the same eruption (enables opx-liq and cpx-liq natural inference)
- Paleo-eruption P-T context for manuscript locality narratives

Not required for primary world maps but populates the opx-liq and cpx-liq predictions where applicable.

Output: `data/natural/natural_glass_with_coords.csv`.

---

## 3. Curated localities for literature P-T cross-check

Per user: "experimental data with known T and P only" for natural samples. Strict interpretation: curated sets with published thermobarometric estimates (not experimental T-P because these are natural rocks, but literature P-T estimates with stated uncertainty bounds).

**Candidate localities (~15 curated):**

**Arc volcanics (primary opx + cpx targets):**
1. Mount St. Helens (Cascades, USA) — Blundy & Cashman series
2. Kamchatka (Klyuchevskoy, Shiveluch) — Portnyagin et al.
3. Central Andes (Licancabur, Lascar) — multiple sources
4. Lesser Antilles (St. Kitts, Dominica) — Macdonald et al.
5. Aegean (Santorini, Nisyros) — Druitt et al.
6. Aleutians (Aniakchak, Okmok) — multiple

**Xenolith sites (opx + cpx cratonic + spinel lherzolite):**
7. Kilbourne Hole, NM (user DAC sample origin!) — Anthony & Titley
8. Spitsbergen — Amundsen et al.
9. Siberian craton — Griffin et al.
10. Kaapvaal craton — Nixon & Boyd
11. Kerguelen Plateau — Gregoire et al.

**OIB + intraplate:**
12. Kilauea (Hawai'i) — Helz & Thornber, Pietruszka
13. Iceland (Askja, Krafla) — Maclennan
14. Réunion (Piton de la Fournaise) — Boivin

**Experimental-P-T test site (if any locality has drill-core + gas-sensor T):**
15. Erta Ale (Ethiopia) — Field et al.

Each locality gets 10-30 samples, literature P-T with reported uncertainty.

**Phase H task 2:** write a `data/natural/curated_localities.csv` with columns:
- locality_name
- sample_name
- citation_pT (paper reporting P-T)
- T_estimate_C
- T_uncertainty_C
- P_estimate_kbar
- P_uncertainty_kbar
- method (e.g., "Putirka 2008 eq33", "Brey-Köhler 1990", "seismic Moho + geobarometry")
- notes

This is the gold set for validation. Manuscript Table X summarizes it.

---

## 4. Natural sample inference workflow

**Phase G collision note (see Section 0, items 1-2):** predictions are
stored *raw*. The G.4 piecewise P bias correction is NOT applied to
natural samples in this pass because its correction coefficients were fit
on train-OOF per *true* P regime, and natural samples have no ground
truth P. The corrected values appear only as an optional supplementary
panel on curated localities (Section 3 / H.6b), where literature P
assigns the regime without circularity.

Per pipeline, per sample in `natural_{mineral}_with_coords.csv`:

1. Build feature vector per model's expected input
2. Predict T and P with every canonical base + stacked model
3. Flag OOD via IsolationForest per T10
4. Monte Carlo uncertainty (100 reps, 1% oxide noise) per sample
5. Store to `results/nb08_natural_predictions_{mineral}.csv`

Columns per row:
- sample_id (CITATION + SAMPLE NAME + sequence)
- lat, lon
- tectonic_setting, location
- oxide values (for reproducibility)
- T_forest_pred, T_forest_lo, T_forest_hi
- T_boosted_pred, T_boosted_lo, T_boosted_hi
- T_stacked_pred, T_stacked_lo, T_stacked_hi
- T_catboost, T_lightgbm, T_elasticnet, T_mlp (similar)
- T_external_agreda, T_external_jorgenson, T_external_wang, T_external_petrelli (where applicable)
- T_putirka_classical (where inputs allow)
- Same 14 columns for P
- OOD score, MC uncertainty, analytical uncertainty
- flag_equilibrium (twopx Fe-Mg equilibrium test)

This one big CSV per mineral is the primary artifact. World map reads from it.

---

## 5. World map figure specification

### 5.1 Static map (publication)

**Tool:** matplotlib + cartopy.
**Projection:** Robinson (default for global-ocean-inclusive maps).
**Size:** 190 mm (double column), aspect ratio ~2:1.
**Background:** Natural Earth low-res coastlines + ocean color (`#f0f4f8` light blue).
**Grid:** 30° graticule, thin gray.
**Title:** "Global distribution of {mineral} samples from GEOROC 2024-12".

**Two panels side by side** per mineral per user Q34:

**Panel A (left):** colored by **tectonic setting**.
- Convergent margin: `#D55E00` (vermillion)
- Intraplate volcanics: `#CC79A7` (reddish purple)
- Rift volcanics: `#E69F00` (orange)
- Continental flood basalt: `#F0E442` (yellow)
- Archean craton: `#999933` (olive)
- Ocean island: `#56B4E9` (sky blue)
- Seamount: `#44AA99` (teal)
- Other: `#888888`
- Markers: small circles, alpha 0.4, size scaled to sample count per hex bin OR kernel density
- Legend: tectonic setting color chips

**Panel B (right):** colored by **predicted T from our best model** (stacked or primary).
- Colormap: `viridis` 600-1500 C
- Colorbar: vertical, with label "Predicted T (°C)" and tick labels
- Markers: same positions, size constant, alpha 0.6
- Annotation: point out the 15 curated localities with labels (sample name abbreviated + country code)

**Phase G collision note (see Section 0, item 1):** Panel B colors are a
*visualization* of predicted T, not a claim. Do NOT partition the map by
predicted-P regime and report per-regime RMSE: such claims are
self-referential on natural samples. If a supplementary map stratifies
by predicted-P regime (e.g. shallow_crustal / deep_crustal_MASH /
lithospheric_mantle / deeper_mantle), caption must explicitly note "no
ground truth P on natural samples; predicted-P regime is a visualization
stratum only."

**File output:**
- `fig_nb08_{mineral}_world_map_static.pdf`
- `fig_nb08_{mineral}_world_map_static.png`

Per mineral: opx, cpx, twopx. Six files per mineral pair (PDF + PNG, twin-panel figure).

### 5.2 Interactive map (supplementary + repo)

**Tool:** folium (leaflet.js backend).
**Projection:** Web Mercator (leaflet default).
**Base map:** OpenStreetMap tiles.

**Features:**
- Marker cluster (`folium.plugins.MarkerCluster`) to handle 50k+ markers without browser lag
- Marker color-coded by tectonic setting
- Popup per marker (click):
  - Sample ID + location
  - Oxide composition summary (5 key ratios)
  - All model P-T predictions in a table
  - Link to literature if curated locality
  - OOD flag
- Layer toggle: "by tectonic setting" / "by predicted T"
- Layer toggle: "show curated localities only"
- Search: locality name search box

**File output:** `fig_nb08_{mineral}_world_map_interactive.html`

Per mineral: three HTML files.

### 5.3 Combined three-mineral map (optional)

Overlay opx + cpx + twopx on single static map with transparent markers, showing where all three cover. Highlights the twopx subset as the most constrained samples.

Stretch figure — if compute and time allow.

---

## 6. Cross-mineral convergence analysis

### 6.1 Per-sample opx-cpx delta

For samples in `natural_twopx_pairs.csv`:

For each sample and each metric (T, P):
- `delta_T_opx_cpx_stacked = T_opx_stacked - T_cpx_stacked`
- `delta_P_opx_cpx_stacked = P_opx_stacked - P_cpx_stacked`

Similar for every (base model, external model).

### 6.2 Equilibrium flag

Fe-Mg exchange: `KD_FeMg_opx_cpx = (Fe_cpx * Mg_opx) / (Fe_opx * Mg_cpx)`.
If `KD_FeMg_opx_cpx` in `[0.95, 1.23]` (Putirka 2008 cpx-opx range), flag as "equilibrated." Else flag "non-equilibrium."

Hypothesis: equilibrated samples have smaller |delta_T| and |delta_P| across models. Tested in `nb08_convergence_by_kd.png`.

### 6.3 Twopx vs separate inference check

For the same natural sample:
- twopx model prediction: T_twopx, P_twopx
- opx-only prediction: T_opx_only, P_opx_only
- cpx-only prediction: T_cpx_only, P_cpx_only
- averaged: T_avg = (T_opx_only + T_cpx_only) / 2
- delta_twopx_vs_avg_T = T_twopx - T_avg

If twopx model is strictly better, |T_twopx - T_truth| < |T_avg - T_truth|. Tested on curated localities where truth is known.

---

## 7. Implementation artifacts

### 7.1 New modules

`src/world_map.py`:

```python
def build_static_world_map(
    df_predictions, mineral='opx', panel='tectonic_setting',
    out_path=None, projection='robinson'
): ...

def build_interactive_world_map(
    df_predictions, mineral='opx',
    curated_localities_df=None, out_path=None
): ...

def add_curated_locality_annotations(ax, curated_df): ...
```

`src/natural_inference.py`:

```python
def predict_all_models_on_natural(
    df_natural, mineral='opx',
    models_list=None, mc_reps=100, ood=True
):
    """Returns merged DataFrame with all model predictions + uncertainties."""
    ...
```

### 7.2 New scripts

`scripts/v10_pull_georoc_cpx.py`:
- Queries GEOROC for 2024-12 cpx release or current equivalent
- Applies same cleaning as opx
- Outputs `data/natural/2024-XX-GEOROC_CLINOPYROXENES.csv` raw + `natural_cpx_with_coords.csv` cleaned

`scripts/v10_natural_twopx_pairs.py`:
- Joins opx and cpx natural datasets on (CITATION, SAMPLE NAME)
- Outputs `natural_twopx_pairs.csv`

`scripts/v10_world_map_static.py`:
- Reads `results/nb08_natural_predictions_{mineral}.csv`
- Calls `src/world_map.build_static_world_map` for each panel
- Saves PDF + PNG

`scripts/v10_world_map_interactive.py`:
- Same but folium
- Saves HTML

---

## 8. NB08 structure (merged NB08 + NB08b)

```
# Phase 8.0: Setup
# Phase 8.1: Load curated localities
# Phase 8.2: Load natural opx, cpx, twopx datasets
# Phase 8.3: Run all model inference on natural opx samples
# Phase 8.4: Run all model inference on natural cpx samples
# Phase 8.5: Run all model inference on twopx pairs
# Phase 8.6: Run universal model inference (isolated section)
# Phase 8.7: Cross-mineral convergence analysis (per Section 6 above)
# Phase 8.8: Curated-locality validation (per Section 3 above)
# Phase 8.9: Produce 2-way comparison scatter plots per `v10_figure_audit.md` Section 2.7
# Phase 8.10: Produce static world maps (6 files per mineral: 3 panels × 2 formats)
# Phase 8.11: Produce interactive world maps (3 HTML)
# Phase 8.12: Produce locality-stratified summary figures and tables
# Phase 8.13: Export to `results/nb08_natural_predictions_*.csv`, `results/nb08_cross_mineral_agreement.csv`, tables
```

---

## 9. Timeline

Per master plan Phase H + I, revised 2026-04-18 to include H.0 prereqs:
- **H.0**: 1-2 days active + ~30 min compute (Thermobar wiring, canonical
  cpx/twopx cell selection, optional Chunks A-C replication for cpx_liq,
  un-defer G.4 ArcPL opx probe)
- **H.1 - H.7**: 2-3 days active + 1 h compute (inference over ~130,000
  natural samples across 12 models = ~90 min)
- **Phase I**: 2-3 days active (mostly figure polish)

Total Phase H: 3-5 days active, ~2 h compute.

---

## 10. Deliverables

- `data/natural/natural_opx_with_coords.csv`
- `data/natural/natural_cpx_with_coords.csv`
- `data/natural/natural_twopx_pairs.csv`
- `data/natural/natural_glass_with_coords.csv`
- `data/natural/curated_localities.csv`
- `results/nb08_natural_predictions_opx.csv`
- `results/nb08_natural_predictions_cpx.csv`
- `results/nb08_natural_predictions_twopx.csv`
- `results/nb08_cross_mineral_agreement.csv`
- `results/nb08_locality_stratified.csv`
- Figures per `v10_figure_audit.md` Section 2.7 (~20 figures including 6 static world maps and 3 interactive)

---

## 11. Risk register

| # | Risk | Mitigation |
|---|---|---|
| 1 | GEOROC cpx release unavailable or format changed | Fallback to web scraping UI; or use cpx from LEPR only + note limitation |
| 2 | Sample name matching fails (different citations use different naming) | Accept subset that matches cleanly; note reduced twopx n |
| 3 | World map too dense (100k markers) | Hex binning for density, sampling for interactive (cluster every 1000) |
| 4 | Curated locality P-T inconsistent across papers for same sample | Use most-cited source; alternate in supplementary |
| 5 | Inference run OOM on large CSVs | Chunk by tectonic setting, stream predictions |
| 6 | Folium interactive map >100 MB | Sample to representative 10k subset for interactive; full dataset in static |
| 7 | Cartopy installation issues on Windows | Use matplotlib Basemap as fallback; or use plotly.geo |

---

## 12. Aesthetic commitments

User: "I want it to be visually impressive."

Specific decisions:

- Use Robinson projection (classical, visually pleasing, familiar)
- Ocean color `#eef5fc` (light blue, not pure white)
- Land color subtle `#f0efe8` (warm gray)
- Coastlines `#444444` thin
- Markers alpha 0.4-0.6 to show density
- Colored-by-T uses viridis with midpoint highlighted at typical arc temperature (1100 C)
- Labels for curated localities: halo effect (white background small margin + black text) for legibility over dense marker clusters
- Title: sans-serif, 14 pt, bold
- Panel letters: top-left, 16 pt, bold "A" and "B"
- Colorbar: horizontal below maps, easy to read
- Minimal gridlines, just 30° spacing

Result should be submission-ready and look like a figure from Nature Geoscience rather than a Jupyter default.
