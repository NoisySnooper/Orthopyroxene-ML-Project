# Phase H Execution Log

**Date started:** 2026-04-18
**Status:** IN PROGRESS (awaiting GEOROC Dataverse recovery for H.1b/c)
**Scope:** Natural-sample worldwide inference, cross-mineral convergence,
world maps, curated-locality validation. Plan doc
`docs/v10_natural_worldwide_plan.md`.

---

## H.0 Prereqs (collision-driven) — DONE

### H.0a Thermobar wiring for classical P-T on natural samples — DONE (b40f58b, 8d1e5cc)

Added opx_liq, opx_only, twopx Putirka wrappers to `src/external_models.py`
(`predict_putirka_opx_liq`, `predict_putirka_opx_only`, `predict_putirka_twopx`,
`compute_cpx_opx_kd_femg`). Created `src/thermobar_adapter.py` to centralize
the phase-frame builders that translate v10-native schemas to Thermobar's
`_Opx` / `_Liq` / `_Cpx` suffixed convention. Smoke-tested on opx-liq
test-set (median abs error ~23 C, ~2.8 kbar after dropping Thermobar
convergence failures).

### H.0b Canonical cell selection for cpx_only / cpx_liq / twopx — DONE (6e4b7a2)

Ran 20-seed aggregate winner selection on
`v10_cpx_multiseed_summary.csv` and `v10_twopx_multiseed_summary.csv`.
Output `results/v10_canonical_cells_h0b.{csv,json}` keyed by (track,
target):

| track | target | model | feature_set | seed_mean RMSE |
|---|---|---|---|---|
| cpx_only | T_C | ERT | pwlr | 127.02 |
| cpx_only | P_kbar | MLP | alr | 13.66 |
| cpx_liq | T_C | ERT | pwlr | 72.48 |
| cpx_liq | P_kbar | LightGBM | pwlr | 6.55 |
| twopx | T_C | ElasticNet | raw | 79.54 |
| twopx | P_kbar | XGB | alr | 4.26 |

Opx canonical cells (separately stored via Phase C aggregate winner
selection on `v10_opx_multiseed_summary.csv`):

| track | target | model | feature_set | seed_mean RMSE |
|---|---|---|---|---|
| opx_only | T_C | LightGBM | alr | 146.63 |
| opx_only | P_kbar | RF | pwlr | 10.35 |
| opx_liq | T_C | ElasticNet | raw | 77.06 |
| opx_liq | P_kbar | MLP | raw | 4.40 |

### H.0c Optional per-regime cpx_liq audit — DEFERRED

User opted out at plan sign-off. Phase G collision 4 resolution stands:
opx carries full per-regime Chunks A-C rigor; cpx and twopx are
descriptive with test-set bootstrap CI only.

### H.0d ArcPL opx G.4 deferral — RESOLVED (6e4b7a2)

Folded into H.6 as a curated locality. ArcPL opx (n=197) lives in
`archive/pre_v10_rebuild_2026_04_16/results/nb04_arcpl_opx_liq_predictions_forest.csv`
with paired opx+liq composition and literature P-T. Same
Thermobar+canonical+G.4 pipeline that runs on the 15 curated sites runs
here too, so a standalone probe is redundant.

---

## H.1 Data pulls — PARTIAL

### H.1a Re-merge lat/lon + K2O + Fe split into natural opx — DONE (e966a67)

`scripts/v10_phase_h1a_opx_with_coords.py` rebuilt the cleaned file
preserving lat/lon (53,023/53,050 samples coordinated), K2O, and
FeO/Fe2O3 split that the original prep script dropped. Output
`data/natural/natural_opx_with_coords.csv` (53,050 x 26).

Tectonic setting counts match docs exactly.

### H.1b GEOROC cpx 2024-12 — BLOCKED (retry scheduled 22:47)

Attempted via pygeoroc 2.0.0 (`georoc download`). Upstream Göttingen
Dataverse (DOI 10.25625/SGFTFN) returned HTTP 500 directly from curl.
STOPPED per user constraint — no automatic retry.

Scheduled one-shot retry via CronCreate at 22:47 local (job
49d1e874). Retry probes the server first, then proceeds through
D3-D7 of the user's provided H.1b prompt if HTTP 200, else re-pauses
for user direction.

Pre-staged `scripts/v10_georoc_cpx_schema_diff.py` (D4 diagnostic);
untracked; retry will include it in the H.1b commit bundle.

### H.1c GEOROC liquid/glass — BLOCKED (same upstream)

### H.1d twopx pairs — BLOCKED (needs H.1b)

---

## H.2 Curated localities CSV scaffold — PARTIAL (8712197)

Populated `data/natural/curated_localities.csv` (15 sites) with bounding
boxes, expected P-regime, approximate T-P ranges from literature.
Diagnostic script counts GEOROC samples per bbox:

| | n >= 20 floor | |
|---|---|---|
| Cleared | 11 | Mount St. Helens, Kamchatka, Central Andes, Lesser Antilles, Aegean, Aleutians, Spitsbergen, Siberian craton, Kaapvaal craton, Kerguelen Plateau, Iceland |
| Below | 4 | Kilbourne Hole (n=0; bbox too tight), Kilauea (n=2), Reunion (n=0; opx rare), Erta Ale (n=0; opx rare) |

User review pending; bboxes and expected ranges approximate.

---

## H.3 Inference — PARTIAL

### H.3a opx_only — DONE (7d6c727)

Applied canonical opx_only models to all 53k natural opx samples.
`results/v10_natural_opx_opx_only_predictions.csv` adds
`pred_T_C_opx_only`, `pred_P_kbar_opx_only`, `ood_isoforest_score`,
`ood_flag`.

Per-tectonic sanity (T/P means):

- Convergent margin 1014 C / 6.0 kbar  (shallow crustal arc magmas)
- Intraplate 1206 C / 17.6 kbar  (mantle melts)
- Archean craton 1170 C / 12.6 kbar  (lithospheric mantle)
- Continental flood basalt 1154 C / 18.9 kbar  (OOD 26%)
- Ocean island 1219 C / 19.0 kbar

Descriptive per-regime summary (8fb6e90):
`results/v10_phase_h3a_regime_summary.csv`. Per Phase G collision 4,
no RMSE claims on natural samples.

### H.3a opx_liq — BLOCKED (H.1c glass)

### H.3b cpx_only + cpx_liq — BLOCKED (H.1b)

### H.3c twopx — BLOCKED (H.1d)

### H.3d MC uncertainty — DEFERRED

Canonical artifacts are single-seed. Adding 20-seed spread on natural
samples requires retraining or aggregating per-seed multiseed
predictions; deferred as follow-up.

---

## H.4 Cross-mineral convergence — BLOCKED (H.3b/c)

---

## H.5 World maps — PARTIAL

### H.5a opx static world map — DONE (36fe52d)

`figures/fig_h5a_opx_world_map.png`. Two-panel Robinson projection:

- Panel A: tectonic setting categorical (10 classes)
- Panel B: predicted T_C (magma colormap, OOD dimmed)

Cartopy 0.25.0 installed this session with user approval.

### H.5b interactive folium — DEFERRED

Folium not installed. Can add after opx+cpx+twopx panels integrate.

### H.5c regime colorbar stratification — DEFERRED

---

## H.6 Validation vs curated localities — BLOCKED

Per-locality RMSE computation pending user review of
`curated_localities.csv` expected T-P ranges (current values are
approximate).

---

## H.7 Phase H self-audit + commit — NOT STARTED
