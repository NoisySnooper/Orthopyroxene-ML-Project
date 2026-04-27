# Phase H natural worldwide run log

Branch: phase_h_natural_worldwide
Start SHA: 1f39749e1a15f394c29d4fd96fdb02bca892af6d
Start UTC: 2026-04-25T00:00:00Z (session start)
Activation prompt: prompts/tabpfn_and_figures_rework/PHASE_H_NATURAL_WORLDWIDE_ACTIVATION.md

## Pre-run audit (2026-04-25T00:00:05Z)

- archive/pre_v10_rebuild_2026_04_16/results/nb04_arcpl_opx_liq_predictions_forest.csv present (151 KB) [halt #6 cleared]
- src/external_models.py present, predict_putirka_classical_natural() ABSENT [H.0a work needed]
- src/bias_correction.py present
- results/cpx_multiseed_summary.csv present
- results/twopx_multiseed_summary.csv ABSENT [H.0b will need fallback for twopx canonical cell]
- data/natural/natural_opx_with_coords.csv present, 53,051 data rows [H.1a essentially complete; verify schema]
- data/natural/curated_localities.csv present, 15 locality boundaries (different schema from prompt)
- data/natural/georoc_cache/ EMPTY
- data/natural/2024-12-SGFTFN_ORTHOPYROXENES.csv present (78,552 rows, manually staged)
- .venv-tabpfn present
- cartopy 0.25.0 + folium 0.20.0 available in .venv

## GEOROC connectivity probe (2026-04-25T00:00:30Z)

Probed candidate endpoints:
- https://api.georoc.eu/ -> SSL cert verify failed (self-signed)
- https://api.georoc.eu/queries -> 404 (with TLS verify off)
- https://api.georoc.eu/openapi.json -> 404
- https://api.georoc.eu/v1/datasets -> 404
- https://api.georoc.eu/queries/datasets -> 404
- https://georoc.eu/ -> 200 (HTML web UI; not a queryable JSON API)
- https://georoc.eu/api/ -> 404
- https://georoc.eu/datasets -> 404
- https://georoc.eu/v2/api/ -> 404
- https://api.georoc.org/ -> connection refused
- https://georoc2.gfz-potsdam.de/ -> DNS resolution failed
- https://georoc.gfz-potsdam.de/ -> DNS resolution failed
- https://georoc.mpch-mainz.gwdg.de/ -> 200 (legacy MPI HTML web UI; not a queryable JSON API)

No reachable GEOROC programmatic endpoint discovered. The user-asserted "GEOROC back online" claim is not consistent with what this sandbox can reach. Halt #1 ("GEOROC API returns < 30,000 raw cpx rows in H.1b") will fire when H.1b is attempted.

Proceeding with H.0 work (Thermobar wiring + canonical-cell lock) which is locally executable, then H.1a verification, then H.1b attempt + halt-report.

## Checkpoints

- 2026-04-25T00:01:00Z (H.0a) predict_putirka_classical_natural() and predict_putirka_cpx_only() added to src/external_models.py
- 2026-04-25T00:01:30Z (H.0a) tests/test_phase_h_external_models.py PASS: 11/11 equations finite on 5-row synthetic input
- 2026-04-25T00:02:00Z (H.0b) docs/preregistration/canonical_cells_cpx_twopx.md locked
  - cpx_liq T_C: TabPFN/raw 69.96+/-0.98
  - cpx_liq P_kbar: LightGBM/pwlr 6.55+/-0.05
  - cpx_only T_C: ERT/pwlr 127.02+/-0.31
  - cpx_only P_kbar: TabPFN/raw 13.42+/-0.42
  - twopx T_C: ElasticNet/raw 79.54+/-0.00 (deterministic estimator)
  - twopx P_kbar: XGB/alr 4.26+/-0.03
- 2026-04-25T00:02:15Z (H.0c) deferred per activation prompt default (skip Chunks A-C replication for cpx_liq)
- 2026-04-25T00:02:30Z (H.0d) ArcPL opx folded into H.6 per plan Section 0 collision 6; archive file present (151 KB), no separate work
- 2026-04-25T00:03:00Z (H.1a) verified data/natural/natural_opx_with_coords.csv: 53,050 rows in [50000, 53050], SHA256 b1969ed8...8ab7, 53,023 rows have lat/lon and tectonic setting
- 2026-04-26T23:53:54Z (H.1b) caveman: H.1b start. need GEOROC cpx 2024-12 dataset
- 2026-04-26T23:53:54Z (H.1b) caveman: legacy_static_zip HTTPError 404
- 2026-04-26T23:53:55Z (H.1b) caveman: legacy_static_csv HTTPError 404
- 2026-04-26T23:53:55Z (H.1b) caveman: api_v1_dataset HTTPError 404
- 2026-04-26T23:53:56Z (H.1b) caveman: api_v2_dataset HTTPError 404
- 2026-04-26T23:53:56Z (H.1b) caveman: api_query HTTPError 404
- 2026-04-26T23:53:57Z (H.1b) caveman: georoc_eu_static HTTPError 404
- 2026-04-26T23:53:57Z (H.1b) caveman: every endpoint failed. halt #1 fires
- 2026-04-26T23:53:57Z (H.1b) caveman: halt report at results\HALT_REPORT_PHASE_H.md

## Phase H halt summary (2026-04-26T23:55:00Z)

Halt fired: condition #1 (GEOROC API returned < 30,000 raw cpx rows).

**Completed before halt** (committed on branch phase_h_natural_worldwide):
- H.0a: src/external_models.py extended with predict_putirka_classical_natural() and predict_putirka_cpx_only(); 5-row smoke test 11/11 PASS
- H.0b: docs/preregistration/canonical_cells_cpx_twopx.md locked with 6 canonical cells (4 cpx, 2 twopx)
- H.0c: deferred per activation prompt default
- H.0d: noted resolved (ArcPL opx archive file present, folded into H.6 plan)
- H.1a: existing data/natural/natural_opx_with_coords.csv verified at 53,050 rows
- H.1b: scripts/v10_pull_georoc_cpx.py written and executed, all 6 endpoint candidates returned HTTPError 404 or DNS/TLS failure, halt #1 wrote results/HALT_REPORT_PHASE_H.md and exited 1

**NOT started** (blocked by halt):
- H.1c (glass), H.1d (twopx pairs)
- H.2 (curated localities — note: existing data/natural/curated_localities.csv has 15 locality boundary rows in a different schema than H.2a requires; H.2 not attempted)
- H.3 (inference — blocked because no cpx data, no twopx pairs)
- H.4 (cross-mineral convergence — blocked)
- H.5 (world maps — blocked, also no cpx data for the cpx map; opx-only map could be partial)
- H.6 (validation vs curated localities — blocked)
- H.7 (audit + manuscript integration — blocked because no data to audit and no §4.8 numbers to write)

Branch state at halt is self-consistent: H.0 prerequisites complete, halt artifacts present, no partial-state corruption of natural-sample CSVs.

End of run log.

- 2026-04-27T00:05:15Z (H.1b) caveman: H.1b start (Dataverse Native API). target dataset=doi:10.25625/SGFTFN
- 2026-04-27T00:05:15Z (H.1b) caveman: GET https://data.goettingen-research-online.de/api/datasets/:persistentId/?persistentId=doi:10.25625/SGFTFN
- 2026-04-27T00:05:16Z (H.1b) caveman: target file_id=118288 name='2024-12-SGFTFN_CLINOPYROXENES.csv' size=338.4 MB
- 2026-04-27T00:05:16Z (H.1b) caveman: stream-GET https://data.goettingen-research-online.de/api/access/datafile/118288 (expect ~338.4 MB)
- 2026-04-27T00:07:18Z (H.1b) caveman: download done. 338.4 MB in 54138 chunks. SHA256[:12]=5598c3bd0b92
- 2026-04-27T00:07:18Z (H.1b) caveman: read raw CSV (338.4 MB)
- 2026-04-27T00:07:26Z (H.1b) caveman: raw row count = 216809
- 2026-04-27T00:07:27Z (H.1b) caveman: rows after CLINOPYROXENE filter: 207031 (from 216809)
- 2026-04-27T00:07:28Z (H.1b) caveman: rows with all 9 oxides: 129279
- 2026-04-27T00:07:28Z (H.1b) caveman: rows after oxide total in [99,101]: 99063
- 2026-04-27T00:07:28Z (H.1b) caveman: rows after Mg# in [0.5,0.95]: 93899
- 2026-04-27T00:07:28Z (H.1b) caveman: rows after Wo > 20: 93163
- 2026-04-27T00:07:28Z (H.1b) caveman: cleaned row count = 93163
- 2026-04-27T00:07:29Z (H.1b) caveman: wrote natural_cpx_with_coords.csv (93163 rows, SHA256[:12]=31e608dc7f75)
- 2026-04-27T00:07:29Z (H.1b) caveman: cleaned n 93163 outside expected band [40000, 80000]; soft warning, not a halt
- 2026-04-27T00:09:12Z (H.1d) caveman: H.1d start. join opx + cpx natural sets
- 2026-04-27T00:09:13Z (H.1d) caveman: loaded opx=53050 cpx=93163
- 2026-04-27T00:09:13Z (H.1d) caveman: keyed opx=53050 cpx=93163
- 2026-04-27T00:09:13Z (H.1d) caveman: post-merge n_pre_filter=1521399
- 2026-04-27T00:09:15Z (H.1d) caveman: after lat/lon filter (0.01 deg): 1513457
- 2026-04-27T00:09:18Z (H.1d) caveman: after rock-name consistency: 1513160
- 2026-04-27T00:09:35Z (H.1d) caveman: wrote natural_twopx_pairs.csv (1513160 pairs, SHA256[:12]=4313c9c2351b)
- 2026-04-27T00:09:35Z (H.1d) caveman: top tectonic settings in pairs:
  CONVERGENT MARGIN: 1351780
  INTRAPLATE VOLCANICS: 64024
  OCEAN ISLAND: 47778
  RIFT VOLCANICS: 24166
  CONTINENTAL FLOOD BASALT: 17026
  SEAMOUNT: 5449
  ARCHEAN CRATON (INCLUDING GREENSTONE BELTS): 2788
  COMPLEX VOLCANIC SETTINGS: 121
- 2026-04-27T00:10:09Z (H.1d) caveman: H.1d start. join opx + cpx natural sets
- 2026-04-27T00:10:10Z (H.1d) caveman: loaded opx=53050 cpx=93163 (grain level)
- 2026-04-27T00:10:10Z (H.1d) caveman: keyed (per-sample medians) opx=11202 cpx=20278
- 2026-04-27T00:10:10Z (H.1d) caveman: post-merge n_pre_filter=6965
- 2026-04-27T00:10:10Z (H.1d) caveman: after lat/lon filter (0.01 deg): 6244
- 2026-04-27T00:10:10Z (H.1d) caveman: after rock-name consistency: 6234
- 2026-04-27T00:10:10Z (H.1d) caveman: wrote natural_twopx_pairs.csv (6234 pairs, SHA256[:12]=92dfcc09d833)
- 2026-04-27T00:10:10Z (H.1d) caveman: top tectonic settings in pairs:
  INTRAPLATE VOLCANICS: 3894
  CONVERGENT MARGIN: 1129
  RIFT VOLCANICS: 482
  CONTINENTAL FLOOD BASALT: 341
  OCEAN ISLAND: 294
  ARCHEAN CRATON (INCLUDING GREENSTONE BELTS): 36
  SEAMOUNT: 35
  COMPLEX VOLCANIC SETTINGS: 18
