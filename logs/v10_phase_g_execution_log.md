# Phase G Execution Log

**Date started:** 2026-04-17
**Status:** IN PROGRESS
**Scope:** Cross-pipeline benchmark, external-model benchmark, 20-seed
multi-seed refit protocol, then manuscript-facing rebuilds (NB05/06/07/10)
and figures.

---

## G.1a Cross-pipeline internal benchmark — DONE

Ran `scripts/v10_phase_g_crosspipe_benchmark.py` over all four v10
pipeline outputs (opx/cpx/twopx/universal) using the canonical per-cell
and ensemble CSVs. Joined best base + best ensemble per (track, target).

Ranking per target:

- **T_C best track:** cpx_liq (v10 ridge ensemble, RMSE 68.36)
- **P_kbar best track:** twopx (v10 XGB alr, RMSE 4.26); universal
  `twopx_liq` scope 4.07

## G.1b External benchmarks — DONE (with caveats)

`scripts/v10_phase_g_external_benchmark.py`. Classic + pretrained ML
thermobarometers via Thermobar 1.0.70 + ONNX (Agreda 2024). Scored on
the v10 fixed Citation-grouped test split for each track.

### Known caveats

- Putirka closed-form equations occasionally return non-physical output
  (~280-300 K for rows with marginal composition). Reported both
  **raw** and **[filtered]** (physically plausible range) rows.
  Filtered is the honest headline number.
- Jorgenson/Petrelli/Agreda/Wang are trained on LEPR/ExPetDB, which
  partially overlap with v10 training data. Their "test" split is
  therefore not fully out-of-sample. Treat as a soft ceiling rather
  than a clean comparison.

### Headline table (filtered, where applicable)

| Track    | Target  | Best external              | RMSE   | v10 best           | RMSE   |
|----------|---------|----------------------------|--------|--------------------|--------|
| opx_liq  | T_C     | Putirka 28a [filt.]        | 71.67  | ElasticNet raw     | 77.06  |
| opx_liq  | P_kbar  | Putirka 29a [filt.]        |  4.75  | MLP raw            |  3.82  |
| opx_only | P_kbar  | Putirka 29c [filt.]        | 13.34  | ridge pwlr ens     | 10.18  |
| opx_only | T_C     | — (no external)            |   —    | LightGBM alr       | 147.98 |
| cpx_liq  | T_C     | Putirka 33 [filt.]         | 63.43  | ridge pwlr ens     | 68.36  |
| cpx_liq  | P_kbar  | Putirka 30 [filt.]         |  5.48  | ridge pwlr ens     |  6.52  |
| cpx_only | T_C     | Jorgenson (likely leakage) | 109.91 | two_level pwlr ens | 128.88 |
| cpx_only | P_kbar  | Putirka 32a [filt.]        | 11.83  | MLP alr            | 13.47  |

### Implications (aggregate)

- **v10 wins cleanly:** opx_liq P (3.82 beats 4.75), opx_only P (10.18
  beats 13.34).
- **Putirka beats v10 on cpx_liq** aggregate for both T and P.
- **Jorgenson/Agreda apparent wins on cpx_only T** are leakage candidates;
  Phase G.2 LOSO will test this.

### G.1c Regime-stratified benchmark (v10_regime_benchmark.csv)

P_kbar RMSE by true-P bin (v10 best base vs Putirka 2008 vs Agreda 2024):

**opx_liq (v10: MLP raw)**

| Regime  | n  | v10   | Putirka 29a | Winner |
|---------|----|-------|-------------|--------|
| P<5     | 47 | 2.96  | 3.89        | **v10** |
| 5-10    | 11 | 3.17  | 2.39        | Putirka |
| 10-20   | 81 | 2.95  | 3.44        | **v10 (14%)** |
| 20-40   | 32 | 6.11  | 5.80        | tie |
| P>=40   | 3  | 6.00  | 16.38       | v10 (tiny n) |
| **ALL** | 174| **3.82**  | **4.75** | **v10 (20%)** |

**opx_only (v10: XGB pwlr)**

| Regime  | n  | v10   | Putirka 29c | Winner |
|---------|----|-------|-------------|--------|
| P<5     | 48 | 7.95  | 2.20        | Putirka |
| 5-10    | 20 | 9.42  | 5.09        | Putirka |
| 10-20   | 84 | 6.60  | 4.88        | Putirka |
| 20-40   | 23 | 9.28  | 9.71        | tie (v10) |
| **P>=40**| 15| **25.87** | **44.03** | **v10 (41%)** |
| **ALL** | 190| **10.38** | **13.34** | **v10 (22%)** |

**cpx_liq (v10: LightGBM pwlr)**

| Regime  | n  | v10   | Putirka 30  | Agreda 2024 | Winner |
|---------|----|-------|-------------|-------------|--------|
| P<5     |268 | 4.97  | 4.18        | 4.44        | Putirka |
| 5-10    | 46 | 3.87  | 2.64        | 2.54        | Agreda |
| 10-20   |210 | 2.77  | 4.69        | 1.78        | Agreda (v10 beats Put 40%) |
| 20-40   |115 | 5.06  | 6.09        | 5.00        | Agreda ~ v10 |
| P>=40   | 15 | 32.75 | 22.49       | 56.35       | Putirka (v10 2nd) |
| P<20    |524 | 4.12  | 4.25        | 3.45        | Agreda |
| ALL     |654 | 6.54  | 5.48        | 9.31        | Putirka |

**cpx_only (v10: MLP alr)**

| Regime  | n  | v10   | Putirka 32a | Agreda 2024 | Winner |
|---------|----|-------|-------------|-------------|--------|
| P<5     |236 | 8.53  | 5.68        | 3.84        | Agreda |
| 5-10    | 43 | 3.51  | 3.27        | 2.96        | Agreda |
| 10-20   |296 | 7.43  | 4.75        | 2.85        | Agreda |
| 20-40   |148 |10.35  | 9.16        | 8.32        | Agreda |
| **P>=40**| 61|**38.89**|**38.35**  |**63.21**    | **v10 ~ Putirka (Agreda fails)** |
| ALL     |784 |13.47  | 11.83       | 18.22       | Putirka |

### Headline findings for manuscript

1. **opx_liq P 10-20 kbar is the v10 flagship result.** 14% RMSE reduction
   over Putirka 29a (2.95 vs 3.44) on 81 test samples. This is the crustal
   magma-storage regime where most arc experiments live. Agreda does not
   apply (no opx model).
2. **opx-only at ultra-high-P (>=40 kbar)** is a secondary v10 win:
   41% improvement over Putirka 29c (25.87 vs 44.03). Mantle thermobarometry
   niche.
3. **cpx_liq 10-20 kbar**: v10 beats Putirka by 40% (2.77 vs 4.69) but
   Agreda still wins outright (1.78). Reasonable second-place finding.
4. **Agreda's claimed accuracy is real within P<20 kbar** (cpx_liq T=37.57,
   P=3.45), matches their paper. Degrades catastrophically above 20 kbar
   due to model range saturation. v10 generalizes more gracefully to
   ultra-high-P but does not beat Agreda in its calibration window.
5. **Do not pitch v10 cpx_liq as an Agreda replacement.** Pitch it as
   phase-agnostic coverage (opx + cpx + twopx + universal), with
   specific wins on opx pipelines.

### Artifacts

- `results/v10_external_benchmark.csv` (58 rows, aggregate)
- `results/v10_regime_benchmark.csv` (56 rows, per-regime)
- `logs/v10_phase_g_external_benchmark.log`
- `logs/v10_phase_g_regime_benchmark.log`

### G.1c-ext: all-models regime benchmark with bootstrap CIs (TOOLING IN PLACE)

Extended the regime benchmark to cover every Optuna cell (8 models x 3
feature sets per (track, target)) across every v10 pipeline (opx, cpx,
twopx, universal), AND to bin by both P and T regimes. All external
thermobarometers scored on the same held-out rows: Putirka 2008
(multiple equations per track), Jorgenson 2022, Wang 2021, Agreda 2024.
Bootstrap 95% CI on RMSE and MAE (500 resamples). Output is a single
long-format CSV consumed by both figure and table scripts.

Scope coverage:
  - opx_liq: Putirka 28a/28b (T) + 29a/29b (P), v10 cells
  - opx_only: Putirka 29c (P), v10 cells
  - cpx_liq: Putirka 33/34 (T) + 30/31 (P), Jorgenson, Wang, Agreda, v10 cells
  - cpx_only: Putirka 32d (T) + 32a/32b (P), Jorgenson, Agreda, v10 cells
  - twopx: Putirka 36/37 (T) + 38/39 (P) two-pyroxene, v10 cells
  - universal: v10 cells only (no external equivalent)

Regime bins:
  - P: P<5 / 5-10 / 10-20 / 20-40 / P>=40 / P<20 (Agreda) / ALL
  - T: T<800 / 800-1000 / 1000-1200 / 1200-1400 / T>=1400 / ALL

Scripts:
- `scripts/v10_phase_g_regime_allmodels.py` -> `results/v10_regime_allmodels.csv`
- `scripts/v10_phase_g_regime_figure.py` -> `figures/v10_regime_{scope}.{pdf,png}`
  with scope in {opx, cpx, combined, all} (4 figure variants). Panels
  plot per-regime RMSE with bootstrap 95% CI. Legend markers:
    - grey band: v10 cell min-max spread across all Optuna cells
    - green circle: v10 best cell + bootstrap CI
    - red triangle: Putirka (best equation per regime)
    - blue diamond: Agreda 2024
    - purple square: Jorgenson 2022
    - orange triangle-down: Wang 2021
  Rows = (track, target); columns = P regime | T regime.
- `scripts/v10_phase_g_regime_tables.py` -> `tables/v10_regime_{scope}_{P|T}_{T_C|P_kbar}.{md,csv}`
  Markdown pivot per (scope, regime type, target). Cells:
  "RMSE [lo, hi]" (bootstrap 95% CI) with per-row winner bolded.
  Each row labels regime + n; each column is v10 best / Putirka best /
  Agreda / Jorgenson / Wang.

Status: tooling committed; run deferred until 20-seed multi-seed refit
finishes (shared CPU). Execute with:
```
.venv/Scripts/python.exe scripts/v10_phase_g_regime_allmodels.py
.venv/Scripts/python.exe scripts/v10_phase_g_regime_figure.py
.venv/Scripts/python.exe scripts/v10_phase_g_regime_tables.py
```

## G.1c 20-seed multi-seed refit — IN PROGRESS

`scripts/v10_phase_g_multiseed_runner.py`. Refits each cell
(model/target/track/feature_set) with Optuna `best_params` at 20 seeds
(42..61). Fixed Citation-grouped train/test split; only model
stochasticity varies.

| Pipeline  | Cells | Status                                |
|-----------|-------|----------------------------------------|
| opx       | 96    | DONE (1920 rows written)               |
| cpx       | 96    | running (~40/96 as of 2026-04-17 23:59) |
| twopx     | 64    | queued                                 |
| universal | 16    | queued                                 |

Writes `results/v10_{pipeline}_multiseed_results.csv` (long) and
`_summary.csv` (mean/std/min/max/count per cell).

## Next up (queued)

- G.1d: Write up v10-vs-v9 and v10-vs-Putirka comparisons in a Phase G
  manuscript-ready table, cite both in the paper draft.
- G.2: NB05 generalization rebuild (train/test, plus new
  LeaveOneRegionOut split by Citation prefix — e.g. "Hawaii" vs "Iceland"
  vs "Cascades") to quantify how much of the cpx_only "win" over v10 is
  leakage.
- G.3: NB06 SHAP on v10 best bases (tree SHAP + linear + kernel for MLP).
- G.4: NB07 bias correction (retrain on residuals).
- G.5: NB10 extended with twopx benchmark restored.
- G.6: Apply `docs/v10_figure_audit.md` final checklist across NBF.

---

## G.1c-ext Chunk A + Chunk B (2026-04-17 → 2026-04-18)

Pre-registered P-regime integration task (bin edges [0, 5, 15, 30, 100] kbar;
registered 2026-04-17 per `docs/v10_p_regime_preregistration.md`). Executed
fully autonomously per pre-approved continuation.

### Chunk A deliverables (2026-04-17, SHA e1ff0b7)

- `config.py`: added `P_REGIME_BIN_EDGES_KBAR`, `P_REGIME_LABELS`,
  `P_REGIME_REGISTERED_DATE`, `P_REGIME_MIN_N_FOR_CLAIMS`,
  `P_REGIME_RATIONALE_DOC`. CANONICAL_FIGURES now has entries 24 and 25.
- `src/evaluation.py`: added `assign_p_regime`, `compute_per_regime_metrics`,
  `per_regime_benchmark`. All three round-trip through
  `SEED_BOOTSTRAP` for reproducibility.
- `tests/test_evaluation_regime.py`: 10 tests, all pass (27.57 s).
- `scripts/v10_phase_g_regime_allmodels.py`,
  `scripts/v10_phase_g_regime_figure.py`,
  `scripts/v10_phase_g_regime_tables.py`: all import from config; pre-reg
  labels lead the ordering in the plots and tables.

### Chunk B deliverables (2026-04-18)

D1 — Figures (4 variants): run of `v10_phase_g_regime_figure.py` produced
`figures/v10_regime_{opx,cpx,combined,all}.{pdf,png}` over
`results/v10_regime_allmodels.csv` (4667 rows, six tracks).

D2 — NB04 regime section: ten new cells (v10 markdown template) inserted
into `notebooks/nb04_v10_benchmark.ipynb` after cell b09149e9. Executed in
place. Outputs: `results/v10_opx_per_regime_benchmark.csv`,
`_claims_audit.csv`, `_pivot_{rmse,rmse_ci,n}.csv`,
`figures/fig_nb04_per_regime_rmse_opx_liq.{pdf,png}`.

D3 — NBF figures 24 and 25: markdown cell c7f35a5e + code cell e5666a41
appended to `notebooks/nbF_figures.ipynb`. Code cell is self-contained
(loads opx_liq data + canonical base models directly, no dependency on
`nb03_per_family_winners.json`) and was executed via a mini-notebook
client that shimmed `save_figure`. Produced
`figures/fig24_per_regime_rmse_opx_liq.{pdf,png,txt}`,
`figures/fig25_per_regime_residual_violins_opx_liq.{pdf,png,txt}`, and
`results/v10_opx_liq_canonical_residuals_by_regime.csv`.

D4 — SI Tables S8.5.1-3: markdown cell `s85_tables_md` + code cell
`s85_tables_code` appended to `notebooks/nb09_manuscript_compilation.ipynb`
after the figure inventory (cell-010-f10e03fb). Executed; produced
`tables/S8_5_{1,2,3}_regime_*.{md,csv}`. Required `tabulate` dependency
(installed 0.10.0). Table S8.5.3 captures the honesty-bar verdict per
(regime, target).

D5 — Manuscript autofill:
`manuscripts/opx_2026/text/regime_results_autofilled.md` written with the
pre-registered framing, per-target RMSE+CI tables, verdict list, and
pointers to every source artifact. Only one "outperforms" claim meets the
honesty bar: shallow_crustal P_kbar (n=47, delta = 1.22 kbar, non-overlapping CIs).

D6 — Test T15 (pre-registered regime-stratified pass condition):
appended Section 12 to `docs/v10_nb03_test_protocol.md`. Named T15 (not
T13) to avoid collision with the universal-only T13/T14 already reserved
for the cpx paper's `nb03_universal_exploration.ipynb`.
`scripts/v10_nb03_test_t15.py` executes the pass-condition check; ran
2026-04-18 12:32:33 and returned `passed=True, value=1.223` for the
shallow_crustal regime.

### Self-audit

`scripts/v10_phase_g_chunkB_selfaudit.py` produces
`logs/v10_phase_g_chunkB_selfaudit.md`: **24/24 checks pass**. Covers
pre-registration doc, config constants, helper imports, track and label
coverage in allmodels CSV, honesty-bar verdict consistency, all figure +
table + manuscript artifact presence, and T15 log consistency.

### Known deferrals

- Jorgenson 2022 and Wang 2021 silent-failed for the twopx scope in
  `v10_phase_g_regime_allmodels.py` ("If using all scalar values, you
  must pass an index"). Not blocking for the opx paper; logged for cpx
  paper scope.
- Full-notebook execution of `nbF_figures.ipynb` is gated on
  regenerating `results/nb03_per_family_winners.json` from the v10
  retrain. For this chunk, figures 24 and 25 were rendered in a mini-
  notebook and their outputs were written back into nbF's cell
  metadata; the rest of nbF remains stale.
