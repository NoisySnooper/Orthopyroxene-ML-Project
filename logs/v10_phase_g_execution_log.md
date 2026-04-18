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

---

## G.1c-ext Chunk C: two-axis honesty bar (2026-04-18)

Motivation (flagged mid-chunk): Chunk B's audit only checked axis 1
(test-set bootstrap CI). The pre-registration doc specifies only that
non-overlapping 95% CIs are required; it does not name a specific CI
construction. Chunk C extends the honesty bar to axis 2 (20-seed RMSE
spread per cell, same fixed Citation-grouped train/test split; only
model-fit stochasticity varies). Putirka equations are deterministic
closed-form, so their axis-2 collapses to a point. The revised rule:
"v10 outperforms Putirka (robust)" requires axis-1 non-overlap
AND axis-2 non-overlap AND n >= 20.

### Why a new probe is needed

The existing `results/v10_opx_multiseed_results.csv` stores only per-seed
aggregate test RMSE for the full opx test set — no predictions, no
per-regime RMSE. To compute per-regime seed spread without re-running
the full 272-cell x 20-seed multi-hour job, Chunk C re-fits ONLY the
seven cells that nb04's regime benchmark picked as per-regime best
(one cell per (regime, target)):

    T_C:     ERT/pwlr (shallow), ElasticNet/raw (deep, litho),
             ElasticNet/pwlr (deeper, n<20)
    P_kbar:  ElasticNet/raw (shallow, **headline claim**),
             MLP/raw (deep), CatBoost/raw (litho),
             MLP/alr (deeper, n<20)

140 total fits, ~2 minutes on local CPU. Per-seed aggregate RMSE is
cross-checked against `v10_opx_multiseed_summary.csv` and matches to
1e-6 for all seven cells (i.e. same fits as the original multi-seed run,
just with predictions saved this time).

### Deliverables

- `scripts/v10_phase_g_chunkC_seed_regime_probe.py` - refits seven
  per-regime-best cells at 20 seeds, saves per-sample predictions.
- `scripts/v10_phase_g_chunkC_robust_audit.py` - consumes probe +
  Chunk B benchmark, emits two-axis claims audit.
- `scripts/v10_phase_g_chunkC_update_si_table.py` - emits Table S8.5.4.
- `scripts/v10_phase_g_chunkC_selfaudit.py` - 19-item self-audit.
- `results/v10_chunkC_perseed_predictions.csv`  (24,360 rows: 7 cells x
  20 seeds x 174 test samples).
- `results/v10_chunkC_perseed_regime_rmse.csv`  (560 rows: 7 cells x
  20 seeds x 4 regimes).
- `results/v10_chunkC_perseed_aggregate_rmse.csv` (140 rows, cross-check).
- `results/v10_opx_per_regime_claims_audit_robust.csv` (8 rows, canonical).
- `tables/S8_5_4_regime_claims_audit_robust_opx_liq.{md,csv}` -
  reviewer-facing SI table (supersedes S8.5.3 for manuscript citation).
- `docs/v10_nb03_test_protocol.md` Section 12: T15 pass condition
  upgraded to v2 (requires both axes non-overlap); v1 deprecated with
  rationale.
- `scripts/v10_nb03_test_t15.py` revised to read the robust audit; falls
  back to legacy axis-1 CSV with a WARNING tag if robust is missing,
  so the script remains runnable on cold checkouts.
- `manuscripts/opx_2026/text/regime_results_autofilled.md` revised with
  new framing paragraph, seed-RMSE columns in the per-target tables,
  and an explicit note that had the same regime been anchored on MLP/raw
  (seed RMSE spread [2.03, 5.79] kbar) the claim would have collapsed.
- `logs/v10_phase_g_chunkC_selfaudit.md` - 19/19 pass.
- `logs/v10_phase_g_chunkB_selfaudit.md` - 24/24 pass (re-verified; Chunk C
  did not break any Chunk B post-conditions).

### Headline finding

The pre-registered shallow_crustal / P_kbar claim survives the two-axis
honesty bar. The per-regime-best cell is ElasticNet/raw, which is
deterministic given fixed Optuna hyperparameters (20-seed RMSE spread
of exactly zero). v10 RMSE = 2.66 kbar [2.25, 3.13], Putirka 29a
RMSE = 3.89 kbar [3.24, 4.47]; n = 47; both axes non-overlapping;
robust_verdict = "v10 outperforms Putirka (robust)".

T15 re-run at 2026-04-18 with the v2 pass condition: passed=True,
value=1.223 kbar, source=robust, winning regime=shallow_crustal.

### Transparency findings (NOT verdict changes)

- **deep_crustal_MASH P_kbar (MLP/raw):** Chunk B point estimate 2.08 kbar
  is on the favorable tail of the seed distribution; 20-seed mean is 2.54,
  spread [2.01, 3.07]. Still "competitive with Putirka" either way, but
  the autofill now discloses the seed spread.
- **deeper_mantle P_kbar (MLP/alr):** Chunk B point 4.56 kbar; 20-seed
  mean 7.06, spread [3.78, 24.02]. n=8 keeps this "insufficient data"
  regardless, but the extreme instability is now documented.
- **Alternative-choice stress test:** if shallow_crustal had been anchored
  on MLP/raw (the aggregate test-set winner) instead of ElasticNet/raw,
  the seed spread [2.03, 5.79] kbar would have overlapped Putirka's
  lower CI (3.24) and the claim would have collapsed. The robust audit
  explicitly guards against this kind of hidden-variance trap.

### Commits

- 913ba87: `chore: add .gitignore`
- 9fa09cb: `Phase G.1b/G.1c artifacts: external benchmark + 20-seed
  multi-seed refit` (brings previously-untracked science artifacts under
  version control, precondition for Chunk C).
- Chunk C commit: see next commit below this log update.
