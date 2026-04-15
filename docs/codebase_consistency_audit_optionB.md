# Part 2: Codebase consistency audit — Option B pre-implementation

**Date:** 2026-04-15
**Scope:** Every mention of ArcPL sample counts (n=197 / 204 / 185 / 394 / 28 / 178 / 166)
and Putirka headline RMSE numbers (118.7, 86.3, 50.1, 64.6, 47.4, 45.7, 70.05) in the live
project tree (notebooks/, scripts/, src/, docs/, top-level *.md, *.py). Logs and archives
excluded because they are frozen artifacts, not authoritative documents.

## Purpose

Option B implementation will apply Thermobar's built-in Putirka 2008 Kd Fe-Mg filter to
the nb04 Part 3 ArcPL set, producing an equilibrated subset of **n=166** (84.3% of 197).
Before editing anything, this audit catalogs every number the rewrite will either match,
update, or invalidate, so no stale reference leaks into the manuscript.

## 1. Classification of live references

| Symbol | File:line | Context | Classification |
|---|---|---|---|
| n=197 | [notebooks/nb04_putirka_benchmark.ipynb](notebooks/nb04_putirka_benchmark.ipynb#L2030) cell 27 | Figure subtitle "Row 1: ML on ALL samples" | **tight** — canonical Part 3 scope after full cleaning pipeline. Option B keeps this denominator. |
| n=197 | [notebooks/nb07b_arcpl_bias_probe.ipynb](notebooks/nb07b_arcpl_bias_probe.ipynb#L13) markdown | Input size declaration | **tight** — correct for the forest predictions file. |
| n=197 | [scripts/audit_putirka_paths.py:139](scripts/audit_putirka_paths.py#L139) | Header of head-to-head table | **tight** — audit-only script, matches ArcPL Part 3. |
| n=197 | [scripts/audit_optionB_prediction.py:19](scripts/audit_optionB_prediction.py#L19) | Comment tagging the rebuild | **tight** — the Option B preflight script. |
| n=197 | [scripts/patch_v8_nb04_putirka_vs_ml_fig.py:146](scripts/patch_v8_nb04_putirka_vs_ml_fig.py#L146) | Patch script for nb04 cell subtitle | **tight** — matches nb04 cell 27. |
| n=204 | [notebooks/nb09_manuscript_compilation.ipynb](notebooks/nb09_manuscript_compilation.ipynb#L1098) cell 22 | Figure caption: `'ArcPL external validation (n=204)'` | **STALE** — refers to the pre-cleaning ArcPL scope (cell 19 of old nb04b). The actual image (`fig09_arcpl_forest_vs_boosted.png`) was produced on the looser scope. Option B will make this inconsistent with nb04's n=197 (and the forthcoming n=166 equilibrated row). Needs a Citation- or scope-aware rewrite. |
| n=204 | [scripts/patch_v8_nb09_fixes.py:61](scripts/patch_v8_nb09_fixes.py#L61) | Upstream source for the nb09 caption | **STALE** — same origin. Fix upstream here, then re-patch nb09. |
| n=197 / n=185 / n=28 / n=394 / 118.7 / 86.3 / 50.1 / 47.4 | [docs/putirka_inconsistency_audit.md](docs/putirka_inconsistency_audit.md) | The audit that explained the gap | **ambiguous in isolation, consistent as a whole** — this doc deliberately compares both scopes. No edit needed when Option B lands; the doc remains a valid historical reconciliation. Add a pointer from the Option B prediction doc. |
| n=197 | [docs/putirka_kd_filter_lookup.md:107](docs/putirka_kd_filter_lookup.md#L107) | "Verified behavior on ArcPL (n=197)" | **tight** — Part 1 of this Option B audit. |

No other live-tree files mention the target numerals. Frozen log files
(`logs/pipeline_resume_*.log`, `logs/run_all_v7_resume.log`, `logs/nb03c_training.log`) and
archived executed notebooks still carry the 118.74 C / n=28 / n=204 numbers, but those
trees are snapshots of prior runs and are not re-executed, so they do not need updates.

### Headline-RMSE references in the live tree

- `45.67`, `70.05`, `118.74`: only in `logs/*.log` and `archive/**`. No live file cites
  them. Safe.
- `50.1`, `47.4`: only inside `docs/putirka_inconsistency_audit.md`. Audit-only context.
- `64.6` and `2.72`: appear in executed notebooks under
  `notebooks/executed/nb07b_arcpl_bias_probe_executed.ipynb` (cell output). Those are
  regenerated per run; a re-execution of nb07b after Option B will refresh them if scope
  changes. Mark as **regenerate-on-rerun**.

## 2. Putirka opx-liq invocations in the live tree

### 2a. `notebooks/nb04_putirka_benchmark.ipynb` (cell 27, the `_pvm_` cell)

```python
pt.calculate_opx_liq_press_temp(
    opx_comps=_pvm_opx_in,
    liq_comps=_pvm_liq_in,     # Fe3Fet_Liq = 0.0 column included
    equationT='T_Put2008_eq28a',
    equationP='P_Put2008_eq29a',
    # NO eq_tests=True --- returns minimal columns only
)
```

- **Kd filter applied:** none (inside the Thermobar call). The only Kd-like filter is
  upstream in Part 3's cleaning pipeline: `config.KD_FEMG_MIN/MAX = [0.23, 0.35]`, the
  user-defined narrow range, which retained 197 rows.
- **Downstream consumers:**
  - `results/nb04_arcpl_opx_liq_predictions_forest.csv` (n=197, ML predictions only)
  - `figures/fig_nb04_putirka_vs_ml_arcpl.{png,pdf}` (the 3×2 figure)
  - Metrics printed in-cell: Putirka T RMSE=50.1 C, P=3.60 kbar on n=185 converged.
- **Option B impact:** needs `eq_tests=True` added, and a new "Row 3" (or replacement
  middle row) that filters on `Kd Eq (Put2008+-0.06) == 'Y'` before computing the RMSE.

### 2b. `scripts/audit_putirka_paths.py` (audit-only, not manuscript-facing)

Calls both `calculate_opx_liq_temp` (one-sided Path A) and
`calculate_opx_liq_press_temp` (iterative Path B) on the n=197 set. **No eq_tests.**
Not edited by Option B — retained as evidence of the two-path reconciliation.

### 2c. `scripts/audit_optionB_prediction.py` (audit-only, new)

Already calls `calculate_opx_liq_press_temp(..., eq_tests=True)` on the n=197 set. This
is the reference implementation that Part 3 of the preflight report consumes.

### 2d. `notebooks/executed/nb04b_lepr_arcpl_validation_executed.ipynb` (archive-adjacent)

Frozen executed copy of the old nb04b. Contains the 86.3% / n=28 / 118.74 C output. Does
not need changes (it is superseded by Part 3 of nb04). Already noted in the supersession
stub `archive/nb04b_superseded_by_nb04_part3_part4_part5.ipynb`.

### 2e. Other opx-liq matches (library-internal, not project code)

The remaining `calculate_opx_liq_*` hits are in
`data/external/thermobar_examples/Thermobar/**` — the vendored Thermobar library itself
and its example notebooks. Not project code. Listed here only for completeness.

## 3. What Option B will change

### 3a. Must update

| Target | Current | After Option B |
|---|---|---|
| nb04 cell 27 Putirka call | no `eq_tests` | add `eq_tests=True`, keep same equationT/P |
| nb04 cell 27 middle-row figure scope | n=185 (converged only) | n=166 (eq AND converged); add subtitle `Kd Eq (Put2008+-0.06)=Y` |
| nb04 cell 27 RMSE verdict print | "Putirka T=50.1 on n=185" | "Putirka T=50.3 on n=166 equilibrated subset" |
| [scripts/patch_v8_nb09_fixes.py:61](scripts/patch_v8_nb09_fixes.py#L61) caption | `n=204` | the applicable scope (likely `n=197` for the forest/boosted side-by-side, since that figure uses forest predictions on the cleaned ArcPL — verify before patching) |
| [notebooks/nb09_manuscript_compilation.ipynb](notebooks/nb09_manuscript_compilation.ipynb#L1098) cell 22 caption | `n=204` | match the nb09 source-of-truth scope |

### 3b. Should regenerate (no manual edit)

- `notebooks/executed/nb07b_arcpl_bias_probe_executed.ipynb` (cell outputs for
  RMSE=64.6 C etc. refresh on re-run).
- `results/nb04_arcpl_opx_liq_predictions_forest.csv` — if Option B restricts the subset,
  add an `eq_pass` boolean column rather than dropping rows (preserves ML coverage
  statistics).

### 3c. Safe to leave as-is

- `docs/putirka_inconsistency_audit.md` — historical reconciliation, mentions n=197 and
  n=204 in the same doc on purpose.
- Logs and archives.

## 4. Risk register before Option B goes in

1. **nb09 caption drift.** Two live references still say `n=204`. If Option B adds an
   `n=166` column somewhere in nb09, the ArcPL figures will have three different sample
   counts on the same page. Plan to reconcile all three (all/eq/converged) as a single
   footnote before regenerating nb09.
2. **Predictions CSV schema.** If Option B adds an `eq_pass` column, downstream probes
   (nb07b) should be re-run so they see it. Otherwise the CSVs drift from the notebook.
3. **Manuscript text.** No live `.md` or `.tex` manuscript draft cites an ArcPL N or a
   Putirka RMSE by value — all manuscript-facing numbers come from executed notebook
   outputs. Option B's numeric claims will flow through the notebook re-execution, not a
   prose edit.
4. **Fe3Fet choice is load-bearing for eq-pass counts.** The n=178 eq-pass count is
   computed with `Fe3Fet_Liq = 0.0` (reduced). If Option B opts for oxidized (0.15),
   expect the eq-pass count to shift by ~5-15 rows (direction: fewer passes because the
   Fe2+-only Kd moves further from `Ideal_Kd`). Whichever is chosen should be stated
   explicitly in the manuscript; 0.0 matches nb04 cell 27's current behavior.

## 5. Recommended sequencing when Option B is approved

1. Add `eq_tests=True` to nb04 cell 27. Save result as a CSV with `eq_pass` column.
2. Regenerate `figures/fig_nb04_putirka_vs_ml_arcpl.{png,pdf}` with the new middle row
   (Kd-equilibrated subset instead of the blanket "converged" subset).
3. Patch nb09 caption upstream (`scripts/patch_v8_nb09_fixes.py:61`) to match whichever
   scope the ArcPL figure actually uses. Re-run nb09.
4. Re-run nb07b so the probe tables absorb the new subset definition (if probes switch
   from "all ArcPL" to "equilibrated ArcPL").
5. Update `docs/putirka_inconsistency_audit.md` with a forward-pointer to the Option B
   prediction doc (no content change; just add a "superseded-for-manuscript-framing by"
   sentence at the top).
6. Leave `scripts/audit_*.py` scripts untouched; they are the evidence trail.

## 6. Total files needing an edit for Option B

- **Must edit**: 3 (nb04 cell 27, `scripts/patch_v8_nb09_fixes.py`, nb09 cell 22).
- **Regenerate on rerun**: 3 (nb07b executed, nb04 executed, results CSV).
- **Leave alone**: all logs, all archives, both audit docs, the Thermobar example library.

This is a narrow change surface. The numeric-consistency risk is low because no live
manuscript draft hardcodes the ArcPL N or the Putirka RMSE. The pipeline runs are the
source of truth.
