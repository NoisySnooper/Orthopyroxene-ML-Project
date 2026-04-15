# Part 3: Option B pre-implementation prediction — ML vs Putirka on the Kd-equilibrated subset

**Date:** 2026-04-15
**Status:** Read-only forecast. No notebook or config file modified. Numbers below come
from `scripts/audit_optionB_prediction.py`, which rebuilds the nb04 Part 3 ArcPL set and
applies Thermobar's built-in `Kd Eq (Put2008+-0.06)` flag. They are the result Option B
will reproduce once implemented.

## TL;DR

On the Kd-equilibrated subset (n=166 of 197, 84.3%), Putirka 2008 wins T by ~12 C and
loses P by ~1.0 kbar versus our ML forest. This does **not** depend on the implementation
details still under discussion (figure layout, scope in probes); the subset and metrics
are fixed once the Thermobar Kd filter is turned on.

## 1. Subset counts

Ground rules: rebuild ArcPL through nb04 Part 3's cleaning pipeline, run
`calculate_opx_liq_press_temp(eq_tests=True)` with `Fe3Fet_Liq = 0.0`, anhydrous when
`H2O_Liq` is missing.

```
n_total                              197
Kd Eq (Put2008+-0.06) == 'Y'         178   (90.4%)
Kd Eq (Put2008+-0.06) == 'N'          19   ( 9.6%)
Solver converged (T, P both finite)  185   (93.9%)
Eq AND converged  <-- OPTION B       166   (84.3%)
Legacy Kd in [0.23, 0.35] (config)   156   (79.2%)  (Thermobar's filter is more permissive)
```

**Option B scope = n=166.** 31 of the 197 rows drop out: 19 fail Thermobar's equilibrium
check, 12 fail Putirka's iterative solver, with overlap (most failures fail both).

## 2. RMSE table by scope

| Scope | n | ML T RMSE (C) | ML P RMSE (kbar) | Putirka T RMSE (C) | Putirka P RMSE (kbar) |
|---|---:|---:|---:|---:|---:|
| Full ArcPL (ML predicts all) | 197 | **64.58** | **2.63** | N/A | N/A |
| Eq pass only | 178 | 63.86 | 2.71 | 50.33 | 3.73 |
| Converged only | 185 | 63.63 | 2.65 | 50.08 | 3.60 |
| **Eq AND converged (Option B)** | **166** | **62.73** | **2.74** | **50.33** | **3.73** |

Source: [results/optionB_preflight_metrics.csv](results/optionB_preflight_metrics.csv)
produced by [scripts/audit_optionB_prediction.py](scripts/audit_optionB_prediction.py).

### Verdict per axis on the Option B subset (n=166)

- **Temperature**: Putirka wins by **12.4 C** (50.33 vs 62.73 C RMSE). Unambiguous.
- **Pressure**: ML wins by **1.0 kbar** (2.74 vs 3.73 kbar RMSE). Unambiguous.

### Verdict including coverage (the full story)

- **ML predicts every ArcPL sample** (100% coverage, RMSE 64.6 C / 2.63 kbar on n=197).
- **Putirka declines 31 of 197 samples** as non-equilibrated or non-convergent, then
  achieves better T on the 166 it accepts — but cannot produce a prediction for the
  other 31.
- The Putirka wins on T are conditional: they are only defensible on the Kd-equilibrated
  subset that Putirka itself selects. On the 19 non-equilibrated samples, Putirka has no
  output and ML alone carries the prediction.

## 3. Recommended manuscript framing

Option B is one of the three options flagged in
[docs/putirka_inconsistency_audit.md](docs/putirka_inconsistency_audit.md#L136):

> Option B (aggressive, "Putirka naively applied") -- use the looser n~200 scope (less
> Kd filtering).

The original Option B was unfiltered. This preflight redefines Option B as
**equilibrated + rigorous**: apply Thermobar's canonical Kd filter, then compare. That is
a stricter, more defensible version of "Putirka as Putirka recommends it."

### Suggested paragraph (for the manuscript body)

> We applied Thermobar's implementation of Putirka's (2008) Fe-Mg exchange equilibrium
> test (`Kd` within `0.4805 − 0.3733·X_Si_Liq ± 0.06`) to the ArcPL external set, retaining
> the 178 of 197 samples that pass and solving Putirka equations 28a/29a jointly via the
> iterative solver. 166 samples (84.3%) both pass the equilibrium test and admit a
> converged solution. On this Kd-equilibrated subset, Putirka 28a/29a predicts temperature
> with RMSE 50 C, outperforming our ML forest (63 C). The opposite is true for pressure:
> ML forest RMSE 2.7 kbar versus Putirka 3.7 kbar on the same subset. Over the full ArcPL
> set, ML predicts every sample (100% coverage, 65 C / 2.6 kbar), while Putirka cannot
> produce an equilibrated prediction on 16% of samples.

### Suggested figure caption (for the regenerated 3x2 panel)

> Figure N. ArcPL external validation. Row 1: ML forest on all n=197 samples. Row 2:
> Putirka 2008 eq 28a/29a on the n=166 subset that passes Thermobar's built-in Fe-Mg
> Kd-exchange equilibrium test (|Kd - Kd_ideal| <= 0.06) AND yields a converged iterative
> solution. Row 3: head-to-head on the shared n=166 subset. Points are colored by observed
> T (left column) or P (right column); dashed line is 1:1. RMSE annotated on each panel.

## 4. Decision checklist for implementation

Before any code edits, confirm these choices:

| Decision | Default | Alternatives | Consequence |
|---|---|---|---|
| Fe3Fet_Liq for liq comps | 0.0 (reduced) | 0.15 (oxidized) | Affects eq-pass count by roughly 10-20 rows in either direction. Current numbers use 0.0, matching nb04 cell 27. |
| Keep n_full = 197 denominator? | Yes | drop to n=166 everywhere | Dropping everywhere hides the coverage advantage of ML. Keep n=197 as the universe; report both numerators. |
| Row 2 scope in the figure | Eq AND converged (n=166) | Eq pass only (n=178, includes 12 non-convergent -> gap rows) | Eq AND converged avoids NaN scatter points; keep it. |
| Add `eq_pass` column to the forest predictions CSV? | Yes | No (only compute at plot time) | Yes enables nb07b and nb08 to re-slice by eq_pass without rerunning Thermobar. |
| Update nb09 caption from n=204 -> actual scope | Yes | leave as stale | Must update; see [docs/codebase_consistency_audit_optionB.md](docs/codebase_consistency_audit_optionB.md). |

## 5. What Option B does *not* change

- ML forest training (still on LEPR internal split, same hyperparameters, same CV).
- Feature engineering (6-oxygen cation recalc + engineered features).
- The full-coverage ML RMSE claim (n=197, 64.6 C / 2.63 kbar).
- nb07b's Probe 7 recommendation (Option A: uncorrected with caveat) — that decision was
  about bias correction, not about the Putirka baseline.

## 6. Implementation cost

- Patch nb04 cell 27: add `eq_tests=True`, add `eq_pass` mask, re-slice the figure's
  middle row. **~45 min**.
- Regenerate the figure + save a new CSV with `eq_pass` column. **~15 min**.
- Patch nb09 cell 22 / `scripts/patch_v8_nb09_fixes.py` caption from n=204 to the correct
  scope. **~15 min**.
- Re-execute nb07b so its probe tables see the refreshed ArcPL prediction file (no code
  change). **~30 min** wall clock via papermill.
- Write manuscript paragraph + caption into the draft. **~30 min**.

**Total: ~2.5 hours** for a clean, end-to-end Option B landing.

## 7. Pointer to the code that produced these numbers

- [scripts/audit_optionB_prediction.py](scripts/audit_optionB_prediction.py) —
  reproduces the n=166 subset and the RMSE table in one run.
- [results/optionB_preflight_metrics.csv](results/optionB_preflight_metrics.csv) —
  4-scope RMSE table.
- [docs/putirka_kd_filter_lookup.md](docs/putirka_kd_filter_lookup.md) — Thermobar API
  reference used.
- [docs/putirka_inconsistency_audit.md](docs/putirka_inconsistency_audit.md) — the
  scope-mismatch audit that motivated this preflight.
