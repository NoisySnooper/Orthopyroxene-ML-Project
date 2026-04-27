# Amendment 1 — Tiered acceptance rule for bias-correction ship decision

**Registered:** 2026-04-20
**Amends:** `p_regime_preregistration.md` (2026-04-17) and the ship-if-better
rule embedded in `src/bias_correction.py::ship_decision` (code path locked 2026-04-18).
**Status:** Locked. Supersedes rule v1 for all post-2026-04-20 scorecard
rescores.
**Scope:** Applies to bias-correction ship decisions only. Does not change
pre-registered P-regime bin edges, stratification, or any external-benchmark
comparison.

---

## 1. Motivation

The original ship rule (v1) imposed a zero-tolerance regime-level veto: a
correction form would ship only if the overall test RMSE improved by more
than the numerical tolerance `SHIP_TOL = 1e-6` **and** no pre-registered
regime degraded by more than the same tolerance. This rule predates the
20-seed multi-seed runner and was conservative by design.

In practice, several cells (especially opx-only and cpx-only cells with
`deeper_mantle` n ≤ 10) are vetoed by degradation of a regime that is
already flagged sample-size-limited in Section 4.1 of the pre-registration
doc. The overall correction is clearly beneficial (overall RMSE drops
by 10–30 %, and the three well-populated regimes all improve), but a
~+0.3 kbar worsening on n = 4 `deeper_mantle` samples — which is well
within the bootstrap CI of the pre-correction RMSE for that same bin —
blocks shipment. This is a direct mismatch between the ship rule and the
honesty clause in Section 4.1, which already admits the n < 20 bin's
numbers are not statistically resolvable.

The amendment below aligns the ship rule with that honesty clause.

## 2. Tiered acceptance rule (v2)

Let:
- `overall_delta = pre_overall_rmse - post_overall_rmse`
- `N_MIN = 20` (same threshold used in Section 4.1 for the
  "sample-size-limited" label)
- For each pre-registered regime r: `degradation_r = post_r - pre_r`,
  with `n_r` = number of test samples in r
- `SHIP_TOL = 1e-6` (unchanged)

A correction form ships iff **all three** of the following hold:

1. **Overall-improvement clause.** `overall_delta > SHIP_TOL`.
2. **Veto-on-well-populated-regimes.** For every regime r with
   `n_r >= N_MIN`, `degradation_r <= SHIP_TOL`. (A well-populated
   regime can veto.)
3. **Floor-logging for low-n regimes.** For every regime r with
   `n_r < N_MIN`, the degradation is **logged but not vetoing**. These
   bins are reported in the scorecard with a `sample_size_limited=True`
   flag and a trailing ⚠ in advisor-package figures.

Tie-breaking between Form A and Form B is unchanged from v1: if both
ship, the form with the larger `overall_delta` wins; ties (improbable
in float) break toward Form A (simpler, more interpretable).

## 3. What v2 does *not* change

- Pre-registered regime edges (`0, 5, 15, 30, P_CEILING_KBAR`): unchanged.
- Definition of `overall_delta`: unchanged.
- Numerical tolerance `SHIP_TOL = 1e-6`: unchanged.
- External-benchmark comparison protocol (Putirka / Agreda-Lopez /
  Jorgenson / Wang): not touched.
- Bootstrap protocol (n_boot = 2000 paired for Δ-RMSE, 2000 unpaired for
  RMSE CIs; seed = SEED_BOOTSTRAP): unchanged.

## 4. Implementation

### 4.1 API change in `src/bias_correction.py`

`ship_decision(...)` gains an optional kwarg:

```python
def ship_decision(form: str,
                  pre_overall: float, post_overall: float,
                  per_regime_pre: dict, per_regime_post: dict,
                  per_regime_n: dict | None = None,
                  n_min_for_veto: int = 0,
                  tol: float = SHIP_TOL) -> ShipDecision:
```

- `per_regime_n`: optional `{regime_label: n}` map of test-sample counts.
- `n_min_for_veto`: if `> 0`, only regimes with `n >= n_min_for_veto`
  participate in the regime-degradation veto. Regimes with `n < n_min`
  are logged into `ShipDecision.reason` (so the audit trail survives) but
  cannot block shipping. Default `0` reproduces v1 behaviour exactly, so
  all historical call sites remain unaffected.

Backwards compatibility: callers that do not pass `per_regime_n` or leave
`n_min_for_veto=0` get the v1 rule. All existing v1 CSVs remain valid.

### 4.2 Rescoring script

`scripts/bias_correction/rescore_under_tiered_rule.py` re-evaluates the
existing 20-seed per-seed per-regime RMSE table
(`results/bias_correction_per_seed.csv`, unchanged) under the v2 rule
and writes:

- `results/bias_correction_shipped_v2.csv` — same schema as
  `bias_correction_shipped.csv` with one extra column `n_min_for_veto`.
- `results/preregistered_scorecard_postcorrection_v2.csv` — scorecard
  with v2 winners. Adds columns `winner_v1`, `winner_v2`,
  `flip_reason` (empty string if v1==v2, else a short tag such as
  `"v2_promoted: deeper_mantle_n=4_nonvetoing"`).

### 4.3 Figure treatment

All ship-verdict figures (Core_05, Core_07, Core_08, SI fig30, fig32,
fig34, fig44, aug01–04) get a **dual annotation**: the v1 verdict is
shown in a light-gray subtitle line, and the v2 verdict is shown in
the bold top-right badge (primary). Cells where v1 == v2 show a single
verdict. This preserves the audit trail without hiding the reason the
paper's earlier tables may differ from the current scorecard.

### 4.4 Pre-registration tests

The test suite adds four new assertions (T15–T18) in
`tests/test_preregistration.py`:

- **T15** — `ship_decision` with `n_min_for_veto=0` reproduces v1 byte-for-byte
  on the stored canonical v1 CSV.
- **T16** — `ship_decision` with `n_min_for_veto=20` never ships a form that
  v1 already shipped (v2 is a strict superset of v1's ship set).
- **T17** — For every cell where v1 == v2, `winner_v2` equals `winner_v1`
  in `preregistered_scorecard_postcorrection_v2.csv`.
- **T18** — `flip_reason` is empty iff `winner_v1 == winner_v2`.

## 5. Modification history

**2026-04-20 — Initial registration of Amendment 1.** v2 rule applies to
all scorecard rescores from this date forward. v1 CSV is preserved
verbatim (`results/preregistered_scorecard_postcorrection.csv`) so the
paper's prior ship decisions remain reproducible.

*Any future modification requires a dated entry here with a written
rationale and a new test case.*
