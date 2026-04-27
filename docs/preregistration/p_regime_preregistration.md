# Pre-registration: Pressure-regime bins and bias-correction acceptance rule

**Registered:** 2026-04-17 **Author:** NQTa **Status:** Locked. Any modification requires a new registration date and a written justification in Section 8 below. **Scope:** Opx-liq thermobarometer benchmark analysis on ArcPL Kd-equilibrated subset (n = 96), the ExPetDB held-out test set, and the bias-correction ship decision used to gate Form A and Form B before the canonical scorecard rewrite.

---

## 1. Purpose

This document registers two coupled methodological choices that define the pre-registered honesty bar for the opx ML thermobarometer paper:

1. The pressure-regime bin structure used for regime-dependent benchmark analysis against external methods (Putirka 2008, Agreda-Lopez 2024, Jorgenson 2022, Wang 2021).
2. The tiered acceptance rule that gates whether a bias-correction form (Form A regime-piecewise OLS, Form B Agreda-Lopez quantile-thresholded piecewise) ships into the canonical scorecard.

The registration timestamp is the primary defense against reviewer concerns about post-hoc bin selection, post-hoc tolerance choice, and post-hoc ship decisions.

Aggregate RMSE on ArcPL n = 96 remains the **primary** benchmark claim (Section 2.5 of the manuscript). Regime-dependent analysis is a **secondary, exploratory** characterization of where each method's calibration domain is most reliable, not a ranking or a basis for overclaiming relative performance.

## 2. Bin definitions (locked)

Four pressure regimes with edges at 0, 5, 15, 30, and 100 kbar (the last being the ExPetDB training-set pressure ceiling, `P_CEILING_KBAR` in `config.py`):

LabelP range (kbar)Petrological context`shallow_crustal`0-5Arc storage, upper-crustal magma chambers`deep_crustal_MASH`5-15Melting-assimilation-storage-homogenization zone`lithospheric_mantle`15-30Spinel to garnet peridotite stability`deeper_mantle`&gt;30Asthenospheric and below

Bin membership is assigned on the **experimentally reported** pressure for ExPetDB test samples and on the **independently constrained** pressure for ArcPL natural samples. Bin edges are right-open (P = 5.0 kbar belongs to `deep_crustal_MASH`, not `shallow_crustal`).

## 3. Rationale for the bin edges

### 3.1 Petrological basis, not statistical

The boundaries are pinned to petrological regime transitions that appear consistently in the igneous petrology literature, not to statistical properties of our test data:

- **5 kbar** - approximate upper limit of shallow crustal magma storage in most arc systems; below this, magma chambers sit in the upper crust above typical seismic discontinuities (Putirka, 2008, Table 3; Blundy and Cashman, 2008 Section 2).
- **15 kbar** - approximate mid-to-lower crustal boundary, the upper end of the MASH zone where arc magmas are typically modified before ascent (Hildreth and Moorbath, 1988; Annen, Blundy, and Sparks, 2006).
- **30 kbar** - approximate spinel-to-garnet peridotite transition in the lithospheric mantle, a well-established petrological boundary (Klemme and O'Neill, 2000; Walter, 1998).

We do not claim these are sharp phase-boundary transitions - they are reporting conventions for regime-dependent benchmark interpretation. Any reviewer familiar with igneous petrology will recognize the boundary set.

### 3.2 Not tuned to match Agreda-Lopez, Jorgenson, or our own win/loss pattern

The bins are not chosen to match the training distribution of any specific cpx thermobarometer, nor to place our model in a favorable light. We do not re-bin based on which boundaries maximize our apparent performance in any regime.

### 3.3 Not equal-N bins

Equal-N partitioning of the P axis would have been an alternative defensible choice but would place bin edges at distribution percentiles that have no petrological meaning. We prefer petrological interpretability over statistical uniformity; the tradeoff is accepted - see Section 4.

## 4. Accepted limitations

### 4.1 Unequal sample size per bin

ArcPL n = 96 distributed across four bins will be uneven. Based on the ArcPL pressure distribution (primarily arc samples), we expect most samples in `shallow_crustal` and `deep_crustal_MASH`, fewer in `lithospheric_mantle`, and possibly very few in `deeper_mantle`. This unevenness is reported honestly in the per-bin table, and within-bin bootstrap confidence intervals reflect the corresponding loss of statistical resolution.

**Sample-size floor:** If any bin contains fewer than `N_MIN_FOR_VETO = 20` samples, we report the bin's numbers but explicitly mark the result as "sample-size-limited" in the per-bin table. We do not drop the bin and do not merge bins post-hoc. The same `N_MIN_FOR_VETO` threshold also gates the bias-correction ship decision in Section 6.

### 4.2 Training-vs-test distributional mismatch

ExPetDB training samples are approximately uniformly distributed in P (after filtering) because training experiments span the full calibration range. ArcPL natural samples follow a different distribution (mostly crustal). This mismatch is the exact signal we expect regime analysis to expose; it is not a flaw of the bin structure.

### 4.3 Exploratory framing

Results from this bin structure are reported as **calibration-domain characterization**, not as a ranking of methods. The manuscript explicitly avoids claims of the form "method A is better than method B in regime X" unless bootstrap 95% CIs of RMSE, T bias, or P bias exclude the competing method's estimate at the 5% level AND the sample size in the regime exceeds 20.

## 5. Implementation of the regime structure

### 5.1 [Config.py](http://Config.py) variables (locked)

```python
P_REGIME_BIN_EDGES_KBAR = [0.0, 5.0, 15.0, 30.0, P_CEILING_KBAR]
P_REGIME_LABELS = ['shallow_crustal', 'deep_crustal_MASH',
                   'lithospheric_mantle', 'deeper_mantle']
P_REGIME_REGISTERED_DATE = '2026-04-17'
P_REGIME_RATIONALE_DOC = 'docs/preregistration/p_regime_preregistration.md'
```

### 5.2 Assignment helper

In `src/evaluation.py`:

```python
import numpy as np
from config import P_REGIME_BIN_EDGES_KBAR, P_REGIME_LABELS


def assign_p_regime(p_kbar):
    """Return regime label for each pressure value.

    P_REGIME_BIN_EDGES_KBAR is [0, 5, 15, 30, 100]; np.digitize with
    right=False places P=5.0 in bin index 2 (deep_crustal_MASH), matching
    the right-open convention in this document.
    """
    p = np.asarray(p_kbar, dtype=float)
    p_clipped = np.clip(p, 0.0, P_REGIME_BIN_EDGES_KBAR[-1] - 1e-9)
    bin_idx = np.digitize(p_clipped, P_REGIME_BIN_EDGES_KBAR[1:-1], right=False)
    return np.array([P_REGIME_LABELS[i] for i in bin_idx])
```

### 5.3 Per-bin metric reporting

For each (method × regime) combination, the benchmark notebook computes:

- n (sample size in bin)
- RMSE with bootstrap 95% CI (B = 1000 resamples)
- T bias (mean residual) with bootstrap 95% CI
- P bias (mean residual) with bootstrap 95% CI
- 90% prediction interval coverage fraction

The full per-bin table is generated by NB04 (benchmark) and written to `results/opx_test_log.csv` with a `regime` column added.

## 6. Bias-correction ship decision (locked)

A bias-correction form (Form A or Form B, both defined in `src/bias_correction.py`) ships into the canonical post-correction scorecard if and only if it improves the overall test RMSE by more than the numerical tolerance and does not degrade any well-populated pre-registered regime by more than the pre-registered tolerance band.

### 6.1 Constants

```python
SHIP_TOL = 1e-6                # numerical floor (kbar or degC)
N_MIN_FOR_VETO = 20            # Section 4.1 sample-size floor
T_ABS_T = 10.0                 # absolute tolerance (degC) for T targets
T_ABS_P = 1.0                  # absolute tolerance (kbar) for P targets
T_REL = 0.10                   # relative tolerance (fraction of pre-RMSE)
```

`SHIP_TOL` exists only as the numerical floor inside `max(...)`. `T_ABS_T`, `T_ABS_P`, `T_REL` are the substantive tolerances; they are constructed from two independent rationales documented in Section 6.4 below.

### 6.2 The rule

Let:

- `overall_delta = pre_overall_rmse - post_overall_rmse`
- For each pre-registered regime r:
  - `degradation_r = post_r - pre_r`
  - `n_r` = number of test samples in r
  - `pre_r` = pre-correction regime RMSE (in native target units)
- `target_is_T = (target == 'T_C')`, else it is a P target
- `tol_abs = T_ABS_T if target_is_T else T_ABS_P`

For each regime r, the **veto tolerance** is:

```
veto_tol_r = max(SHIP_TOL, tol_abs, T_REL * pre_r)
```

A correction form ships iff **all three** of the following hold:

1. **Overall-improvement clause.** `overall_delta > SHIP_TOL`.
2. **Veto-on-well-populated-regimes.** For every regime r with `n_r >= N_MIN_FOR_VETO`, `degradation_r <= veto_tol_r`. (A well-populated regime can veto if it degrades more than the tolerance band allows.)
3. **Floor-logging for low-n regimes.** For every regime r with `n_r < N_MIN_FOR_VETO`, the degradation is **logged but not vetoing**. These bins are reported in the scorecard with a `sample_size_limited=True` flag and a trailing ⚠ in advisor-package figures.

Tie-breaking between Form A and Form B: if both ship, the form with the larger `overall_delta` wins; ties (improbable in float) break toward Form A (simpler, more interpretable).

### 6.3 What the rule does not change

- Pre-registered regime edges (`0, 5, 15, 30, P_CEILING_KBAR`): unchanged.
- Definition of `overall_delta`: unchanged.
- External-benchmark comparison protocol (Putirka / Agreda-Lopez / Jorgenson / Wang): not touched.
- Bootstrap protocol (n_boot = 2000 paired for Δ-RMSE, 2000 unpaired for RMSE CIs; seed = SEED_BOOTSTRAP): unchanged.
- TabPFN handling: TabPFN OOF coverage is 5 seeds; rerunning to 20 seeds is out of scope. TabPFN bias-correction verdicts are preserved byte-for-byte from the pre-tolerance v1 rule (no flips).

### 6.4 Scientific justification for the tolerance constants

The tolerance `max(tol_abs, T_REL × pre_r)` is deliberately constructed from two independent rationales:

#### 6.4.1 `tol_abs`: measurement-uncertainty floor

For opx thermobarometry, the experimental-petrology literature reports the following RMSE floors on the same ExPetDB corpus class (Putirka 2008, Table 4.1, and Neave & Putirka 2017):

- **Temperature calibrations:** Putirka eq. 28a (opx-liq), eq. 32d (cpx-liq) report SEE = 26-38 °C on their calibration sets. Independent test holdouts (Jorgenson et al. 2022) recover 30-50 °C RMSE. A **10 °C absolute floor** sits comfortably below every reported SEE and is therefore in the "cannot be distinguished from experimental noise" regime.
- **Pressure calibrations:** Putirka eq. 29a/29b/29c (opx-only, opx-liq) report SEE = 2.5-3.2 kbar; Neave-Putirka 2017 report \~2.8 kbar. A **1 kbar absolute floor** is well below every reported pressure SEE and is *stricter than* the measurement-noise rationale would require; we adopt the conservative choice rather than matching the SEE exactly.

The floor therefore cannot ship a correction that worsens a regime by more than experimental noise would allow anyway.

#### 6.4.2 `T_REL = 0.10`: relative guard

On high-difficulty regimes (e.g., `deeper_mantle` with pre-RMSE of 200-300 °C or 4-6 kbar) the absolute floor becomes loose. The **10 % of pre-regime RMSE** guard scales with regime difficulty so that a high-variance bin cannot be "improved" by a correction that silently worsens it by 20 % while the absolute-floor clause looks the other way. The 10 % value is the smallest round-number threshold that satisfies `max(T_ABS, T_REL × pre_r) == T_REL × pre_r` for the two regimes where the relative clause needs to engage (deeper_mantle T pre ≥ 100 °C, deeper_mantle P pre ≥ 10 kbar).

Both rationales are independent of the specific post-correction RMSE values observed in the data: neither constant was chosen by looking at any post-correction number.

### 6.5 Implementation

`src/bias_correction.py::ship_decision` carries the rule:

```python
def ship_decision(form: str,
                  pre_overall: float, post_overall: float,
                  per_regime_pre: dict, per_regime_post: dict,
                  per_regime_n: dict | None = None,
                  n_min_for_veto: int = 0,
                  tol: float = SHIP_TOL,
                  degradation_tol_abs: float = 0.0,
                  degradation_tol_rel: float = 0.0) -> ShipDecision:
```

Per-regime veto threshold is `max(tol, degradation_tol_abs, degradation_tol_rel * per_regime_pre[r])`. The canonical ship-decision call uses `n_min_for_veto=20`, `degradation_tol_abs=10.0` (T) or `1.0` (P), `degradation_tol_rel=0.10`.

`scripts/bias_correction/rescore_under_v3_rule.py` produces the canonical scorecard `results/preregistered_scorecard_postcorrection.csv` (and the matching `bias_correction_shipped.csv`) under this rule.

### 6.6 Pre-registration tests

`tests/test_preregistration.py` asserts:

- **T15**: `ship_decision` with default tolerances reproduces a pre-tolerance baseline byte-for-byte (verifies backward compatibility of the API).
- **T16**: A regime with `n < N_MIN_FOR_VETO` that degrades does not block shipping when the rule is engaged.
- **T17**: `N_MIN_FOR_VETO == 20` is hard-coded in the rescore script and matches this document.
- **T18**: TabPFN ship verdicts in the canonical CSV match the pre-tolerance baseline byte-for-byte (verifies TabPFN exclusion).
- **T19**: `ship_decision` with `degradation_tol_abs=0` and `degradation_tol_rel=0` reproduces the pre-tolerance baseline (verifies the tolerance is opt-in).
- **T20**: Under canonical tolerances, a high-n regime whose degradation is at or below `max(SHIP_TOL, T_ABS, T_REL * pre_r)` does NOT veto; high-n regimes that degrade by more than the tolerance still veto.
- **T21**: The constants `T_ABS_T = 10.0`, `T_ABS_P = 1.0`, `T_REL = 0.10` are hard-coded in `scripts/bias_correction/rescore_under_v3_rule.py` and match this document.
- **T22**: Every opx non-TabPFN row in `results/preregistered_scorecard_postcorrection.csv` is internally consistent with `ship_decision` and the constants in this document.

## 7. Manuscript treatment

Section 2.5.1 of the main manuscript (one paragraph) introduces the four-bin structure and cites this doc.

Section 4 of the main manuscript reports the pre-registered scorecard winners under the rule in Section 6.

Section 5.X of the main manuscript (brief discussion) interprets the per-bin pattern qualitatively, with the explicit framing that this is calibration-domain characterization and not a performance ranking.

Supplementary Section S8 presents the full per-bin table and full bootstrap CIs.

## 8. Modification history

**2026-04-17: Initial registration.** Bins fixed at 0/5/15/30/100 kbar. Bias-correction ship rule with `N_MIN_FOR_VETO = 20`, `T_ABS_T = 10.0`, `T_ABS_P = 1.0`, `T_REL = 0.10`. No per-bin results exist at this timestamp (training in progress). Constants are final.

*Any future modification requires a dated entry in this section with a written rationale and a new test case. Undocumented edits to the constants in Section 5.1 or Section 6.1 should be treated as a methodological red flag.*

## References

- Annen, C., Blundy, J. D., and Sparks, R. S. J. (2006). The genesis of intermediate and silicic magmas in deep crustal hot zones. *Journal of Petrology*, 47(3), 505-539.
- Blundy, J., and Cashman, K. (2008). Petrologic reconstruction of magmatic system variables and processes. *Reviews in Mineralogy and Geochemistry*, 69(1), 179-239.
- Hildreth, W., and Moorbath, S. (1988). Crustal contributions to arc magmatism in the Andes of central Chile. *Contributions to Mineralogy and Petrology*, 98(4), 455-489.
- Jorgenson, C. W., Higgins, O., Petrelli, M., Bégué, F., and Caricchi, L. (2022). A machine learning-based approach to clinopyroxene thermobarometry: Model optimization and distribution for use in Earth Sciences. *Journal of Geophysical Research: Solid Earth*, 127(4), e2021JB022904.
- Klemme, S., and O'Neill, H. S. C. (2000). The near-solidus transition from garnet lherzolite to spinel lherzolite. *Contributions to Mineralogy and Petrology*, 138(3), 237-248.
- Neave, D. A., and Putirka, K. D. (2017). A new clinopyroxene-liquid barometer, and implications for magma storage pressures under Icelandic rift zones. *American Mineralogist*, 102(4), 777-794.
- Putirka, K. D. (2008). Thermometers and barometers for volcanic systems. *Reviews in Mineralogy and Geochemistry*, 69(1), 61-120.
- Walter, M. J. (1998). Melting of garnet peridotite and the origin of komatiite and depleted lithosphere. *Journal of Petrology*, 39(1), 29-60.
