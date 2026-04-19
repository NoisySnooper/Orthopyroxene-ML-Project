# Pre-registration: Pressure-regime bins for calibration-domain characterization

**Registered:** 2026-04-17
**Author:** NQTa
**Status:** Locked. Any modification requires a new registration date and a written justification in Section 2 below.
**Scope:** Opx-liq thermobarometer benchmark analysis on ArcPL Kd-equilibrated subset (n = 96) and on the ExPetDB held-out test set.

---

## 1. Purpose

This document registers the pressure-regime bin structure used for regime-dependent benchmark analysis of the opx-liq thermobarometer against external methods (Putirka 2008, Agreda-Lopez 2024, Jorgenson 2022, Wang 2021). The registration timestamp is the primary defense against reviewer concerns about post-hoc bin selection.

Aggregate RMSE on ArcPL n = 96 remains the **primary** benchmark claim (Section 2.5 of the manuscript). Regime-dependent analysis is a **secondary, exploratory** characterization of where each method's calibration domain is most reliable, not a ranking or a basis for overclaiming relative performance.

## 2. Bin definitions (locked)

Four pressure regimes with edges at 0, 5, 15, 30, and 100 kbar (the last being the ExPetDB training-set pressure ceiling, `P_CEILING_KBAR` in `config.py`):

| Label | P range (kbar) | Petrological context |
|---|---|---|
| `shallow_crustal` | 0-5 | Arc storage, upper-crustal magma chambers |
| `deep_crustal_MASH` | 5-15 | Melting-assimilation-storage-homogenization zone |
| `lithospheric_mantle` | 15-30 | Spinel to garnet peridotite stability |
| `deeper_mantle` | >30 | Asthenospheric and below |

Bin membership is assigned on the **experimentally reported** pressure for ExPetDB test samples and on the **independently constrained** pressure for ArcPL natural samples. Bin edges are right-open (P = 5.0 kbar belongs to `deep_crustal_MASH`, not `shallow_crustal`).

## 3. Rationale

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

**Fallback protocol:** If any bin contains fewer than **20 samples** in ArcPL n = 96, we report the bin's numbers but explicitly mark the result as "sample-size-limited" in the per-bin table. We do not drop the bin and do not merge bins post-hoc.

### 4.2 Training-vs-test distributional mismatch

ExPetDB training samples are approximately uniformly distributed in P (after filtering) because training experiments span the full calibration range. ArcPL natural samples follow a different distribution (mostly crustal). This mismatch is the exact signal we expect regime analysis to expose; it is not a flaw of the bin structure.

### 4.3 Exploratory framing

Results from this bin structure are reported as **calibration-domain characterization**, not as a ranking of methods. The manuscript explicitly avoids claims of the form "method A is better than method B in regime X" unless bootstrap 95% CIs of RMSE, T bias, or P bias exclude the competing method's estimate at the 5% level AND the sample size in the regime exceeds 20.

## 5. Implementation

### 5.1 Config.py variables (locked)

```python
P_REGIME_BIN_EDGES_KBAR = [0.0, 5.0, 15.0, 30.0, P_CEILING_KBAR]
P_REGIME_LABELS = ['shallow_crustal', 'deep_crustal_MASH',
                   'lithospheric_mantle', 'deeper_mantle']
P_REGIME_REGISTERED_DATE = '2026-04-17'
P_REGIME_RATIONALE_DOC = 'docs/v10_p_regime_preregistration.md'
```

### 5.2 Assignment helper

In `src/evaluation.py` (or equivalent), add:

```python
import numpy as np
from config import P_REGIME_BIN_EDGES_KBAR, P_REGIME_LABELS


def assign_p_regime(p_kbar):
    """Return regime label for each pressure value.

    P_REGIME_BIN_EDGES_KBAR is [0, 5, 15, 30, 100]; np.digitize with
    right=False places P=5.0 in bin index 2 (deep_crustal_MASH), matching
    the right-open convention in the registration doc.
    """
    p = np.asarray(p_kbar, dtype=float)
    p_clipped = np.clip(p, 0.0, P_REGIME_BIN_EDGES_KBAR[-1] - 1e-9)
    bin_idx = np.digitize(p_clipped, P_REGIME_BIN_EDGES_KBAR[1:-1], right=False)
    return np.array([P_REGIME_LABELS[i] for i in bin_idx])
```

### 5.3 Per-bin metric reporting

For each (method x regime) combination, the benchmark notebook computes:
- n (sample size in bin)
- RMSE with bootstrap 95% CI (B = 1000 resamples)
- T bias (mean residual) with bootstrap 95% CI
- P bias (mean residual) with bootstrap 95% CI
- 90% prediction interval coverage fraction

The full per-bin table is generated by NB04 (benchmark) and written to `results/v10_opx_test_log.csv` with a `regime` column added.

## 6. Modification history

**2026-04-17 - Initial registration.** Bins fixed at 0/5/15/30/100 kbar. No per-bin V10 results exist at this timestamp (V10 Phase E training in progress).

*Any future modification requires a dated entry in this section with a written rationale. Undocumented edits to `P_REGIME_BIN_EDGES_KBAR` should be treated as a methodological red flag.*

## 7. Manuscript treatment

Section 2.5.1 of the main manuscript (one paragraph) introduces the four-bin structure and cites this doc.

Section 5.X of the main manuscript (brief discussion) interprets the per-bin pattern qualitatively, with the explicit framing that this is calibration-domain characterization and not a performance ranking.

Supplementary Section S8 presents the full per-bin table and full bootstrap CIs.

## References

- Annen, C., Blundy, J. D., and Sparks, R. S. J. (2006). The genesis of intermediate and silicic magmas in deep crustal hot zones. *Journal of Petrology*, 47(3), 505-539.
- Blundy, J., and Cashman, K. (2008). Petrologic reconstruction of magmatic system variables and processes. *Reviews in Mineralogy and Geochemistry*, 69(1), 179-239.
- Hildreth, W., and Moorbath, S. (1988). Crustal contributions to arc magmatism in the Andes of central Chile. *Contributions to Mineralogy and Petrology*, 98(4), 455-489.
- Klemme, S., and O'Neill, H. S. C. (2000). The near-solidus transition from garnet lherzolite to spinel lherzolite. *Contributions to Mineralogy and Petrology*, 138(3), 237-248.
- Putirka, K. D. (2008). Thermometers and barometers for volcanic systems. *Reviews in Mineralogy and Geochemistry*, 69(1), 61-120.
- Walter, M. J. (1998). Melting of garnet peridotite and the origin of komatiite and depleted lithosphere. *Journal of Petrology*, 39(1), 29-60.
