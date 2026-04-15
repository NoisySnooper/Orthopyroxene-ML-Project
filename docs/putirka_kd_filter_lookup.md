# Part 1: Thermobar's built-in Putirka 2008 opx-liq Kd filter

**Status:** Definitive. No manual reimplementation required.

## Summary

Thermobar 1.0.70 does **not** expose a standalone
`calculate_opx_liq_eq_tests` helper (the cpx equivalent exists:
`calculate_cpx_liq_eq_tests`). The canonical way to apply Putirka's
recommended opx-liq equilibrium filter is via the `eq_tests=True`
switch on the regular solver:

- `Thermobar.calculate_opx_liq_press(..., eq_tests=True)`
- `Thermobar.calculate_opx_liq_press_temp(..., eq_tests=True)` (iterative joint; **the one to use**)

With `eq_tests=True`, the returned DataFrame gains a column named
literally `Kd Eq (Put2008+-0.06)` containing 'Y' / 'N' strings --
Thermobar's own labeling of whether each row passes Putirka 2008's
Fe-Mg equilibrium criterion.

## Reference (Putirka 2008)

- Putirka, K. D. (2008), *Thermometers and Barometers for Volcanic
  Systems*, Reviews in Mineralogy & Geochemistry, vol. 69, p. 61-120.
- Opx-liquid Kd_Fe-Mg expression: `Kd = 0.4805 - 0.3733 * X_Si_Liq`
  (Putirka 2008 eq ~35, from the Toplis-style liquid composition
  dependence). Thermobar labels this `Ideal_Kd`.
- Tolerance: `Kd_measured` within `Kd_ideal +- 0.06` at 1-sigma.
  Thermobar's constant is hard-coded as ±0.06 in the column name.
- The fallback constant `Kd = 0.29 +- 0.06` (from
  `calculate_opx_rhodes_diagram_lines` docstring) is the
  averaged expected value for common basaltic-andesitic liquids.

## Function signature

```
Thermobar.calculate_opx_liq_press_temp(
    *,
    opx_comps:  pandas.DataFrame,     # SiO2_Opx, TiO2_Opx, ..., MgO_Opx, FeOt_Opx, ...
    liq_comps:  pandas.DataFrame,     # SiO2_Liq, ..., H2O_Liq, Fe3Fet_Liq
    equationT:  str = 'T_Put2008_eq28a',
    equationP:  str = 'P_Put2008_eq29a',
    iterations: int = 30,
    T_K_Guess:  float = 1300,
    eq_tests:   bool = True,          # <-- the switch
    H2O_Liq:    float | array | None = None,
    Fe3Fet_Liq: float | array | None = None,
)
```

Return with `eq_tests=True`: a `pandas.DataFrame` with 89 columns. The
manuscript-relevant ones:

| Column | Meaning |
|---|---|
| `T_K_calc` | Iterative T solution (Kelvin) |
| `P_kbar_calc` | Iterative P solution (kbar) |
| `Kd_Fe_Mg_Fet` | Measured Kd Fe-Mg using total Fe |
| `Kd_Fe_Mg_Fe2` | Measured Kd using Fe2+ only (requires `Fe3Fet_Liq`) |
| `Ideal_Kd` | Putirka's liquid-Si-dependent ideal Kd |
| `Delta_Kd_Fe_Mg_Fe2` | Measured minus ideal |
| `Kd Eq (Put2008+-0.06)` | **'Y' if `\|Kd - Ideal_Kd\| <= 0.06`, else 'N'** |
| `Mgno_OPX`, `Mgno_Liq_noFe3`, `Mgno_Liq_Fe2` | Mg#'s on both sides |

## Working code snippet (drop-in for the nb04 cell 27 replacement)

```python
import numpy as np
import pandas as pd
import Thermobar as pt

def run_putirka_with_eq_filter(arcpl, fe3fet=0.0):
    """Run Putirka 28a/29a with Thermobar's built-in equilibrium test.

    Returns (df_results, mask_eq_pass).
      df_results : n x 89 pandas DataFrame from calculate_opx_liq_press_temp
      mask_eq_pass : boolean numpy array, True where row passes Put2008+-0.06
    """
    def g(col, d=0.0):
        if col in arcpl.columns:
            return pd.to_numeric(arcpl[col], errors='coerce').fillna(d).values
        return np.zeros(len(arcpl))

    opx = pd.DataFrame({
        'SiO2_Opx':  g('SiO2'),  'TiO2_Opx':  g('TiO2'),  'Al2O3_Opx': g('Al2O3'),
        'FeOt_Opx':  g('FeO_total'), 'MgO_Opx': g('MgO'), 'CaO_Opx':  g('CaO'),
        'MnO_Opx':   g('MnO'),   'Cr2O3_Opx': g('Cr2O3'), 'Na2O_Opx': g('Na2O'),
    })
    liq = pd.DataFrame({
        'SiO2_Liq':  g('liq_SiO2'),  'TiO2_Liq':  g('liq_TiO2'),
        'Al2O3_Liq': g('liq_Al2O3'), 'FeOt_Liq':  g('liq_FeO'),
        'MgO_Liq':   g('liq_MgO'),   'CaO_Liq':   g('liq_CaO'),
        'MnO_Liq':   g('liq_MnO'),   'K2O_Liq':   g('liq_K2O'),
        'Na2O_Liq':  g('liq_Na2O'),  'Cr2O3_Liq': g('liq_Cr2O3'),
        'P2O5_Liq':  g('P2O5_Liq'),  'H2O_Liq':   g('H2O_Liq'),
        'Fe3Fet_Liq': np.full(len(arcpl), fe3fet),
    })
    out = pt.calculate_opx_liq_press_temp(
        opx_comps=opx, liq_comps=liq,
        equationT='T_Put2008_eq28a', equationP='P_Put2008_eq29a',
        eq_tests=True,
    )
    mask = out['Kd Eq (Put2008+-0.06)'].astype(str).str.upper().str.startswith('Y').values
    return out, mask
```

## Verified behavior on ArcPL (n=197 from nb04 Part 3)

```
n_total                       197
Kd Eq (Put2008+-0.06) == 'Y'  178   (90.4%)
Kd Eq (Put2008+-0.06) == 'N'   19   ( 9.6%)
Solver converged              185   (93.9%)
Eq AND converged              166   (84.3%)  <-- Option B scope
```

## Why Thermobar's built-in and not a manual Kd range

- Thermobar's ±0.06 tolerance is what the Putirka 2008 paper recommends
  and what Thermobar exposes as its one-liner acceptance check. Using
  the canonical label is easier to defend to a reviewer than a
  user-defined `[0.23, 0.35]` range.
- The custom range `[0.23, 0.35]` in `config.KD_FEMG_MIN/MAX` is
  looser on the center (ignores the liquid-Si-dependent `Ideal_Kd`
  shift). For ArcPL, this accepts 156/197 rows vs. Thermobar's 178.
- Recommendation for Option B: use `Kd Eq (Put2008+-0.06) == 'Y'`
  as the canonical filter.

## Notes on related Thermobar helpers

- `calculate_opx_liq_press_temp_matching` -- a meltmatch helper that
  evaluates all pairs of opx x liq rows. Not needed for our use case
  where opx-liq pairs are already defined by the experiment.
- `calculate_opx_rhodes_diagram_lines` -- returns x-y data for plotting
  Kd equilibrium lines on an opx/liq Mg# diagram. Useful for Appendix
  figure but not for filtering.
- `calculate_cpx_opx_eq_tests` -- for cpx-opx pairs (two-pyroxene),
  not opx-liq.

## Checklist before applying

- [x] `Fe3Fet_Liq` explicitly set in `liq_comps` (0.0 for reduced,
      0.15 for oxidized). Filter results depend on this choice --
      the `Kd_Fe_Mg_Fe2` column reflects post-Fe3 correction.
- [x] `H2O_Liq` column present (use 0.0 for anhydrous defaults).
- [x] All 9 opx oxides (SiO2..Na2O_Opx) and 12 liq oxides populated;
      missing columns default to 0.0 -- Thermobar tolerates this.
- [x] `eq_tests=True` toggled on every solver call whose result feeds
      the manuscript.
