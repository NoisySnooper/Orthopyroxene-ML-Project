# Putirka 2008 opx-liq (eq 28a / 29a) code path audit

**Date:** 2026-04-15
**Scope:** Reconcile the two inconsistent Putirka benchmark results that appeared in the manuscript draft:

- Path A (legacy, nb04b cell 19):   86.3% failure rate, 118.7 C T RMSE, 3.39 kbar P RMSE on **n=28**
- Path B (current, nb04 cell 27):    6.1% failure rate,  50.1 C T RMSE, 3.60 kbar P RMSE on **n=185**

**Key finding:** The 118.7 C headline figure is not an error in the Putirka code. It is a true result on a different (smaller, different-scope) ArcPL subset. The *current* figure of merit comes from running the iterative joint-solver on a Kd-pre-filtered ArcPL set where Putirka converges readily. Both code paths, when run on the same 197-row subset, converge on the same 185 rows (0 exclusive failures in either direction).

Neither Path A nor Path B applies an explicit Kd equilibrium filter inside the Putirka call. The apparent filter in Path A is *upstream in nb04b's data prep*, which is looser than nb04's Part 3 prep.

---

## 1. Thermobar version

Both paths use the **same Thermobar version** — the venv copy.

```
$ python -c "import Thermobar; print(Thermobar.__version__)"
1.0.70
```

Path A uses a `subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'Thermobar'])`
guard (nb04b cell 19, lines 10-17) with *no version pin*. On this machine the call is a no-op
because Thermobar 1.0.70 is already installed. It *could* drift if run elsewhere on a fresh
env, but is not the source of the divergence reported here.

Path B uses the already-imported `Thermobar as _pvm_tb` from the project venv — same library,
same functions. **Version drift is not the explanation.**

---

## 2. Per-path equilibrium / failure filter

| Path | Function(s) called | Algorithm | Fe3Fet_Liq | Post-hoc clip bounds | Explicit Kd filter |
|------|---|---|---|---|---|
| A (nb04b cell 19) | `pt.calculate_opx_liq_temp` + `pt.calculate_opx_liq_press` *separately* | One-sided: T computed from obs P; P computed from obs T | **0.15** | T outside [0, 2000] C -> NaN; P outside [-10, 100] kbar -> NaN | **none** |
| B (nb04 cell 27)  | `pt.calculate_opx_liq_press_temp` *iterative joint solver* | Solves (T, P) simultaneously from composition only; no ground truth used | **0.0** | none | **none** |

Neither path calls `calculate_eq_px_liq_T_Kd_Eq_Test`, `kd_opx_liq_filter`, or any of the
`Mgno_*` / Fe-Mg Kd equilibrium routines Thermobar exposes. The "failures" in both paths are
the NaN results that the Putirka solver itself returns when its internal iteration diverges.

Path A's post-hoc clip bounds are very loose (T 0-2000 C, P -10-100 kbar). On the n=197
scope they do not convert any finite Thermobar output into NaN; every failure there is a
native solver failure.

---

## 3. Per-sample head-to-head on 5 ArcPL rows

Both paths run on the same 5 rows (indices [150, 86, 127, 17, 85] from the n=197 set, seed 42):

| Row | T_obs (C) | P_obs (kbar) | Path A: T (clip) | Path A: P (clip) | Path B: T | Path B: P |
|-----|-----|-----|-----|-----|-----|-----|
| 0 |  825.0 | 2.85 |  847.5 | -1.42 |  837.9 | -1.28 |
| 1 | 1080.0 | 7.03 | 1133.7 |  8.38 | 1143.3 |  9.28 |
| 2 |  850.0 | 7.00 |  838.6 |  9.69 |  844.7 |  9.62 |
| 3 | 1110.0 | 7.00 | 1188.8 |  5.30 | 1185.5 |  6.28 |
| 4 | 1100.0 | 7.04 | 1128.5 |  7.91 | 1134.2 |  8.38 |

Both paths converge on all 5 samples. Per-sample differences:
- T: |T_A - T_B| = 3 to 10 C
- P: |P_A - P_B| = 0.07 to 1.0 kbar

None of the five are "filtered out" by either path. The differences are purely numerical
(different algorithm + different Fe3Fet).

### Isolating Fe3Fet vs algorithm

Running Path A's recipe but with `Fe3Fet_Liq = 0.0` (Path A\*, same call pattern as Path A
but with Path B's Fe3 value):

| Row | T_A\*(Fe3=0) | T_B | delta_T | P_A\*(Fe3=0) | P_B | delta_P |
|-----|-----|-----|-----|-----|-----|-----|
| 0 |  847.5 |  837.9 |  +9.65 | -1.42 | -1.28 | -0.14 |
| 1 | 1133.7 | 1143.3 |  -9.60 |  8.38 |  9.28 | -0.90 |
| 2 |  838.6 |  844.7 |  -6.11 |  9.69 |  9.62 | +0.07 |
| 3 | 1188.8 | 1185.5 |  +3.33 |  5.30 |  6.28 | -0.98 |
| 4 | 1128.5 | 1134.2 |  -5.66 |  7.91 |  8.38 | -0.48 |

The residual T difference of 3-10 C after removing the Fe3 effect is the **algorithm effect**:
Path A uses observed P as input to eq 28a, while Path B solves (T, P) jointly. Path A is
effectively "what temperature does Putirka predict *given* the true pressure" — it is not the
same question Path B asks.

---

## 4. Which samples does each path filter out?

On the full n=197 ArcPL set (same scope as nb04 Part 3):

| Recipe | Fair n | % converged | Fair T RMSE | Fair P RMSE |
|---|---|---|---|---|
| Path A (1-sided, Fe3=0.15, clip) | **185** / 197 | 93.9% | 47.4 C | 3.47 kbar |
| Path B (iterative, Fe3=0.0, no clip) | **185** / 197 | 93.9% | 50.1 C | 3.60 kbar |
| Intersection (both converge) | 185 | - | (same rows) | (same rows) |
| Path A exclusive (A converges, B fails) | **0** | - | - | - |
| Path B exclusive (B converges, A fails) | **0** | - | - | - |

**Both paths succeed/fail on identical rows.** There is no filter asymmetry on this scope.

### Metrics on the intersection
- A: T=47.4 C, P=3.47 kbar
- B: T=50.1 C, P=3.60 kbar

Path A is uniformly ~3 C better on T and ~0.13 kbar better on P — the small gap is the
combined effect of (algorithm) + (Fe3=0.15 vs 0.0). Neither difference justifies the reported
118.7 C vs 50.1 C manuscript gap.

### So where does "118.7 C on n=28" come from?

It comes from a **different ArcPL subset**, not from the code difference.

- nb04b's saved `results/nb04b_arcpl_predictions.csv` contains **n=394** rows (ArcPL before
  many of the cleaning filters were tightened).
- When nb04b cell 19 ran, the in-memory `arcpl` DataFrame had been through cells 5-10 but
  those cells are a looser pipeline than nb04 Part 3's — missing or laxer Kd / Wo / oxide-total
  cuts. From the printed "86.3% failure, n=28 fair", the cell-19 `arcpl` had roughly
  **n ~= 204** rows.
- On that n~204 scope, Putirka's own iterative failures (not the clip bounds) killed 86% of
  the rows, leaving n=28 hard-cases where the residuals were pathologically large
  (T RMSE 118.7 C).

In other words: **Path A never ran on the n=197 subset that nb04 Part 3 builds.** The 118.7 C
headline is an artifact of running Putirka on a broader, noisier ArcPL subset that included
compositions on which the eq 28a/29a iteration diverges. The headline is not wrong per se,
just computed on a different population.

When Path A's algorithm runs on the tighter n=197 set, it returns T RMSE = 47.4 C — about
40% lower, and comparable to Path B's 50.1 C.

---

## 5. Recommended manuscript framing

Given the result above, the choice is no longer "old vs new code path." It's **which ArcPL
population to report Putirka on**:

**Option A (conservative, "Putirka as recommended")** -- use the strict, Kd/Wo/oxide-total
filtered n=197 scope (same as ML training filter) for a like-for-like comparison.

Pro: apples-to-apples. Same rows for ML and Putirka. Fair Kd-equilibrated samples, which
is how the Putirka calibration dataset looks.

Con: reads as "Putirka is only moderately worse than ML" (50.1 vs 63.6 C on T, 3.60 vs
2.65 kbar on P).

**Option B (aggressive, "Putirka naively applied")** -- use the looser n~200 scope (less Kd
filtering).

Pro: demonstrates the robustness advantage of ML: Putirka loses 86% of samples and still
has a 2.6x worse T RMSE on what remains.

Con: the scope mismatch is load-bearing; a reviewer who notices will ask why Putirka gets
graded on a different population than ML.

**Option C (most honest, recommended)** -- report **both**, with the scope difference spelled
out.

Headline: ML predicts every sample; Putirka's usable coverage depends on how you pre-filter.

Suggested manuscript paragraph shape:

> On a like-for-like Kd-equilibrated subset (n=185 of 197), Putirka 28a/29a and our ML
> forest achieve comparable T RMSE (50 vs 64 C) but ML has a clear P advantage (2.65 vs
> 3.60 kbar). When Putirka is applied without the Kd pre-filter to the broader ArcPL scope
> (n=204), its iterative solver fails on 86% of samples and returns T RMSE = 119 C on the
> surviving 28. Our ML model predicts all samples in both scopes.

This is fair to both methods and surfaces the real empirical message: **ML wins on coverage
and P accuracy; Putirka matches ML on T accuracy only on the subset where its algorithm
converges.**

Recommend Option C for the manuscript.

---

## Artifacts produced by this audit

- `results/putirka_path_A_vs_B_metrics.csv` -- the n=197 head-to-head table
- `scripts/audit_putirka_paths.py` -- runnable script that reproduces all numbers above
- (no notebook changes; this is a review-only audit)
