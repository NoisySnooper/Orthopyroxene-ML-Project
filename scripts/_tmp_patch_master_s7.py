import os

p = r"C:\Users\NQTa\Documents\MLCourse\Final Project\docs\v10_master_plan.md"
with open(p, "r", encoding="utf-8") as f:
    txt = f.read()

old_s7 = """## 7. Test-first protocol summary

See `v10_nb03_test_protocol.md` for details. Twelve tests per pipeline:

| Test | Hypothesis |
|---|---|
| T01 | Boosted is primary |
| T02 | Ensemble beats best base (compares 4 ensemble methods) |
| T03 | Resampling hurts (replicates v9 finding; re-tests per pipeline) |
| T04 | N_AUG=1 beats N_AUG=5 |
| T05 | Feature set winners reproduce v9 (opx only) / are stable (cpx, twopx) |
| T06 | Mineral-only pipeline is worse than +liq (for opx_only and cpx_only) |
| T07 | Composition-conditional bias correction improves ArcPL T |
| T08 | P piecewise bias correction improves test P |
| T09 | CV-predict stacking beats full-fit stacking |
| T10 | IsolationForest OOD correlates with residual magnitude |
| T11 | CatBoost or LightGBM beats XGB on any pipeline (NEW) |
| T12 | MLP or ElasticNet beats tree models on any pipeline (NEW) |

Each test runs per pipeline. Results logged to `results/v10_{pipeline}_test_log.csv` with pass/fail."""

new_s7 = """## 7. Test-first protocol summary

See `v10_nb03_test_protocol.md` for details. **Twelve tests per primary pipeline plus two additional tests in the universal exploration track.**

### Primary tests T01-T12 (opx, cpx, twopx pipelines)

| Test | Hypothesis |
|---|---|
| T01 | Boosted is primary |
| T02 | Ensemble beats best base (compares 4 ensemble methods) |
| T03 | Resampling hurts (replicates v9 finding; re-tests per pipeline) |
| T04 | N_AUG=1 beats N_AUG=5 |
| T05 | Feature set winners reproduce v9 (opx only) / are stable (cpx, twopx) |
| T06 | Mineral-only pipeline is worse than +liq (for opx_only and cpx_only) |
| T07 | Composition-conditional bias correction improves ArcPL T |
| T08 | P piecewise bias correction improves test P |
| T09 | CV-predict stacking beats full-fit stacking |
| T10 | IsolationForest OOD correlates with residual magnitude |
| T11 | CatBoost or LightGBM beats XGB on any pipeline (NEW) |
| T12 | MLP or ElasticNet beats tree models on any pipeline (NEW) |

Each test runs per pipeline. Results logged to `results/v10_nb03_test_log.csv` with `pipeline` column.

### Universal-only tests T13-T14 (isolated)

Defined in `docs/v10_universal_model_exploration.md`. Run only in `nb03_universal_exploration.ipynb`. Logged to separate `results/universal/v10_nb03_test_log.csv`. Never combined with primary pipeline results.

| Test | Hypothesis |
|---|---|
| T13 | Universal model degrades gracefully (1-phase < 2-phase < 3-phase in R^2 ordering) |
| T14 | Universal with all 3 phases beats specialized opx-liq / cpx-liq / twopx models on same sample |

Decision tree: T13 fails -> ablate universal architecture, exploration ends. T13 passes but T14 fails -> keep as honest-negative exploration artifact. Both pass -> universal becomes lead novelty claim in cpx paper."""

if old_s7 not in txt:
    print("ERROR: old Section 7 not found verbatim in file")
else:
    new_txt = txt.replace(old_s7, new_s7)
    with open(p, "w", encoding="utf-8") as f:
        f.write(new_txt)
    print(f"Replaced Section 7. File size: {os.path.getsize(p)} bytes")
