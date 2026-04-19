# v10 notebook markdown template

**Status:** canonical for every code cell in every v10 notebook
**Author:** NQTa
**Date:** 2026-04-16
**Use:** before every code cell, a markdown cell with this exact structure

---

## 1. Purpose

Every code cell in every v10 notebook carries a preceding markdown cell with five required sections. This is non-negotiable. Reviewers, future Claude instances, your PhD advisor, and you-in-six-months all need this to read the code without loading the whole pipeline into working memory.

---

## 2. Template

```markdown
### [cell purpose in one sentence, sentence case, no period]

**Input:** [what comes in — files, variables from prior cells, parameters. Include file paths and variable names literally.]

**Output:** [what goes out — files written, figures saved, variables created. Include absolute paths and names.]

**Method:** [2-5 sentences describing what the cell does. Name the algorithm, not just the library. If it calls into `src/*.py`, name the function.]

**Connection:** [how this feeds the bigger pipeline. Which downstream cell / notebook / figure depends on this output. Which upstream cell this consumes from.]

**Why:** [scientific justification. Why this specific algorithm / data / threshold was chosen over alternatives. 1-3 sentences. If a test in `v10_nb03_test_protocol.md` justifies this, cite the test ID (e.g. "Shipped per T04 which showed N_AUG=1 beats N_AUG=5 on test RMSE").]
```

---

## 3. Filled example

```markdown
### Fit Ridge stacking meta-model on OOF predictions from 4 base learners

**Input:**
- `oof_predictions_RF` (np.ndarray, shape (n_train,), from cell 3.10.2)
- `oof_predictions_ERT` (np.ndarray, shape (n_train,), from cell 3.10.3)
- `oof_predictions_XGB` (np.ndarray, shape (n_train,), from cell 3.10.4)
- `oof_predictions_GB` (np.ndarray, shape (n_train,), from cell 3.10.5)
- `y_train` (np.ndarray, shape (n_train,), from `data/splits/train_indices_opx_liq.npy`)

**Output:**
- `models/meta_ridge_T_C_opx_liq_stacked.joblib` (fitted RidgeCV)
- `results/nb03_stacking_diagnostics.json` (alpha_, coef_, per-base OOF RMSE, correlations)
- variable `meta_T_C_opx_liq` available to downstream cells

**Method:** Column-stack the 4 OOF vectors to `(n_train, 4)`. Fit `RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], cv=5)` against `y_train`. Flag if `alpha_` hits either endpoint of the grid (would indicate collinearity or too-narrow grid). Calls `src.stacking.fit_ridge_meta_model`.

**Connection:** Consumed by cell 3.10.7 (stacked test-set predictions), NB04 cell 4 (ArcPL benchmark), NB05 cell 5 (LOSO on stacked), NB06 cell 7 (linear-SHAP on meta). Upstream: the OOF generation cells 3.10.2-3.10.5.

**Why:** Ridge regression provides regularized linear blending when base predictions are highly correlated (documented at 0.86-0.99 OOF correlation in v9). Unregularized linear regression is unstable under this collinearity; nonlinear meta-models add variance without gain at 4 features. Shipped per test T02 which confirmed stacking helps ArcPL T by 2.65 C over best base. See `docs/stacking_strategy.md` v9 outcome block.
```

---

## 4. Short form — when the full template is overkill

For utility cells that are genuinely obvious (imports, setup, print statements), a one-line markdown header is acceptable:

```markdown
### Imports and seed setup
```

Use short form **only** for cells that would not confuse a first-time reader. When in doubt, use the full template.

---

## 5. What not to do

- Do not use ambiguous pronouns ("it", "the data") without antecedent. Say `opx_liq_train_df` not "the data."
- Do not write "see the code below" — the whole point of the template is that the markdown describes the code before you read it.
- Do not skip sections. If a cell has no obvious "Why" (e.g. pure boilerplate), escalate to the short form instead.
- Do not narrate line-by-line what the code does. The **Method** section is strategic summary, not pseudocode.
- Do not use the same header text on two cells in the same notebook. Headers must be unique enough to appear in the notebook outline cleanly.

---

## 6. Enforcement

Two checks during Phase D notebook rebuild:

1. Automated lint: `scripts/v10_audit_notebook_markdown.py` verifies every code cell has a preceding markdown cell and that the markdown contains the required section headers (Input, Output, Method, Connection, Why) OR is an approved short-form header.

2. Manual spot-check before Phase E: pull 5 random code cells per notebook, read only the markdown, ask "can I predict what the code does?" If not, rewrite the markdown.

---

## 7. Markdown for section headers (not code cells)

Section-level markdown (H1, H2) introduces groups of code cells. Structure:

```markdown
# Phase 3.10: Ridge stacking

Out-of-fold prediction generation, meta-model fit, sanity checks, diagnostics export. Canonical for opx_liq. See `docs/stacking_strategy.md`.

## 3.10.1 Setup
## 3.10.2 RF OOF
## 3.10.3 ERT OOF
...
```

Phase headers get a paragraph of context. Subsection headers get the per-cell template.

---

## 8. Conclusion

Rigor costs typing time. Reviewer confusion costs weeks. The template pays for itself on the first revision round.
