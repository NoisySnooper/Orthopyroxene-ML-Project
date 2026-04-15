"""Append the three-way ML benchmark (v6 Part 5) to nb04_putirka_benchmark.ipynb.

Idempotent: if a cell tagged `v6-three-way` already exists, replace that block
in-place instead of duplicating.

Sections added:
1. Markdown header
2. Load cpx + liq + opx merged ArcPL data
3. Run Agreda-Lopez (cpx-only + cpx-liq) on that data
4. Run Jorgenson (cpx-only + cpx-liq) via Thermobar_onnx
5. Run Wang 2021 cpx-liq via Thermobar
6. Run Putirka 2008 cpx-liq via Thermobar
7. Assemble comparison table + bootstrap CI
8. 4-panel figure (pred-vs-obs T, pred-vs-obs P, grouped bar RMSE, coverage)
9. Post-figure markdown explainer
"""
from __future__ import annotations

from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / 'notebooks' / 'nb04_putirka_benchmark.ipynb'

TAG = 'v6-three-way'

HEADER_MD = """## Three-way ML benchmark on ArcPL (v6)

**Why this comparison.** Our model predicts P and T from opx composition.
Agreda-Lopez 2024, Jorgenson 2022, and Wang 2021 all predict P/T from cpx
(cpx-only or cpx-liq). These phases are physically distinct but record the
same magmatic conditions when co-existing in the same experiment. Running all
four models on the same ArcPL experiments answers: does opx-liq ML achieve
error rates comparable to the state-of-the-art cpx-liq ML method, given that
opx is inherently less pressure-sensitive than cpx?

**What we run.**
- Ours: opx-liq RF (canonical), predictions already cached in nb04b output
- Agreda-Lopez 2024 ExtraTrees cpx-only (T + P) + cpx-liq (T + P)
- Jorgenson 2022 cpx-only RF (T + P)
- Wang 2021 cpx-liq (T eq2, P eq1)
- Putirka 2008 cpx-liq (T eq33, P eq30) as the classical benchmark

**What we report.**
- `results/nb04_three_way_ml_benchmark.csv` — RMSE + 95% bootstrap CI + R2 + coverage
- `figures/fig_nb04_three_way.{png,pdf}` — 4-panel overlay
"""

LOAD_CODE = """# Three-way ML benchmark - load merged LEPR ArcPL data (cpx + opx + liq + true P/T)
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import numpy as np
import pandas as pd
from scipy import stats

from config import LEPR_XLSX, RESULTS, FIGURES, MODELS, SEED_BOOTSTRAP
from src.external_models import (
    predict_agreda_from_df, predict_jorgenson, predict_wang,
    predict_putirka_cpx_liq, AGREDA_CPX_COLS, AGREDA_LIQ_COLS,
)

xls = pd.ExcelFile(LEPR_XLSX)
cpx = pd.read_excel(xls, sheet_name='Cpx')
liq = pd.read_excel(xls, sheet_name='Liq')
opx = pd.read_excel(xls, sheet_name='Opx')

# Cast all oxide columns to numeric; LEPR has string placeholders in minor oxides.
def _numerify(df, suffix):
    for c in df.columns:
        if c.endswith(suffix):
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
    return df

cpx = _numerify(cpx.drop_duplicates('Experiment'), '_Cpx')
liq = _numerify(liq.drop_duplicates('Experiment'), '_Liq')
opx = _numerify(opx.drop_duplicates('Experiment'), '_Opx')

# Merge cpx + liq + opx on Experiment. Outer merge keeps experiments that have
# only a subset, which we mask per-method below.
m = cpx.merge(liq, on='Experiment', how='outer', suffixes=('', '_liq_dup'))
m = m.merge(opx, on='Experiment', how='outer', suffixes=('', '_opx_dup'))
# Pick ground-truth T and P from cpx sheet first, fall back to liq, fall back to opx.
for tgt in ['T_K', 'P_kbar']:
    for src in ['', '_liq_dup', '_opx_dup']:
        col = f'{tgt}{src}'
        if col in m.columns:
            if tgt not in m.columns:
                m[tgt] = m[col]
            else:
                m[tgt] = m[tgt].combine_first(m[col])
m['T_C'] = pd.to_numeric(m['T_K'], errors='coerce') - 273.15
m['P_kbar'] = pd.to_numeric(m['P_kbar'], errors='coerce')
# Wang/Putirka require Fe3Fet_Liq; default to 0 (reduced Fe reporting convention).
if 'Fe3Fet_Liq' not in m.columns:
    m['Fe3Fet_Liq'] = 0.0

# Scope to the ArcPL-filtered subset from v5 nb04b so the comparison stays fair
# across methods (cpx vs opx-liq) on the same natural-sample experiments.
arcpl_preds = pd.read_csv(RESULTS / 'nb04b_arcpl_predictions.csv')
keep_exp = set(arcpl_preds['Experiment'].astype(str))
m = m[m['Experiment'].astype(str).isin(keep_exp)].reset_index(drop=True)

has_cpx = m[AGREDA_CPX_COLS].apply(pd.to_numeric, errors='coerce').fillna(0.0).sum(axis=1) > 80
has_liq = m[AGREDA_LIQ_COLS].apply(pd.to_numeric, errors='coerce').fillna(0.0).sum(axis=1) > 80
m = m[has_cpx & np.isfinite(m['T_C']) & np.isfinite(m['P_kbar'])].reset_index(drop=True)
print(f'Three-way benchmark set (ArcPL scope): n={len(m)} experiments')
print(f'  of which n_cpx_liq={int(has_liq.sum())}')
"""

RUN_CODE = """# Run all five methods (Ours opx-liq, Agreda x2, Jorgenson, Wang, Putirka).
y_T = m['T_C'].values
y_P = m['P_kbar'].values

preds = {}  # name -> (T, P)
# 1. Agreda-Lopez cpx-only
a_cpx_T = predict_agreda_from_df(m, MODELS / 'external', 'cpx_only', 'T')['median']
a_cpx_P = predict_agreda_from_df(m, MODELS / 'external', 'cpx_only', 'P')['median']
preds['Agreda-Lopez cpx-only']  = (a_cpx_T, a_cpx_P)

# 2. Agreda-Lopez cpx-liq (where liq present)
try:
    a_liq_T = predict_agreda_from_df(m, MODELS / 'external', 'cpx_liq', 'T')['median']
    a_liq_P = predict_agreda_from_df(m, MODELS / 'external', 'cpx_liq', 'P')['median']
    preds['Agreda-Lopez cpx-liq'] = (a_liq_T, a_liq_P)
except KeyError as e:
    print(f'Agreda cpx-liq skipped ({e})')

# 3. Jorgenson cpx-only (via Thermobar_onnx)
try:
    j_T = predict_jorgenson(m, 'T', phase='cpx_only', P_kbar=y_P)
    j_P = predict_jorgenson(m, 'P', phase='cpx_only', T_K=y_T + 273.15)
    preds['Jorgenson cpx-only'] = (j_T, j_P)
except Exception as e:
    print(f'Jorgenson skipped ({e})')

# 4. Wang 2021 cpx-liq
try:
    w_T = predict_wang(m, 'T', P_kbar=y_P)
    w_P = predict_wang(m, 'P', T_K=y_T + 273.15)
    preds['Wang 2021 cpx-liq'] = (w_T, w_P)
except Exception as e:
    print(f'Wang skipped ({e})')

# 5. Putirka 2008 cpx-liq eq33/30
try:
    p_T = predict_putirka_cpx_liq(m, 'T', P_kbar=y_P)
    p_P = predict_putirka_cpx_liq(m, 'P', T_K=y_T + 273.15)
    preds['Putirka 2008 cpx-liq'] = (p_T, p_P)
except Exception as e:
    print(f'Putirka skipped ({e})')

# 6. Ours opx-liq RF (from canonical predictions file if available)
try:
    ours_df = pd.read_csv(RESULTS / 'nb04b_arcpl_predictions.csv')
    # align by Experiment if possible, else use the order of m
    if 'Experiment' in ours_df.columns:
        o = m.merge(ours_df[['Experiment', 'T_pred', 'P_pred']],
                    on='Experiment', how='left')
        preds['Ours opx-liq RF'] = (o['T_pred'].values, o['P_pred'].values)
    else:
        preds['Ours opx-liq RF'] = (ours_df['T_pred'].values[:len(m)],
                                     ours_df['P_pred'].values[:len(m)])
except Exception as e:
    print(f'Ours opx-liq skipped ({e})')

print('\\nMethods available:', list(preds.keys()))
"""

METRICS_CODE = """# Per-method metrics (RMSE + 95% bootstrap CI + R2 + coverage).
rng = np.random.default_rng(SEED_BOOTSTRAP)
BOOT = 2000

def rmse_ci(y, yhat, B=BOOT):
    mask = np.isfinite(y) & np.isfinite(yhat)
    n = int(mask.sum())
    if n < 3:
        return np.nan, np.nan, np.nan, n, np.nan
    yt = y[mask]; yp = yhat[mask]
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    idx = rng.integers(0, n, size=(B, n))
    boots = np.sqrt(np.mean((yt[idx] - yp[idx]) ** 2, axis=1))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return rmse, float(lo), float(hi), n, r2

rows = []
for name, (pT, pP) in preds.items():
    mT = rmse_ci(y_T, pT)
    mP = rmse_ci(y_P, pP)
    rows.append({
        'Method': name,
        'T_n': mT[3], 'T_RMSE': mT[0], 'T_RMSE_CI_lo': mT[1], 'T_RMSE_CI_hi': mT[2], 'T_R2': mT[4],
        'T_coverage_pct': 100 * mT[3] / len(m),
        'P_n': mP[3], 'P_RMSE': mP[0], 'P_RMSE_CI_lo': mP[1], 'P_RMSE_CI_hi': mP[2], 'P_R2': mP[4],
        'P_coverage_pct': 100 * mP[3] / len(m),
    })
three_way = pd.DataFrame(rows)
three_way.to_csv(RESULTS / 'nb04_three_way_ml_benchmark.csv', index=False)
print(three_way.round(3).to_string(index=False))
"""

FIG_CODE = """# 4-panel three-way comparison figure.
import matplotlib.pyplot as plt
from src.plot_style import apply_style, OKABE_ITO
apply_style()

COLORS = {
    'Ours opx-liq RF':        OKABE_ITO['blue'],
    'Agreda-Lopez cpx-liq':   OKABE_ITO['orange'],
    'Agreda-Lopez cpx-only':  OKABE_ITO['yellow'],
    'Jorgenson cpx-only':     OKABE_ITO['green'],
    'Wang 2021 cpx-liq':      OKABE_ITO['vermillion'],
    'Putirka 2008 cpx-liq':   OKABE_ITO['sky_blue'],
}

fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

# Panel A: predicted vs true T
ax = axes[0, 0]
lim_T = [400, 1500]
ax.plot(lim_T, lim_T, 'k--', lw=1, alpha=0.7)
for name, (pT, pP) in preds.items():
    mk = np.isfinite(y_T) & np.isfinite(pT)
    if mk.sum() == 0: continue
    ax.scatter(y_T[mk], pT[mk], s=15, alpha=0.5,
               color=COLORS.get(name, '#777'), edgecolor='k', lw=0.2,
               label=f'{name} (n={int(mk.sum())})')
ax.set(xlabel='True T (C)', ylabel='Predicted T (C)',
       title=f'(a) Predicted vs true T (n total={len(m)})', xlim=lim_T, ylim=lim_T)
ax.legend(fontsize=7, loc='upper left')
ax.grid(alpha=0.3)

# Panel B: predicted vs true P
ax = axes[0, 1]
lim_P = [-1, 30]
ax.plot(lim_P, lim_P, 'k--', lw=1, alpha=0.7)
for name, (pT, pP) in preds.items():
    mk = np.isfinite(y_P) & np.isfinite(pP)
    if mk.sum() == 0: continue
    ax.scatter(y_P[mk], pP[mk], s=15, alpha=0.5,
               color=COLORS.get(name, '#777'), edgecolor='k', lw=0.2,
               label=f'{name} (n={int(mk.sum())})')
ax.set(xlabel='True P (kbar)', ylabel='Predicted P (kbar)',
       title=f'(b) Predicted vs true P (n total={len(m)})', xlim=lim_P, ylim=lim_P)
ax.legend(fontsize=7, loc='upper left')
ax.grid(alpha=0.3)

# Panel C: RMSE grouped bar with CI
ax = axes[1, 0]
x = np.arange(len(three_way))
w = 0.35
bars_T = ax.bar(x - w/2, three_way['T_RMSE'].values / 100, w,
                yerr=np.array([three_way['T_RMSE'].values - three_way['T_RMSE_CI_lo'].values,
                               three_way['T_RMSE_CI_hi'].values - three_way['T_RMSE'].values]) / 100,
                color=[COLORS.get(n, '#777') for n in three_way['Method']],
                edgecolor='k', label='T RMSE / 100 (C)')
bars_P = ax.bar(x + w/2, three_way['P_RMSE'].values, w,
                yerr=np.array([three_way['P_RMSE'].values - three_way['P_RMSE_CI_lo'].values,
                               three_way['P_RMSE_CI_hi'].values - three_way['P_RMSE'].values]),
                color=[COLORS.get(n, '#777') for n in three_way['Method']],
                edgecolor='k', hatch='//', label='P RMSE (kbar)')
ax.set_xticks(x)
ax.set_xticklabels(three_way['Method'].values, rotation=30, ha='right', fontsize=8)
ax.set_ylabel('RMSE (T/100 C or kbar)')
ax.set_title('(c) RMSE with 95% bootstrap CI')
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.3)

# Panel D: coverage
ax = axes[1, 1]
ax.bar(x, three_way['T_coverage_pct'].values,
       color=[COLORS.get(n, '#777') for n in three_way['Method']],
       edgecolor='k')
ax.set_xticks(x)
ax.set_xticklabels(three_way['Method'].values, rotation=30, ha='right', fontsize=8)
ax.set_ylabel('Coverage (% of ArcPL samples)')
ax.set_title('(d) Fraction of ArcPL each method can predict')
ax.set_ylim(0, 110)
ax.grid(axis='y', alpha=0.3)

for p in (FIGURES / 'fig_nb04_three_way.png', FIGURES / 'fig_nb04_three_way.pdf'):
    fig.savefig(p, bbox_inches='tight', dpi=300 if p.suffix == '.png' else 'figure')
plt.show()
print('Wrote fig_nb04_three_way.{png,pdf}')
"""

EXPLAIN_MD = """**How to read the three-way benchmark figure.**

- **Panel (a)/(b):** Every point is the same ArcPL experiment seen through a
  different thermobarometer. Closer to the 1:1 dashed line is better. Vertical
  spread at a given x value shows between-method disagreement on the same
  sample.
- **Panel (c):** Bar height is RMSE (T scaled by 1/100 so T and P share a
  y-axis); error bars are 95% bootstrap CIs. Overlapping CIs mean no
  statistically significant gap.
- **Panel (d):** Bar height is the fraction of ArcPL each method successfully
  predicts. Classical Putirka drops coverage when any required oxide is missing
  or out-of-range; ML methods return a prediction on every row that has the
  required phase data.

**Phase-mismatch caveat.** Agreda-Lopez, Jorgenson, and Wang predict from cpx;
ours predicts from opx. Direct RMSE comparison treats them as alternative tools
for the same sample - the practical deployment question. It is not a claim that
opx and cpx carry the same information content.
"""


def _del_by_tag(nb, tag):
    new_cells = []
    for c in nb.cells:
        md = c.get('metadata', {})
        if md.get('id') == tag:
            continue
        new_cells.append(c)
    nb.cells = new_cells


def main():
    nb = nbformat.read(str(NB), as_version=4)
    _del_by_tag(nb, TAG)
    cells = [
        nbformat.v4.new_markdown_cell(HEADER_MD, metadata={'id': TAG}),
        nbformat.v4.new_code_cell(LOAD_CODE,    metadata={'id': TAG}),
        nbformat.v4.new_code_cell(RUN_CODE,     metadata={'id': TAG}),
        nbformat.v4.new_code_cell(METRICS_CODE, metadata={'id': TAG}),
        nbformat.v4.new_code_cell(FIG_CODE,     metadata={'id': TAG}),
        nbformat.v4.new_markdown_cell(EXPLAIN_MD, metadata={'id': TAG}),
    ]
    nb.cells.extend(cells)
    nbformat.write(nb, str(NB))
    print(f'appended {len(cells)} cells tagged {TAG} to {NB.name}')


if __name__ == '__main__':
    main()
