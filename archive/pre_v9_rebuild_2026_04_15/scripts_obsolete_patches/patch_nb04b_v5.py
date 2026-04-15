"""Patch nb04b_lepr_arcpl_validation.ipynb with v5 reframe cells.

Inserts after cell-010-d1da047a:
  - Part 1 reframe markdown
  - bootstrap CIs per scope
  - Wilcoxon measured vs VBD on |residuals|
  - 4-panel figure (pred vs obs T/P + residual histograms)
  - operator decision block
"""
from __future__ import annotations

from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / 'notebooks' / 'nb04b_lepr_arcpl_validation.ipynb'

nb = nbformat.read(str(NB_PATH), as_version=4)
cells = list(nb.cells)

# Find anchor cell.
anchor_id = 'cell-010-d1da047a'
anchor_idx = next(i for i, c in enumerate(cells) if c.get('id') == anchor_id)
# Don't double-insert.
if any(c.get('id', '').startswith('v5-nb04b') for c in cells):
    print('v5 cells already present, aborting')
    raise SystemExit(0)


def md(src, cid):
    c = nbformat.v4.new_markdown_cell(source=src)
    c.id = cid
    return c


def code(src, cid):
    c = nbformat.v4.new_code_cell(source=src)
    c.id = cid
    c.outputs = []
    c.execution_count = None
    return c


INTRO = '''## Part 1: Reframe external validation (v5)

External-validation RMSE with bootstrap 95% CIs (B = 2000) across three scopes
(all ArcPL, measured-H2O subset, VBD subset), plus a paired Wilcoxon test on
residual magnitudes between the measured-H2O and VBD subsets. The 4-panel
figure (pred vs obs T/P plus residual histograms T/P) replaces the earlier
two-column scatter.
'''

BOOTSTRAP = '''# Part 1a: bootstrap RMSE CIs (B=2000) across three scopes.
from scipy import stats as _stats

BOOT_B = 2000
_rng = np.random.default_rng(42)


def _rmse_ci(y_true, y_pred, mask, B=BOOT_B, rng=_rng):
    m = mask & np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 2:
        return np.nan, np.nan, np.nan, int(m.sum())
    yt = y_true[m]; yp = y_pred[m]
    n = len(yt)
    point = float(np.sqrt(np.mean((yt - yp) ** 2)))
    idx = rng.integers(0, n, size=(B, n))
    boots = np.sqrt(np.mean((yt[idx] - yp[idx]) ** 2, axis=1))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return point, float(lo), float(hi), n


scopes = {
    'ArcPL all':          np.ones(len(arcpl), dtype=bool),
    'ArcPL measured H2O': (~arcpl['is_vbd']).values,
    'ArcPL VBD H2O':      arcpl['is_vbd'].values,
}

boot_rows = []
for scope_name, scope_mask in scopes.items():
    for tgt, ycol, pcol in [('T', 'T_C', 'T_pred'), ('P', 'P_kbar', 'P_pred')]:
        y = arcpl[ycol].values.astype(float)
        p = arcpl[pcol].values.astype(float)
        rmse, lo, hi, n = _rmse_ci(y, p, scope_mask)
        boot_rows.append({
            'scope':      scope_name,
            'target':     tgt,
            'n':          n,
            'rmse':       rmse,
            'rmse_ci_lo': lo,
            'rmse_ci_hi': hi,
        })

boot_df = pd.DataFrame(boot_rows)
boot_df.to_csv(RESULTS / 'nb04b_arcpl_rmse_bootstrap.csv', index=False)

print('ArcPL RMSE with 95% bootstrap CIs (B=2000):')
for _, r in boot_df.iterrows():
    if np.isfinite(r['rmse']):
        print(f"  {r['scope']:22s} {r['target']}  n={int(r['n']):4d}  "
              f"RMSE={r['rmse']:.2f} [{r['rmse_ci_lo']:.2f}, {r['rmse_ci_hi']:.2f}]")
    else:
        print(f"  {r['scope']:22s} {r['target']}  n={int(r['n'])}  (insufficient)")
'''

WILCOXON = '''# Part 1b: Wilcoxon test - is the measured-H2O subset significantly better
# than the VBD subset in residual magnitude? (two-sample Mann-Whitney, since
# these are different samples, not paired.)
wilc_rows = []
for tgt, ycol, pcol, unit in [('T', 'T_C', 'T_pred', 'C'),
                               ('P', 'P_kbar', 'P_pred', 'kbar')]:
    y = arcpl[ycol].values.astype(float)
    p = arcpl[pcol].values.astype(float)
    res = np.abs(y - p)
    m_meas = (~arcpl['is_vbd']).values & np.isfinite(res)
    m_vbd  = arcpl['is_vbd'].values & np.isfinite(res)
    r_meas = res[m_meas]; r_vbd = res[m_vbd]
    if len(r_meas) < 3 or len(r_vbd) < 3:
        wilc_rows.append({'target': tgt, 'unit': unit,
                          'n_meas': len(r_meas), 'n_vbd': len(r_vbd),
                          'median_meas_resid': np.nan,
                          'median_vbd_resid':  np.nan,
                          'mannwhitney_U':     np.nan,
                          'p_value':           np.nan})
        continue
    U, pval = _stats.mannwhitneyu(r_meas, r_vbd, alternative='two-sided')
    wilc_rows.append({
        'target':            tgt,
        'unit':              unit,
        'n_meas':            int(len(r_meas)),
        'n_vbd':             int(len(r_vbd)),
        'median_meas_resid': float(np.median(r_meas)),
        'median_vbd_resid':  float(np.median(r_vbd)),
        'mannwhitney_U':     float(U),
        'p_value':           float(pval),
    })

wilc_df = pd.DataFrame(wilc_rows)
wilc_df.to_csv(RESULTS / 'nb04b_measured_vs_vbd_test.csv', index=False)
print('Mann-Whitney U test on |residual|, measured H2O vs VBD:')
print(wilc_df.round(4).to_string(index=False))
'''

FIGURE_4 = '''# Part 1c: 4-panel figure (pred vs obs T/P + residual histograms).
try:
    from src.plot_style import apply_style, PUTIRKA_COLOR, ML_COLOR, TOL_BRIGHT
    apply_style()
    COLOR_MEAS = TOL_BRIGHT['blue']
    COLOR_VBD  = TOL_BRIGHT['red']
except Exception:
    COLOR_MEAS = '#0072B2'
    COLOR_VBD  = '#D55E00'

fig, axes = plt.subplots(2, 2, figsize=(11, 9))

# Panels (a), (b): pred vs obs T, P (all ArcPL, colored by H2O method)
for (ax_idx, ycol, pcol, unit, tlabel) in [
    ((0, 0), 'T_C',     'T_pred',  'C',     'T'),
    ((0, 1), 'P_kbar',  'P_pred',  'kbar',  'P'),
]:
    ax = axes[ax_idx]
    y = arcpl[ycol].values.astype(float)
    p = arcpl[pcol].values.astype(float)
    m_meas = (~arcpl['is_vbd']).values
    m_vbd  =  arcpl['is_vbd'].values
    ax.scatter(y[m_meas], p[m_meas], s=24, alpha=0.7, color=COLOR_MEAS,
               edgecolor='k', lw=0.3, label=f'Measured H2O (n={int(m_meas.sum())})')
    ax.scatter(y[m_vbd],  p[m_vbd],  s=24, alpha=0.7, color=COLOR_VBD,
               edgecolor='k', lw=0.3, marker='^',
               label=f'VBD H2O (n={int(m_vbd.sum())})')
    lim = [np.nanmin([y.min(), p.min()]), np.nanmax([y.max(), p.max()])]
    ax.plot(lim, lim, 'k--', lw=1, alpha=0.7)
    rmse_all = float(np.sqrt(np.nanmean((y - p) ** 2)))
    ax.set_xlabel(f'Observed {tlabel} ({unit})')
    ax.set_ylabel(f'Predicted {tlabel} ({unit})')
    panel = 'a' if ax_idx == (0, 0) else 'b'
    ax.set_title(f'({panel}) {tlabel} pred vs obs, ArcPL (n={len(arcpl)})\\n'
                 f'RMSE = {rmse_all:.2f} {unit}')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

# Panels (c), (d): residual histograms
for (ax_idx, ycol, pcol, unit, tlabel, panel) in [
    ((1, 0), 'T_C',    'T_pred',  'C',    'T', 'c'),
    ((1, 1), 'P_kbar', 'P_pred',  'kbar', 'P', 'd'),
]:
    ax = axes[ax_idx]
    y = arcpl[ycol].values.astype(float)
    p = arcpl[pcol].values.astype(float)
    res = p - y
    m_meas = (~arcpl['is_vbd']).values & np.isfinite(res)
    m_vbd  =  arcpl['is_vbd'].values & np.isfinite(res)
    res_meas = res[m_meas]; res_vbd = res[m_vbd]
    if len(res_meas) > 0 and len(res_vbd) > 0:
        lo, hi = np.nanmin(res), np.nanmax(res)
    else:
        lo, hi = -1, 1
    bins = np.linspace(lo, hi, 25)
    if len(res_meas) > 0:
        ax.hist(res_meas, bins=bins, alpha=0.55, color=COLOR_MEAS,
                edgecolor='k', lw=0.3,
                label=f'Measured (mean {np.mean(res_meas):+.2f})')
    if len(res_vbd) > 0:
        ax.hist(res_vbd, bins=bins, alpha=0.55, color=COLOR_VBD,
                edgecolor='k', lw=0.3,
                label=f'VBD (mean {np.mean(res_vbd):+.2f})')
    ax.axvline(0, color='k', lw=0.7)
    ax.set_xlabel(f'Residual {tlabel} (predicted - observed, {unit})')
    ax.set_ylabel('Count')
    ax.set_title(f'({panel}) {tlabel} residual distribution (n={len(arcpl)})')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out_png = FIGURES / 'fig_nb04b_arcpl_4panel.png'
out_pdf = FIGURES / 'fig_nb04b_arcpl_4panel.pdf'
fig.savefig(out_png, dpi=300, bbox_inches='tight')
fig.savefig(out_pdf, bbox_inches='tight')
plt.show()
plt.close(fig)
print(f'Wrote {out_png.name} and {out_pdf.name}')
'''

DECISION = '''# Part 1d: OPERATOR DECISION REQUIRED - ArcPL framing.
# This prints the external-validation summary and does not auto-select.

bd = boot_df.set_index(['scope', 'target'])
wd = wilc_df.set_index('target')

print('=' * 72)
print('OPERATOR DECISION REQUIRED - NB04b external-validation framing')
print('=' * 72)
print()
print(f'Dataset: ArcPL subset of LEPR, n = {len(arcpl)} samples')
for sc in ['ArcPL all', 'ArcPL measured H2O', 'ArcPL VBD H2O']:
    for tgt, unit in [('T', 'C'), ('P', 'kbar')]:
        if (sc, tgt) in bd.index:
            r = bd.loc[(sc, tgt)]
            if np.isfinite(r['rmse']):
                print(f'  {sc:22s} {tgt}  n={int(r["n"]):4d}  '
                      f'RMSE={r["rmse"]:.2f} [{r["rmse_ci_lo"]:.2f}, '
                      f'{r["rmse_ci_hi"]:.2f}] {unit}')
for tgt in ['T', 'P']:
    if tgt in wd.index:
        r = wd.loc[tgt]
        print(f'  Mann-Whitney {tgt}: p = {r["p_value"]:.4g}  '
              f'(median |resid| measured {r["median_meas_resid"]:.2f} vs '
              f'VBD {r["median_vbd_resid"]:.2f})')
print()
print('Framing options:')
print('  A. "Comparable performance on measured-H2O subset": headline the')
print('     measured-H2O bootstrap CI (apples-to-apples vs experimental training).')
print('  B. "VBD inflates error": headline the measured-vs-VBD gap and the')
print('     Mann-Whitney p, arguing measured H2O is required for valid')
print('     external benchmarking.')
print('  C. "Hybrid": report all three scopes + the figure; let the reader see')
print('     both the measured-subset parity and the VBD degradation.')
print()
print('Tell Claude which framing (A / B / C) to keep in the manuscript body.')
print('=' * 72)
'''

new_cells = [
    md(INTRO,     'v5-nb04b-intro'),
    code(BOOTSTRAP, 'v5-nb04b-boot'),
    code(WILCOXON,  'v5-nb04b-wilc'),
    code(FIGURE_4,  'v5-nb04b-fig4'),
    code(DECISION,  'v5-nb04b-dec'),
]

cells[anchor_idx + 1:anchor_idx + 1] = new_cells
nb.cells = cells
nbformat.write(nb, str(NB_PATH))
print(f'Inserted {len(new_cells)} cells after {anchor_id} in {NB_PATH.name}')
