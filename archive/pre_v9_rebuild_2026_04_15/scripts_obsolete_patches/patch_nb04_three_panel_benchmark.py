"""v7 Part H+: replace nb04 cell 26 with three-panel method-benchmark figure.

Produces three figures (one per scope), each with panels [T RMSE | P RMSE
| Coverage] sharing the same method ordering on the y-axis, per the
product spec:

- ArcPL three-phase paired (headline): fig_nb04_method_benchmark_paired.*
- Full LEPR three-phase paired (supp): fig_nb04_method_benchmark_lepr_full.*
- ArcPL cpx-bearing scope (supp):     fig_nb04_method_benchmark_cpx_scope.*

Per-scope method metrics also written to nb04_method_benchmark_<scope>.csv.

Methods with <20% coverage on the scope are dropped from the figure but
kept in the CSV. Methods with <90% coverage get a dynamic superscript
referencing a footnote block below the figure.

Scientific caveat noted in the caption: Ours opx-liq on lepr_full overlaps
with the training set; readers should interpret that scope as an upper
bound on fit quality, not a held-out benchmark. arcpl_paired and
cpx_scope use the ArcPL held-out natural samples so remain fair.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "nb04_putirka_benchmark.ipynb"

MARKER = "# v7 Part H+: three-panel method benchmark across three scopes"

NEW_CELL26 = r'''# v7 Part H+: three-panel method benchmark across three scopes
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import joblib
from src.plot_style import apply_style
from src.data import canonical_model_path, canonical_model_spec
from src.features import build_feature_matrix
from config import FAMILY_COLORS
from src.plot_style import OKABE_ITO

apply_style()

COLORS = {
    'Ours opx-liq forest':    FAMILY_COLORS['forest'],
    'Ours opx-liq boosted':   FAMILY_COLORS['boosted'],
    'Agreda-Lopez cpx-liq':   FAMILY_COLORS['external_cpx'],
    'Agreda-Lopez cpx-only':  OKABE_ITO['yellow'],
    'Jorgenson cpx-only':     OKABE_ITO['green'],
    'Wang 2021 cpx-liq':      OKABE_ITO['vermillion'],
    'Putirka 2008 cpx-liq':   FAMILY_COLORS['putirka'],
}

BOOT_NEW = 2000


def _merge_scope(keep_arcpl, require_liq, require_opx):
    """Build a scope DataFrame from the raw cpx/liq/opx sheets loaded in cell 23."""
    how = 'inner' if (require_liq or require_opx) else 'left'
    s = cpx.merge(liq, on='Experiment', how=how if require_liq else 'left',
                  suffixes=('', '_liq_dup'))
    s = s.merge(opx, on='Experiment', how=how if require_opx else 'left',
                suffixes=('', '_opx_dup'))
    for tgt in ['T_K', 'P_kbar']:
        for src in ['', '_liq_dup', '_opx_dup']:
            col = f'{tgt}{src}'
            if col in s.columns:
                if tgt not in s.columns:
                    s[tgt] = s[col]
                else:
                    s[tgt] = s[tgt].combine_first(s[col])
    s['T_C'] = pd.to_numeric(s['T_K'], errors='coerce') - 273.15
    s['P_kbar'] = pd.to_numeric(s['P_kbar'], errors='coerce')
    if 'Fe3Fet_Liq' not in s.columns:
        s['Fe3Fet_Liq'] = 0.0
    if keep_arcpl:
        s = s[s['Experiment'].astype(str).isin(keep_exp)]
    has_c = s[AGREDA_CPX_COLS].apply(pd.to_numeric, errors='coerce').fillna(0.0).sum(axis=1) > 80
    s = s[has_c & np.isfinite(s['T_C']) & np.isfinite(s['P_kbar'])].reset_index(drop=True)
    return s


def _predict_all_methods(scope_df):
    y_T_s = scope_df['T_C'].values
    y_P_s = scope_df['P_kbar'].values
    p = {}
    try:
        a = predict_agreda_from_df(scope_df, MODELS / 'external', 'cpx_only', 'T')['median']
        b = predict_agreda_from_df(scope_df, MODELS / 'external', 'cpx_only', 'P')['median']
        p['Agreda-Lopez cpx-only'] = (a, b)
    except Exception as e:
        print(f'  Agreda cpx-only skipped ({e})')
    try:
        a = predict_agreda_from_df(scope_df, MODELS / 'external', 'cpx_liq', 'T')['median']
        b = predict_agreda_from_df(scope_df, MODELS / 'external', 'cpx_liq', 'P')['median']
        p['Agreda-Lopez cpx-liq'] = (a, b)
    except Exception as e:
        print(f'  Agreda cpx-liq skipped ({e})')
    try:
        p['Jorgenson cpx-only'] = (
            predict_jorgenson(scope_df, 'T', phase='cpx_only', P_kbar=y_P_s),
            predict_jorgenson(scope_df, 'P', phase='cpx_only', T_K=y_T_s + 273.15),
        )
    except Exception as e:
        print(f'  Jorgenson skipped ({e})')
    try:
        p['Wang 2021 cpx-liq'] = (
            predict_wang(scope_df, 'T', P_kbar=y_P_s),
            predict_wang(scope_df, 'P', T_K=y_T_s + 273.15),
        )
    except Exception as e:
        print(f'  Wang skipped ({e})')
    try:
        p['Putirka 2008 cpx-liq'] = (
            predict_putirka_cpx_liq(scope_df, 'T', P_kbar=y_P_s),
            predict_putirka_cpx_liq(scope_df, 'P', T_K=y_T_s + 273.15),
        )
    except Exception as e:
        print(f'  Putirka skipped ({e})')
    for fam in ['forest', 'boosted']:
        try:
            sT = canonical_model_spec('T_C', 'opx_liq', fam, RESULTS)
            sP = canonical_model_spec('P_kbar', 'opx_liq', fam, RESULTS)
            mT = joblib.load(canonical_model_path('T_C', 'opx_liq', fam, MODELS, RESULTS))
            mP = joblib.load(canonical_model_path('P_kbar', 'opx_liq', fam, MODELS, RESULTS))
            Xt, _ = build_feature_matrix(scope_df, sT['feature_set'], use_liq=True)
            Xp, _ = build_feature_matrix(scope_df, sP['feature_set'], use_liq=True)
            p[f'Ours opx-liq {fam}'] = (mT.predict(Xt), mP.predict(Xp))
        except Exception as e:
            print(f'  Ours opx-liq {fam} partial/skipped ({e})')
            p[f'Ours opx-liq {fam}'] = (np.full(len(scope_df), np.nan),
                                         np.full(len(scope_df), np.nan))
    return p


def _metrics_df(preds_s, y_T, y_P, n_total):
    rows = []
    for name, (pT, pP) in preds_s.items():
        mT = rmse_ci(y_T, pT, B=BOOT_NEW)
        mP = rmse_ci(y_P, pP, B=BOOT_NEW)
        rows.append({
            'Method': name,
            'T_n': mT[3], 'T_RMSE': mT[0],
            'T_RMSE_CI_lo': mT[1], 'T_RMSE_CI_hi': mT[2], 'T_R2': mT[4],
            'T_coverage_pct': 100 * mT[3] / n_total if n_total else 0,
            'P_n': mP[3], 'P_RMSE': mP[0],
            'P_RMSE_CI_lo': mP[1], 'P_RMSE_CI_hi': mP[2], 'P_R2': mP[4],
            'P_coverage_pct': 100 * mP[3] / n_total if n_total else 0,
        })
    return pd.DataFrame(rows)


def _is_putirka(name):
    return name.startswith('Putirka')


def _render_three_panel(df_m, n_total, scope_label, out_stem):
    min_cov = df_m[['T_coverage_pct', 'P_coverage_pct']].min(axis=1)
    dropped = df_m.loc[min_cov < 20, 'Method'].tolist()
    shown_mask = min_cov >= 20
    shown = df_m.loc[shown_mask].copy()
    shown['_min_cov'] = min_cov.loc[shown_mask].values
    shown['_max_cov'] = df_m.loc[shown_mask, ['T_coverage_pct',
                                              'P_coverage_pct']].max(axis=1).values
    if dropped:
        print(f'  Dropped from figure (<20% cov): {dropped}')
    shown = shown.sort_values('T_RMSE', na_position='last').reset_index(drop=True)

    footnotes = {}
    foot_text = []
    for i, row in shown.iterrows():
        if row['_min_cov'] < 90:
            num = len(footnotes) + 1
            footnotes[row['Method']] = str(num)
            foot_text.append(
                f"{num} {row['Method']}: {row['_min_cov']:.0f}% coverage "
                f"(limited by {'T' if row['T_coverage_pct'] < row['P_coverage_pct'] else 'P'} "
                f"missing-row filter)"
            )

    n_methods = len(shown)
    h = max(8.0, 0.4 * n_methods + 3)
    fig = plt.figure(figsize=(16, h))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[4, 4, 1.5])
    ax_T = fig.add_subplot(gs[0, 0])
    ax_P = fig.add_subplot(gs[0, 1], sharey=ax_T)
    ax_C = fig.add_subplot(gs[0, 2], sharey=ax_T)
    y_pos = np.arange(n_methods)

    for ax, col, col_lo, col_hi, xlab, fmt in [
        (ax_T, 'T_RMSE', 'T_RMSE_CI_lo', 'T_RMSE_CI_hi', 'T RMSE (C)', '{:.0f}'),
        (ax_P, 'P_RMSE', 'P_RMSE_CI_lo', 'P_RMSE_CI_hi', 'P RMSE (kbar)', '{:.2f}'),
    ]:
        vals = shown[col].values.astype(float)
        lo = shown[col_lo].values.astype(float)
        hi = shown[col_hi].values.astype(float)
        xerr_lo = np.where(np.isfinite(vals) & np.isfinite(lo), vals - lo, 0.0)
        xerr_hi = np.where(np.isfinite(vals) & np.isfinite(hi), hi - vals, 0.0)
        xerr = np.vstack([xerr_lo, xerr_hi])
        colors = [COLORS.get(n, '#777') for n in shown['Method']]
        bars = ax.barh(y_pos, np.nan_to_num(vals, nan=0.0), xerr=xerr,
                       color=colors, edgecolor='black',
                       error_kw={'elinewidth': 1.1, 'capsize': 2.5})
        for b, n in zip(bars, shown['Method']):
            if _is_putirka(n):
                b.set_hatch('///')
            if not np.isfinite(shown.loc[b.get_y() // 1, col]) if False else False:
                pass
        for i, v in enumerate(vals):
            if np.isfinite(v):
                ax.text(v * 1.01 + 0.001, i, fmt.format(v),
                        va='center', fontsize=8)
            else:
                ax.text(0.01, i, 'n/a', va='center', fontsize=8,
                        color='gray', style='italic')
        if np.isfinite(vals).any():
            best = float(np.nanmin(vals))
            ax.axvline(best, color='gray', linestyle=':', lw=1, alpha=0.6)
        ax.set_xlabel(xlab)
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(left=0)

    ax_T.set_yticks(y_pos)
    labels = []
    for name in shown['Method']:
        sup = footnotes.get(name)
        labels.append(f'{name}$^{{{sup}}}$' if sup else name)
    ax_T.set_yticklabels(labels, fontsize=9)
    ax_T.invert_yaxis()
    plt.setp(ax_P.get_yticklabels(), visible=False)
    plt.setp(ax_C.get_yticklabels(), visible=False)

    cov_bar = shown['_min_cov'].values
    cov_mark = shown['_max_cov'].values
    c_colors = [COLORS.get(n, '#777') for n in shown['Method']]
    ax_C.barh(y_pos, cov_bar, color=c_colors, edgecolor='black', alpha=0.6)
    for i, (mn, mx) in enumerate(zip(cov_bar, cov_mark)):
        ax_C.text(mn + 1.5, i, f'{mn:.0f}%', va='center', fontsize=8)
        if abs(mx - mn) > 0.5:
            ax_C.plot(mx, i, marker='x', color='black', ms=5, alpha=0.8)
    ax_C.axvline(100, color='gray', linestyle='--', lw=1, alpha=0.6)
    ax_C.set_xlim(0, 105)
    ax_C.set_xlabel('Coverage (%)')
    ax_C.grid(axis='x', alpha=0.3)

    fig.suptitle(f'Method benchmark — {scope_label} (n_total = {n_total})',
                 fontsize=12, fontweight='bold')

    caption = ('Bar height (panels A, B): RMSE with 95% bootstrap CI (B=2000). '
               'Bar height (panel C): coverage = min(T, P) finite-prediction '
               'rate; x marker = higher of T/P if differs. Hatched bars = '
               'classical (Putirka 2008); solid bars = ML methods. Methods '
               'sorted by T RMSE ascending.')
    fig.text(0.5, -0.02, caption, ha='center', fontsize=8, wrap=True)
    if foot_text:
        fig.text(0.05, -0.06, '    '.join(foot_text), ha='left',
                 fontsize=7, style='italic')

    for ext in ['png', 'pdf']:
        fp = FIGURES / f'{out_stem}.{ext}'
        fig.savefig(fp, bbox_inches='tight',
                    dpi=300 if ext == 'png' else None)
    plt.show()
    plt.close(fig)

    cov_lo, cov_hi = (cov_bar.min(), cov_bar.max()) if len(cov_bar) else (0, 0)
    low_cov_names = [n for n, c in zip(shown['Method'], cov_bar) if c < 90]
    print(f"Saved {out_stem}: panels [A, B, C] with n_methods={n_methods} rows. "
          f"Coverage range: {cov_lo:.0f}% - {cov_hi:.0f}%. "
          f"Methods with cov<90%: {low_cov_names}")


# ---- Build three scopes ----
m_arcpl_paired = _merge_scope(keep_arcpl=True, require_liq=True, require_opx=True)
m_lepr_full = _merge_scope(keep_arcpl=False, require_liq=True, require_opx=True)
m_cpx_scope = _merge_scope(keep_arcpl=True, require_liq=False, require_opx=False)
print(f'Scope arcpl_paired: n={len(m_arcpl_paired)}')
print(f'Scope lepr_full:    n={len(m_lepr_full)}  (includes training rows for Ours)')
print(f'Scope cpx_scope:    n={len(m_cpx_scope)}')

SCOPES = [
    (m_arcpl_paired, 'ArcPL three-phase paired (cpx + opx + liq)',
     'fig_nb04_method_benchmark_paired',   'arcpl_paired'),
    (m_lepr_full,    'Full LEPR three-phase paired',
     'fig_nb04_method_benchmark_lepr_full', 'lepr_full'),
    (m_cpx_scope,    'ArcPL cpx-bearing scope',
     'fig_nb04_method_benchmark_cpx_scope', 'cpx_scope'),
]
for df_scope, label, fn, slug in SCOPES:
    if len(df_scope) < 10:
        print(f'\n[{slug}] SKIPPED: n={len(df_scope)} too small')
        continue
    print(f'\n[{slug}] running {len(df_scope)} rows ...')
    preds_s = _predict_all_methods(df_scope)
    ms = _metrics_df(preds_s, df_scope['T_C'].values,
                     df_scope['P_kbar'].values, len(df_scope))
    ms.round(3).to_csv(RESULTS / f'nb04_method_benchmark_{slug}.csv', index=False)
    _render_three_panel(ms, len(df_scope), label, fn)
'''


def main() -> int:
    nb = nbformat.read(str(NB), as_version=4)
    cell = nb.cells[26]
    if MARKER in cell.source:
        print("nb04 cell 26: already patched.")
        return 0
    cell.source = NEW_CELL26
    nbformat.write(nb, str(NB))
    print("nb04 cell 26: replaced with three-panel method benchmark (3 scopes).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
