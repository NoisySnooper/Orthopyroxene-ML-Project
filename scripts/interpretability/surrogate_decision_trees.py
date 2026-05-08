#!/usr/bin/env python3
"""Phase 2 P2.6: surrogate decision trees.

For each shipped opx cell, fit DecisionTreeRegressor(max_depth=4) using
the winner's ML predictions as the target. High surrogate R^2 means the
ML's decision structure can be approximated as a readable flowchart.

Output:
  results/surrogate_tree_r_squared.csv
  figures/core/Core_18_fig_surrogate_trees.pdf (+ .png, 4 pages)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor, plot_tree

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from src.prepare_train_test import prepare_train_test  # noqa: E402
from scripts.figures._style import apply_pub_style, resolve_out_dir, jgr_figsize # noqa: E402
from scripts.figures._labels import TARGET_UNIT, panel_header  # noqa: E402

apply_pub_style()

MODELS_DIR = PROJECT_ROOT / 'models' / 'canonical'
BOOT_CSV = PROJECT_ROOT / 'results' / 'bootstrap_rmse_cis_all_cells.csv'
OUT_CSV = PROJECT_ROOT / 'results' / 'surrogate_tree_r_squared.csv'
OUT_DIR = resolve_out_dir(PROJECT_ROOT)
FIG_STEM = OUT_DIR / 'supp_fig_8'


CELLS = [
    ('opx', 'opx_liq', 'T_C'),
    ('opx', 'opx_liq', 'P_kbar'),
    ('opx', 'opx_only', 'T_C'),
    ('opx', 'opx_only', 'P_kbar'),
]


def canonical_path(pipeline, model, target, track, fs):
    return (MODELS_DIR / pipeline
            / f'base_{model}_{target}_{track}_{fs}.joblib')


def bootstrap_r2(y, yhat, n_boot=500, random_state=42):
    rng = np.random.default_rng(random_state)
    n = len(y)
    rs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        rs[b] = r2_score(y[idx], yhat[idx])
    lo, hi = np.percentile(rs, [2.5, 97.5])
    return float(lo), float(hi)


def interpret(r2):
    if r2 >= 0.85:
        return ('ML prediction well-approximated by a depth-4 decision tree '
                f'(R^2={r2:.2f}); the ML can be read as a flowchart')
    if r2 >= 0.70:
        return ('surrogate tree approximates ML prediction with moderate '
                f'fidelity (R^2={r2:.2f}); tree remains a useful approximate '
                'explanation but smooth/interaction effects are lost')
    return ('surrogate tree fails to approximate ML prediction '
            f'(R^2={r2:.2f}); ML relies on smooth non-tree structure and '
            'this probe is not informative for this cell')


def main():
    boot = pd.read_csv(BOOT_CSV)
    rows = []
    pdf = PdfPages(f'{FIG_STEM}.pdf')
    png_first_fig = None

    for pipe, track, tgt in CELLS:
        winner = (boot[(boot.pipeline == pipe) & (boot.track == track)
                       & (boot.target == tgt)]
                  .sort_values('rmse_point').iloc[0])
        family = winner['model']
        fs = winner['feature_set']
        print(f'\n=== {track}/{tgt} winner {family}/{fs} ===')

        if family == 'TabPFN':
            tab = pd.read_csv('results/tabpfn_predictions.csv')
            m = tab[(tab.track == track) & (tab.target == tgt)
                    & (tab.seed == 42)]
            y_ml = m['y_pred'].to_numpy(dtype=float)
        else:
            mp = canonical_path(pipe, family, tgt, track, fs)
            est = joblib.load(mp)
            dw = prepare_train_test(pipe, track, tgt, fs)
            y_ml = np.asarray(est.predict(dw['X_te']), dtype=float)

        d = prepare_train_test(pipe, track, tgt, 'raw')
        X = d['X_te']
        feat_names = list(d['feat_names'])

        tree = DecisionTreeRegressor(max_depth=4, min_samples_leaf=10,
                                     random_state=42)
        tree.fit(X, y_ml)
        yhat = tree.predict(X)
        r2 = float(r2_score(y_ml, yhat))
        lo, hi = bootstrap_r2(y_ml, yhat, n_boot=500, random_state=42)
        n_leaves = int(tree.get_n_leaves())
        print(f'  R^2={r2:.3f} [{lo:.3f}, {hi:.3f}]  leaves={n_leaves}')

        rows.append({
            'pipeline':              pipe,
            'track':                 track,
            'target':                tgt,
            'winner_family':         family,
            'winner_feature_set':    fs,
            'surrogate_r_squared':   r2,
            'surrogate_r_squared_ci_lo': lo,
            'surrogate_r_squared_ci_hi': hi,
            'n_leaves':              n_leaves,
            'interpretation':        interpret(r2),
        })

        fig, ax = plt.subplots(figsize=jgr_figsize((11, 6)))
        plot_tree(tree, feature_names=feat_names, filled=True,
                  rounded=True, ax=ax, fontsize=6)
        unit = TARGET_UNIT[tgt]
        ax.set_title(f'{panel_header(track, tgt)}  '
                     f'winner {family}/{fs}  '
                     f'R²={r2:.2f} [{lo:.2f}, {hi:.2f}]  ({unit})\n'
                     f'ExPetDB held-out (n=174 / n=190), seed 42',
                     fontsize=10, loc='center')
        pdf.savefig(fig, bbox_inches='tight')
        # First page is the .png representative for docx + web preview.
        if png_first_fig is None:
            png_first_fig = fig
        else:
            plt.close(fig)

    pdf.close()
    if png_first_fig is not None:
        png_first_fig.savefig(f'{FIG_STEM}.png', bbox_inches='tight', dpi=300)
        plt.close(png_first_fig)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    print(f'\nwrote {OUT_CSV}')
    print(out[['track', 'target', 'surrogate_r_squared',
               'surrogate_r_squared_ci_lo',
               'surrogate_r_squared_ci_hi', 'n_leaves']].to_string(index=False))

    caption = (
        'Supp. Figure 8. Surrogate decision trees for the four opx '
        '(track, target) cells. A DecisionTreeRegressor(max_depth=4, '
        'min_samples_leaf=10, seed 42) is fit to each per-cell winner\'s '
        'ML predictions on raw features. Each page shows one cell; '
        'reported R² is the surrogate fidelity with 500-bootstrap 95% CI. '
        'A high R² (≥0.85) means the ML is well-approximated by a '
        'depth-4 flowchart; a low R² means the ML uses smooth/'
        'interaction structure the tree cannot recover. Source: '
        'results/surrogate_tree_r_squared.csv.'
    )
    (OUT_DIR / 'supp_fig_8.txt').write_text(caption, encoding='utf-8')
    print(f'wrote {FIG_STEM}.pdf and supp_fig_8.txt')


if __name__ == '__main__':
    main()
