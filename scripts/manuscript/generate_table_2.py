"""Build tables/table_2_per_cell_winners.csv from results/bootstrap_rmse_cis_all_cells.csv.

Locks the four opx per-cell winners and pulls bootstrap CIs for each.
"""
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

WINNERS = [
    ('opx_liq',  'T_C',    'ElasticNet', 'raw'),
    ('opx_liq',  'P_kbar', 'MLP',        'raw'),
    ('opx_only', 'T_C',    'LightGBM',   'alr'),
    ('opx_only', 'P_kbar', 'RF',         'pwlr'),
]


def main():
    df = pd.read_csv(PROJECT_ROOT / 'results' / 'bootstrap_rmse_cis_all_cells.csv')
    rows = []
    for track, target, model, fs in WINNERS:
        sel = df[(df.track == track) & (df.target == target)
                 & (df['model'] == model) & (df.feature_set == fs)]
        if sel.empty:
            raise RuntimeError(f'No row for {track}/{target}/{model}/{fs}')
        r = sel.iloc[0]
        target_label = 'T (deg C)' if target == 'T_C' else 'P (kbar)'
        rows.append({
            'Track':        track.replace('_', '-'),
            'Target':       target_label,
            'Best model':   model,
            'Feature set':  fs,
            'RMSE (point)': f'{r.rmse_point:.2f}',
            '95% CI':       f'[{r.rmse_ci_lo:.2f}, {r.rmse_ci_hi:.2f}]',
            'n_test':       int(r.n_test),
        })
    out = pd.DataFrame(rows)
    out_path = PROJECT_ROOT / 'tables' / 'table_2_per_cell_winners.csv'
    out.to_csv(out_path, index=False)
    print(f'wrote {len(out)} rows to {out_path.relative_to(PROJECT_ROOT)}')


if __name__ == '__main__':
    main()
