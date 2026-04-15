"""NB02 v6 overhaul: add figure explanations + petrologic cluster-name labels.

- Adds per-figure markdown interpretation cells after fig01, fig02,
  fig_nb02_clusters.
- Inserts a cluster-labeling code cell that assigns a petrologic name to
  each k-means cluster from its centroid chemistry + P/T.
- Updates the final save cell (cell-011-*) to include the cluster_name column.

Idempotent: cells tagged 'v6-nb02-' are removed before re-insertion.
"""
from __future__ import annotations

from pathlib import Path
import nbformat
from nbformat.v4 import new_markdown_cell, new_code_cell

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / 'notebooks' / 'nb02_eda_pca.ipynb'
TAG = 'v6-nb02-'

# ---------- new markdown explanation cells ---------------------------------
FIG01_MD = """**Figure interpretation (fig01 P-T distribution):** The training set spans
roughly 700-1600 C and 0-30 kbar, with density concentrated along the hot,
shallow corner (high-T, low-P) where most arc and MOR basalt experiments
sit. The tails (cold + deep) are sparsely populated, so test-time
performance in those regions rests on relatively few training examples -
expect wider conformal prediction intervals there. The marginal histograms
give the univariate distribution that enters the multi-seed split in NB03."""

FIG02_MD = """**Figure interpretation (fig02 PCA biplot):** PC1 captures the MgO-FeO vs
SiO2-alkali (fractionation) axis and PC2 captures an Al-Cr + Cr/Mg variance
axis. Colouring by T and P shows that P gradients are mostly perpendicular
to PC1 (i.e. PCA learned from oxides alone picks up the T signal
immediately but only partially resolves P), which is why our P RMSE stays
stubbornly above T RMSE in every model family tested in NB03."""

CLUSTERS_MD = """**Figure interpretation (fig_nb02_clusters):** The k-means clusters
separate the training opx into chemically distinct populations. The
centroid-chemistry + P-T table below assigns each cluster a petrologic
label (primitive basaltic, decompression residue, evolved/depleted, etc.).
These labels are used downstream in NB05 for LOSO stress testing - holding
one cluster out and retraining tests how well the model generalizes across
magmatic regimes, not just across random splits."""


CLUSTER_LABELER = """# Phase 2R.5 (v6): assign petrologic names to the k-means clusters.
# Rules are applied to the cluster centroid chemistry + mean P/T. Note
# Mg_num is stored on a 0-100 (mol%) scale, so thresholds are scaled
# accordingly. The canonical cleaned table is rewritten so NB03/NB05/NB08
# pick up the `cluster_name` column without refitting.
def _label_cluster(row):
    T = row['T_mean']; P = row['P_mean']
    Si = row['SiO2_mean']; Mg = row['Mg_num_mean']
    # Low-Mg, shallow, cooler -> differentiated
    if Mg < 75 or T < 1100:
        return 'evolved/depleted'
    # High-Mg and deep split by SiO2: low-Si = basaltic melt residue,
    # high-Si = decompression-melted harzburgitic residue.
    if P >= 12 and Si < 55:
        return 'primitive basaltic'
    if P >= 12 and Si >= 55:
        return 'decompression residue'
    return 'intermediate'

_name_map = summary.apply(_label_cluster, axis=1).to_dict()
df['cluster_name'] = df['chemical_cluster'].map(_name_map)

print('Cluster labels assigned:')
_label_tbl = (df.groupby(['chemical_cluster', 'cluster_name'])
                .size().reset_index(name='n')
                .sort_values('chemical_cluster'))
print(_label_tbl.to_string(index=False))
"""

SAVE_REPLACEMENT = """# Save with cluster assignments + petrologic labels so the cleaning output
# from NB01 (`opx_clean_core.parquet`) is not overwritten. Downstream
# notebooks read this clustered file for the `chemical_cluster` +
# `cluster_name` columns used by NB03 stratification and NB05 LOSO.
df.to_parquet(DATA_PROC / 'opx_clean_core_with_clusters.parquet')
print('Saved opx_clean_core_with_clusters.parquet with '
      'chemical_cluster + cluster_name columns')
print(f'  cluster_name unique: {sorted(df[\"cluster_name\"].dropna().unique())}')
print('\\n=== NB02 (v6) COMPLETE ===')
"""


def _find(cells, cid):
    for i, c in enumerate(cells):
        if str(c.get('id', '')) == cid:
            return i
    raise RuntimeError(f'cell id {cid} not found')


def main():
    nb = nbformat.read(str(NB_PATH), as_version=4)

    # 1) Remove any previously inserted v6 cells.
    nb.cells = [c for c in nb.cells if not str(c.get('id', '')).startswith(TAG)]

    # 2) Replace the original save cell (id cell-011-*) with the v6 version.
    save_idx = _find(nb.cells, 'cell-011-e9019960')
    nb.cells[save_idx].source = SAVE_REPLACEMENT

    # 3) Insert the cluster-labeling code cell BEFORE the (now-replaced) save
    # cell, i.e. after the cluster-summary cell (cell-010-*). Tag it so reruns
    # replace cleanly.
    summary_idx = _find(nb.cells, 'cell-010-f7fe2a19')

    def _mk(kind, suffix, src):
        c = (new_markdown_cell(src) if kind == 'markdown'
             else new_code_cell(src))
        c['id'] = TAG + suffix
        return c

    nb.cells.insert(summary_idx + 1, _mk('code', 'labeler', CLUSTER_LABELER))

    # 4) Insert figure-explanation markdown cells right after each figure cell.
    #    Order: fig01 (cell-003), fig02 (cell-005), fig_nb02_clusters (cell-009).
    # We insert newest-first to keep earlier indices valid; actually safer to
    # compute all positions up-front then insert from the highest index down.
    targets = [
        ('cell-009-e1151890', _mk('markdown', 'clusters-md', CLUSTERS_MD)),
        ('cell-005-39cec2ed', _mk('markdown', 'fig02-md',    FIG02_MD)),
        ('cell-003-a67e1cb6', _mk('markdown', 'fig01-md',    FIG01_MD)),
    ]
    # Sort descending by current index to avoid shift.
    indexed = [(_find(nb.cells, anchor), cell) for anchor, cell in targets]
    indexed.sort(key=lambda p: p[0], reverse=True)
    for idx, cell in indexed:
        nb.cells.insert(idx + 1, cell)

    nbformat.write(nb, str(NB_PATH))
    print(f'NB02 v6 overhaul: 1 labeler code + 3 figure-explanation markdown cells '
          f'inserted; save cell replaced. Total cells: {len(nb.cells)}')


if __name__ == '__main__':
    main()
