"""
Notebook 02: Exploratory Data Analysis & PCA
Opx ML Thermobarometer
Author: [Your name]
Date: 2026-04-04

Input:  opx_clean_core.csv
Output: fig01_pt_distribution.png
        fig02_pca_biplot.png
        fig_eda_correlation.png
        fig_eda_distributions.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

OUTDIR = Path('.')
df = pd.read_csv('opx_clean_core.csv')
print(f"Loaded core dataset: {len(df)} rows\n")

# ============================================================
# 1. SUMMARY STATISTICS
# ============================================================
print("=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)

print("\nTargets:")
for col in ['T_C', 'P_kbar']:
    s = df[col]
    print(f"  {col:10s}: n={s.notna().sum()}, mean={s.mean():.1f}, std={s.std():.1f}, "
          f"min={s.min():.1f}, Q1={s.quantile(0.25):.1f}, med={s.median():.1f}, "
          f"Q3={s.quantile(0.75):.1f}, max={s.max():.1f}")

print("\nRaw oxides (wt%):")
oxide_cols = ['SiO2', 'Al2O3', 'FeO_total', 'MgO', 'CaO', 'TiO2', 'Cr2O3', 'MnO', 'Na2O']
for col in oxide_cols:
    s = df[col]
    nn = s.notna().sum()
    if nn > 0:
        print(f"  {col:10s}: n={nn:4d} ({100*nn/len(df):5.1f}%), mean={s.mean():7.2f}, "
              f"std={s.std():6.2f}, range=[{s.min():.2f}, {s.max():.2f}]")

print("\nEngineered features:")
eng_cols = ['Mg_num', 'Al_IV', 'Al_VI', 'En_frac', 'Fs_frac', 'Wo_frac']
for col in eng_cols:
    s = df[col]
    print(f"  {col:10s}: mean={s.mean():.4f}, std={s.std():.4f}, range=[{s.min():.4f}, {s.max():.4f}]")

# ============================================================
# 2. CORRELATIONS WITH TARGETS
# ============================================================
print("\n" + "=" * 60)
print("CORRELATIONS WITH TARGETS")
print("=" * 60)

feature_cols = ['SiO2', 'Al2O3', 'FeO_total', 'MgO', 'CaO', 'Mg_num', 'Al_IV', 'Al_VI',
                'En_frac', 'Fs_frac', 'Wo_frac', 'cat_Si', 'cat_Al', 'cat_Fe', 'cat_Mg', 'cat_Ca']

print("\nCorrelation with T_C:")
corr_T = df[feature_cols + ['T_C']].corr()['T_C'].drop('T_C').sort_values(key=abs, ascending=False)
for feat, val in corr_T.items():
    print(f"  {feat:12s}: {val:+.3f}")

print("\nCorrelation with P_kbar:")
corr_P = df[feature_cols + ['P_kbar']].corr()['P_kbar'].drop('P_kbar').sort_values(key=abs, ascending=False)
for feat, val in corr_P.items():
    print(f"  {feat:12s}: {val:+.3f}")

# ============================================================
# 3. PUBLICATION COVERAGE
# ============================================================
print("\n" + "=" * 60)
print("PUBLICATION COVERAGE")
print("=" * 60)

pub_counts = df['Citation'].value_counts()
print(f"Total publications: {len(pub_counts)}")
print(f"Rows from top 10 pubs: {pub_counts.head(10).sum()} ({100*pub_counts.head(10).sum()/len(df):.1f}%)")
print(f"Pubs with >=20 rows: {(pub_counts >= 20).sum()}")
print(f"Pubs with 10-19 rows: {((pub_counts >= 10) & (pub_counts < 20)).sum()}")
print(f"Pubs with 5-9 rows: {((pub_counts >= 5) & (pub_counts < 10)).sum()}")
print(f"Pubs with 1-4 rows: {(pub_counts < 5).sum()}")
print(f"Single-row pubs: {(pub_counts == 1).sum()}")

# ============================================================
# FIGURE 1: P-T DISTRIBUTION WITH MARGINAL HISTOGRAMS
# ============================================================
fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                       hspace=0.05, wspace=0.05)

ax_main = fig.add_subplot(gs[1, 0])
ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

sc = ax_main.scatter(df['T_C'], df['P_kbar'], c=df['Mg_num'], cmap='viridis',
                     s=12, alpha=0.6, edgecolors='none')
ax_main.set_xlabel('Temperature (°C)', fontsize=12)
ax_main.set_ylabel('Pressure (kbar)', fontsize=12)
ax_main.invert_yaxis()
plt.colorbar(sc, ax=ax_main, pad=0.15, shrink=0.6, label='Mg#')

ax_top.hist(df['T_C'], bins=40, color='#2c7bb6', edgecolor='white', linewidth=0.3)
ax_top.set_ylabel('Count')
plt.setp(ax_top.get_xticklabels(), visible=False)

ax_right.hist(df['P_kbar'], bins=40, orientation='horizontal', color='#d7191c', edgecolor='white', linewidth=0.3)
ax_right.set_xlabel('Count')
ax_right.invert_yaxis()
plt.setp(ax_right.get_yticklabels(), visible=False)

ax_top.set_title(f'ExPetDB Orthopyroxene: {len(df)} experiments, {df["Citation"].nunique()} studies',
                 fontsize=13, pad=10)
fig.savefig(OUTDIR / 'fig01_pt_distribution.png', dpi=200, bbox_inches='tight')
plt.close()
print("\nSaved fig01_pt_distribution.png")

# ============================================================
# FIGURE 2: PCA BIPLOT
# ============================================================
pca_features = ['SiO2', 'Al2O3', 'FeO_total', 'MgO', 'CaO']
X_pca = df[pca_features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)

pca = PCA()
scores = pca.fit_transform(X_scaled)

print("\n" + "=" * 60)
print("PCA RESULTS (5 core oxides, standardized)")
print("=" * 60)
cum_var = np.cumsum(pca.explained_variance_ratio_)
for i, (ev, cv) in enumerate(zip(pca.explained_variance_ratio_, cum_var)):
    print(f"  PC{i+1}: {100*ev:.1f}% variance (cumulative: {100*cv:.1f}%)")

print(f"\nPCs needed for 90% variance: {np.argmax(cum_var >= 0.90) + 1}")
print(f"PCs needed for 95% variance: {np.argmax(cum_var >= 0.95) + 1}")

print("\nLoadings:")
loadings = pca.components_
for i in range(min(3, len(loadings))):
    print(f"  PC{i+1}: ", end='')
    for j, feat in enumerate(pca_features):
        print(f"{feat}={loadings[i,j]:+.3f}  ", end='')
    print()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sc1 = axes[0].scatter(scores[:, 0], scores[:, 1], c=df['T_C'], cmap='RdYlBu_r',
                      s=12, alpha=0.6, edgecolors='none')
plt.colorbar(sc1, ax=axes[0], label='Temperature (°C)', shrink=0.8)
axes[0].set_xlabel(f'PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)', fontsize=11)
axes[0].set_ylabel(f'PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)', fontsize=11)
axes[0].set_title('(a) Colored by Temperature', fontsize=12)

sc2 = axes[1].scatter(scores[:, 0], scores[:, 1], c=df['P_kbar'], cmap='viridis',
                      s=12, alpha=0.6, edgecolors='none')
plt.colorbar(sc2, ax=axes[1], label='Pressure (kbar)', shrink=0.8)
axes[1].set_xlabel(f'PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)', fontsize=11)
axes[1].set_ylabel(f'PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)', fontsize=11)
axes[1].set_title('(b) Colored by Pressure', fontsize=12)

for ax in axes:
    for j, feat in enumerate(pca_features):
        scale = 3.0
        ax.annotate('', xy=(loadings[0,j]*scale, loadings[1,j]*scale), xytext=(0, 0),
                     arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        ax.text(loadings[0,j]*scale*1.15, loadings[1,j]*scale*1.15, feat,
                fontsize=9, ha='center', va='center', fontweight='bold')

plt.tight_layout()
fig.savefig(OUTDIR / 'fig02_pca_biplot.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig02_pca_biplot.png")

# ============================================================
# FIGURE: CORRELATION HEATMAP
# ============================================================
corr_cols = ['SiO2', 'Al2O3', 'FeO_total', 'MgO', 'CaO', 'Mg_num', 'Al_IV', 'Al_VI', 'T_C', 'P_kbar']
corr_matrix = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(len(corr_cols)))
ax.set_yticks(range(len(corr_cols)))
ax.set_xticklabels(corr_cols, rotation=45, ha='right', fontsize=10)
ax.set_yticklabels(corr_cols, fontsize=10)

for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        val = corr_matrix.iloc[i, j]
        color = 'white' if abs(val) > 0.5 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color)

plt.colorbar(im, ax=ax, label='Pearson r', shrink=0.8)
ax.set_title('Feature-Target Correlation Matrix', fontsize=13)
plt.tight_layout()
fig.savefig(OUTDIR / 'fig_eda_correlation.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig_eda_correlation.png")

# ============================================================
# FIGURE: OXIDE DISTRIBUTIONS
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
plot_cols = ['SiO2', 'Al2O3', 'FeO_total', 'MgO', 'CaO', 'Mg_num']
colors = ['#2c7bb6', '#d7191c', '#fdae61', '#1a9641', '#762a83', '#313695']

for i, (col, color) in enumerate(zip(plot_cols, colors)):
    ax = axes[i]
    ax.hist(df[col].dropna(), bins=40, color=color, edgecolor='white', linewidth=0.3, alpha=0.8)
    ax.set_xlabel(col + (' (wt%)' if col != 'Mg_num' else ''), fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title(f'n={df[col].notna().sum()}, μ={df[col].mean():.2f}, σ={df[col].std():.2f}', fontsize=9)

plt.suptitle('Core Feature Distributions', fontsize=13, y=1.01)
plt.tight_layout()
fig.savefig(OUTDIR / 'fig_eda_distributions.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig_eda_distributions.png")
print("\nNotebook 02 complete.")
