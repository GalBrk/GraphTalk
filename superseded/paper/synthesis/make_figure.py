"""Redraw density_diagnostics.pdf for synthesis.tex from the audited CSVs.

Same data and styling as ../compare/independent/make_assets.py; only the
legend placement differs, so no legend covers a plotted point.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent.parent / 'compare' / 'independent' / 'results'
OUT = Path(__file__).resolve().parent / 'density_diagnostics.pdf'

plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'font.family': 'DejaVu Sans'})
fig, ax = plt.subplots(1, 3, figsize=(7.05, 2.0), layout='constrained')

d = pd.read_csv(R / 'by_density.csv')
for cond, label, color in [('none', 'No primer', '#3f4c65'), ('degree', 'Degree primer', '#c35521')]:
    x = d[(d.arm == 'qwen3-4b') & (d.task == 'node_degree') & (d.condition == cond)]
    ax[0].plot(x.p, 100 * x.success, 'o-', label=label, color=color, ms=4)
ax[0].set(ylabel='Success (%)', title='(a) 4B: degree lookup', ylim=(0, 105))

x = pd.read_csv(R / 'edge_balance.csv').query('arm == "qwen3-1.7b" and condition == "none"')
ax[1].plot(x.p, 100 * x.accuracy, 'o-', label='Raw success', color='#3f4c65', ms=4)
ax[1].plot(x.p, 100 * x.balanced, 's-', label='Balanced success', color='#c35521', ms=4)
ax[1].plot(x.p, 100 * np.maximum(x.prevalence, 1 - x.prevalence), ':', label='Majority label', color='#65845a')
# headroom above the data holds the legend; ticks stop at 100
ax[1].set(title='(b) 1.7B: edge existence', ylim=(40, 128), yticks=range(40, 101, 10))

x = pd.read_csv(R / 'topology.csv').groupby('p').mean(numeric_only=True).reset_index()
for col, label, style, color in [('rwse_unique', '2 decimals', 'o-', '#c35521'),
                                 ('rwse_unique_4dp', '4 decimals', 's--', '#65845a'),
                                 ('rwse_unique_6dp', '6 decimals', '^-', '#3f4c65')]:
    ax[2].plot(x.p, x[col], style, label=label, color=color, ms=4)
ax[2].set(title='(c) RWSE feature diversity', ylabel='Unique pairs / 40 nodes', ylim=(0, 43))

for a, loc in zip(ax, ['lower left', 'upper center', 'center right']):
    a.set_xlabel('Edge probability p'); a.set_xticks([.1, .35, .5, .65, .85]); a.tick_params(labelsize=8)
    a.axvspan(.575, .9, color='#e9eef2', alpha=.6, zorder=-1); a.set_xlim(.075, .875)
    a.legend(fontsize=7, loc=loc, frameon=False, ncol=1)
fig.savefig(OUT)
fig.savefig(OUT.with_suffix('.png'), dpi=200)
