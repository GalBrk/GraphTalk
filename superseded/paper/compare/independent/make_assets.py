"""Generate every paper table/plot directly from the audited CSVs."""
from pathlib import Path
import pandas as pd
import numpy as np
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
R=P/'results'; O=P/'assets'; O.mkdir(exist_ok=True)
arms=['qwen3-1.7b','qwen3-1.7b-think','qwen3-4b','qwen3-4b-think']
labels=['1.7B','1.7B-T','4B','4B-T']
conds=['filler','components','clustering','rwse','degree','all']
tasks=['node_degree','connected_nodes','edge_existence','edge_count']
tasklabels=['Degree','Neighbors','Edge exists','Edge count']
e=pd.read_csv(R/'paired_effects.csv'); pool=pd.read_csv(R/'pooled.csv')
lines=[r'\begin{tabular}{llrrrrrrr}',r'\toprule',r'Arm & Task & None & Filler & Comp. & Clust. & RWSE & Degree & All \\',r'\midrule']
for arm,label in zip(arms,labels):
 for k,(task,tl) in enumerate(zip(tasks,tasklabels)):
  g=e[(e.arm==arm)&(e.task==task)&(e.control=='none')].set_index('condition')
  base=g.iloc[0].baseline
  cells=[label if k==0 else '',tl,f'{base:.2f}']
  for c in conds:
   x=g.loc[c]; cells.append(f'${x.delta:+.2f}'+(r'^{*}' if x.qvalue<.05 else '')+'$')
  lines.append(' & '.join(cells)+r' \\')
 if arm!=arms[-1]:lines.append(r'\addlinespace[2pt]')
lines +=[r'\bottomrule',r'\end{tabular}']
(O/'main_table.tex').write_text('\n'.join(lines))
c=pd.read_csv(R/'consistency.csv')
lines=[r'\begin{tabular}{lrrr}',r'\toprule',r'Arm & None & Degree & All \\',r'\midrule']
for arm,label in zip(arms,labels):
 g=c[c.arm==arm].set_index('condition')
 lines.append(' & '.join([label]+[f'{100*g.loc[k,"consistent"]:.1f}/{100*g.loc[k,"joint_success"]:.1f}' for k in ['none','degree','all']])+r' \\')
lines +=[r'\bottomrule',r'\end{tabular}']
(O/'joint_table.tex').write_text('\n'.join(lines))
lines=[r'\begin{tabular}{lrrr}',r'\toprule',r'Arm & Completion & Conditional & Success \\',r'\midrule']
for arm,label in zip(arms,labels):
 x=pool[(pool.arm==arm)&(pool.task=='edge_count')&(pool.condition=='none')].iloc[0]
 lines.append(f'{label} & {100*(1-x.cap):.2f} & {100*x.success/(1-x.cap):.2f} & {100*x.success:.2f}'+r' \\')
lines +=[r'\bottomrule',r'\end{tabular}']
(O/'completion_table.tex').write_text('\n'.join(lines))
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'font.family':'DejaVu Sans'})
fig,ax=plt.subplots(1,3,figsize=(7.05,2.45),layout='constrained')
d=pd.read_csv(R/'by_density.csv')
for cond,label,color in [('none','No primer','#3f4c65'),('degree','Degree primer','#c35521')]:
 x=d[(d.arm=='qwen3-4b')&(d.task=='node_degree')&(d.condition==cond)]
 ax[0].plot(x.p,100*x.success,'o-',label=label,color=color,ms=4)
ax[0].set(ylabel='Success (%)',title='(a) 4B: degree lookup',ylim=(0,105))
x=pd.read_csv(R/'edge_balance.csv').query('arm == "qwen3-1.7b" and condition == "none"')
ax[1].plot(x.p,100*x.accuracy,'o-',label='Raw success',color='#3f4c65',ms=4)
ax[1].plot(x.p,100*x.balanced,'s-',label='Balanced success',color='#c35521',ms=4)
ax[1].plot(x.p,100*np.maximum(x.prevalence,1-x.prevalence),':',label='Majority label',color='#65845a')
ax[1].set(title='(b) 1.7B: edge existence',ylim=(40,105))
x=pd.read_csv(R/'topology.csv').groupby('p').mean(numeric_only=True).reset_index()
for col,label,style,color in [('rwse_unique','2 decimals','o-','#c35521'),('rwse_unique_4dp','4 decimals','s--','#65845a'),('rwse_unique_6dp','6 decimals','^-','#3f4c65')]:
 ax[2].plot(x.p,x[col],style,label=label,color=color,ms=4)
ax[2].set(title='(c) RWSE feature diversity',ylabel='Unique pairs / 40 nodes',ylim=(0,43))
for a in ax:
 a.set_xlabel('Edge probability p'); a.set_xticks([.1,.35,.5,.65,.85]); a.tick_params(labelsize=8)
 a.axvspan(.575,.9,color='#e9eef2',alpha=.6,zorder=-1); a.set_xlim(.075,.875)
 a.legend(fontsize=7,loc='lower left',frameon=False)
buf=io.BytesIO(); fig.savefig(buf,format='pdf'); (O/'density_diagnostics.pdf').write_bytes(buf.getvalue())
fig.savefig(O/'density_diagnostics.png',dpi=200)
# Full intervals for every primary task contrast, for reproduction outside the paper.
e.to_csv(O/'all_paired_intervals.csv',index=False)
