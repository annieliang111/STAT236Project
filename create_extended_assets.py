from pathlib import Path
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parent
r=ROOT/'results'
diag=pd.read_csv(r/'extended_risk_diagnostics.csv')
sub=diag[diag.split=='heldout'].set_index('method')
lines=['\\begin{tabular}{lrrr}\\toprule','Estimator & Brier $\\downarrow$ & Log loss $\\downarrow$ & AUC $\\uparrow$ \\\\ \\midrule']
for key,label in [('filter','Mean-field filter'),('fitted','Fitted logistic')]:
 a=sub.loc[key]
 lines.append(f'{label} & {a.brier:.4f} & {a.logloss:.4f} & {a.auc:.3f} \\\\')
lines.append('\\bottomrule\\end{tabular}')
(r/'extended_risk_table.tex').write_text('\n'.join(lines))
sumr=pd.read_csv(r/'extended_haslemere_summary.csv')
contr=pd.read_csv(r/'extended_haslemere_contrasts.csv')
a=sumr[sumr.visibility==1].set_index('policy').attack
b=sumr[sumr.visibility==.5].set_index('policy').attack
cc=contr[contr.visibility==1].set_index('contrast')
missing=(b['opt_temporal_fitted']-a['opt_temporal_fitted'])*100
text=rf'''With complete contact information, the temporal fitted optimizer has {a['opt_temporal_fitted']*100:.2f}\% ever infected, compared with {a['opt_cumulative_fitted']*100:.2f}\% for cumulative-history fitting and {a['adaptive']*100:.2f}\% for the original adaptive heuristic. The paired temporal-minus-cumulative difference is {cc.loc['temporal fitted - opt_cumulative_fitted','diff_pp']:+.2f} percentage points (95\% MC interval [{cc.loc['temporal fitted - opt_cumulative_fitted','ci_low']:+.2f},{cc.loc['temporal fitted - opt_cumulative_fitted','ci_high']:+.2f}]), which includes zero; therefore the longer simulation still does \emph{{not}} demonstrate an advantage from recency weighting. Relative to random allocation, the temporal optimizer reduces the mean simulated infection percentage by {(a['random']-a['opt_temporal_fitted'])*100:.2f} points. With half the dyads visible, its mean infection fraction increases by {missing:.2f} points. These Monte Carlo findings are exploratory because they use only 12 epidemic draws and are conditional on the day-block resampling model.'''
(r/'extended_results_text.tex').write_text(text)
pairs=[]
for vis in [1,.5]:
 r0=contr[np.isclose(contr.visibility,vis)]
 for base in ['random','adaptive','opt_cumulative_fitted']:
  row=r0[r0.contrast=='temporal fitted - '+base].iloc[0]
  pairs.append((vis,base,row.diff_pp,row.ci_low,row.ci_high))
lines=['\\begin{tabular}{rlrr}\\toprule','Observed dyads & Comparator & Difference (pp) & 95\\% MC interval \\\\ \\midrule']
for vis,base,val,lo,hi in pairs:
 labels={'random':'Random','adaptive':'Adaptive heuristic','opt_cumulative_fitted':'Cumulative optimizer'}
 lines.append(f'{int(vis*100)}\\% & {labels[base]} & {val:+.2f} & [{lo:+.2f},{hi:+.2f}] \\\\')
lines.append('\\bottomrule\\end{tabular}')
(r/'extended_contrasts_table.tex').write_text('\n'.join(lines))
(r/'extended_contrasts_text.tex').write_text(r'''\begin{table}[htbp]\centering\small
\input{results/extended_contrasts_table.tex}
\caption{Extended 50-day paired differences: temporal fitted optimizer minus comparator; negative indicates fewer simulated infections under the temporal rule. These intervals quantify Monte Carlo variability across 12 shared simulated epidemics, not uncertainty about real population effects.}
\end{table}''')
print('Created extended risk/results/contrast inserts.')
