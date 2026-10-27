"""Generate LaTeX tables from executed, saved model-based experiments."""
from pathlib import Path
import pandas as pd
R=Path(__file__).resolve().parent/'results'
sm=pd.read_csv(R/'optimized_sbm_summary.csv')
hs=pd.read_csv(R/'optimized_haslemere_summary.csv')
sc=pd.read_csv(R/'optimized_sbm_contrasts.csv')
hc=pd.read_csv(R/'optimized_haslemere_contrasts.csv')
si=pd.read_csv(R/'optimized_sbm_interactions.csv')
smi=pd.read_csv(R/'optimized_sbm_missingness.csv')
hmi=pd.read_csv(R/'optimized_haslemere_missingness.csv')
rd=pd.read_csv(R/'risk_model_diagnostics.csv')
pols=['random','degree','contact','adaptive','opt_filter','opt_fitted']
labels={'random':'Random','degree':'Degree','contact':'Recent contact','adaptive':'Adaptive heuristic','opt_filter':'Mean-field optimizer','opt_fitted':'Fitted-risk optimizer'}

fmt=lambda x:f'{100*x:.1f}'
ci=lambda x:f'[{100*x.ci_low:+.1f}, {100*x.ci_high:+.1f}]'

def write(name,content):
 (R/name).write_text(content,encoding='utf-8')

lines=[r'\begin{tabular}{lrrr}\toprule',r'Prediction method & Brier $\downarrow$ & Log loss $\downarrow$ & AUC $\uparrow$\\\midrule']
for label,key in [('Mean-field filter','meanfield'),('Fitted logistic model','fitted')]:
 t=rd[(rd.split=='holdout')&(rd.model==key)].iloc[0]
 lines.append(f'{label} & {t.brier:.4f} & {t.logloss:.3f} & {t.auc:.3f}'+r'\\')
lines.append(r'\bottomrule\end{tabular}')
write('new_risk_diagnostics_table.tex','\n'.join(lines)+'\n')

lines=[r'\begin{tabular}{lrr}\toprule',r'Allocation policy & $r=1$ & $r=12$\\\midrule']
for pol in pols:
 a=sm[(sm.policy==pol)&(sm.visibility==1)&(sm.ratio==1)].iloc[0].attack
 b=sm[(sm.policy==pol)&(sm.visibility==1)&(sm.ratio==12)].iloc[0].attack
 lines.append(f'{labels[pol]} & {fmt(a)} & {fmt(b)}'+r'\\')
lines.append(r'\bottomrule\end{tabular}')
write('new_sbm_policy_table.tex','\n'.join(lines)+'\n')

lines=[r'\begin{tabular}{lrrr}\toprule',r'Policy & $q=0.25$ & $q=0.60$ & $q=1.00$\\\midrule']
for pol in pols:
 vals=[hs[(hs.policy==pol)&(hs.visibility==q)].iloc[0].attack for q in [.25,.6,1.]]
 lines.append(labels[pol]+' & '+' & '.join(fmt(v) for v in vals)+r'\\')
lines.append(r'\bottomrule\end{tabular}')
write('new_haslemere_policy_table.tex','\n'.join(lines)+'\n')

lines=[r'\begin{tabular}{llrr}\toprule',r'Network & Edge visibility & Fitted $-$ heuristic (pp) & 95\% MC interval\\\midrule']
for network,ct in [('SBM $r=1$',sc[sc.ratio==1]),('SBM $r=12$',sc[sc.ratio==12]),('Haslemere',hc)]:
 for q in [1.,.25]:
  t=ct[(ct.visibility==q)&(ct.contrast=='opt_fitted - adaptive')].iloc[0]
  lines.append(network+f' & {int(q*100)}\\% & {fmt(t.estimate)} & {ci(t)}'+r'\\')
lines.append(r'\bottomrule\end{tabular}')
write('new_paired_table.tex','\n'.join(lines)+'\n')

lines=[r'\begin{tabular}{lrr}\toprule',r'Comparison & Effect (pp) & 95\% MC interval\\\midrule']
for q in [1.,.25]:
 t=si[(si.ratio==12)&(si.visibility==q)].iloc[0]
 lines.append(f'$(r=12)-(r=1)$, $q={q:g}$ & {fmt(t.estimate)} & {ci(t)}'+r'\\')
for network,t in [('SBM $r=12$',smi[(smi.ratio==12)&(smi.policy=='opt_fitted')&(smi.visibility==.25)].iloc[0]),
                  ('Haslemere',hmi[(hmi.policy=='opt_fitted')&(hmi.visibility==.25)].iloc[0])]:
 lines.append(f'{network}, $q=.25$ vs 1 & {fmt(t.estimate)} & {ci(t)}'+r'\\')
lines.append(r'\bottomrule\end{tabular}')
write('new_interaction_table.tex','\n'.join(lines)+'\n')
coef=pd.read_csv(R/'risk_model_coefficients.csv')
lines=[r'\begin{tabular}{lr}\toprule',r'Feature & Standardized coefficient\\\midrule']
feature_label={'filter_logit_H':'Filter log-odds of hidden infection',
 'log1p_recent_contacts':'Recent confirmed one-hop contacts',
 'log1p_twohop_per_degree':'Two-hop contact exposure',
 'log1p_observed_degree':'Observed degree',
 'day_scaled':'Day',
 'log1p_days_since_test':'Time since last test',
 'filter_prob_susceptible':'Filtered susceptibility'}
for row in coef.itertuples():
 lines.append(f'{feature_label[row.feature]} & {row.coef_standardized:+.3f}'+r'\\')
lines.append(r'\bottomrule\end{tabular}')
write('new_risk_coefficients_table.tex','\n'.join(lines)+'\n')
print('Generated model-based tables')
