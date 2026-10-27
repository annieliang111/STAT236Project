"""Controlled SBM sensitivity study: community strength x contact observability.

All models use the same epidemic and screening implementations as
simulate_testing.py. Within a replicate, graph-construction uniforms, observed
edge-mask uniforms, initial seeds, and epidemic random innovations are shared
across the factorial design, allowing *paired* contrasts. Report uncertainty
is across independent replicate seeds, not observed epidemiological outcomes.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulate_testing import ROOT, N, K, T, SEEDS, EXPECTED_DEGREE, POLICIES, simulate

RATIOS=[1.,1.5,2.,4.,8.,12.]
VIS=[.25,.50,.60,.75,1.]
R=ROOT/'results'; F=ROOT/'figures'
R.mkdir(exist_ok=True);F.mkdir(exist_ok=True)


def graph_from_uniforms(u, ratio):
    size=N//K
    pin=EXPECTED_DEGREE/((size-1)+(N-size)/ratio)
    pout=pin/ratio
    z=np.repeat(np.arange(K),size)
    p=np.where(z[:,None]==z[None,:],pin,pout)
    up=np.triu(u<p,1).astype(float)
    return up+up.T


def paired_summary(x):
    values=np.asarray(x,dtype=float)
    n=len(values)
    assert n>1
    se=values.std(ddof=1)/np.sqrt(n)
    return dict(n_replicates=n,estimate=values.mean(),se=se,
                ci_low=values.mean()-1.96*se,ci_high=values.mean()+1.96*se)


def run(reps=100,seed=2362026,budget=20):
    rows=[]
    for rep in range(reps):
        rng=np.random.default_rng(np.random.SeedSequence([seed,101,rep]))
        graph_u=rng.random((N,N))
        mask_up=np.triu(rng.random((N,N)),1)
        mask_u=mask_up+mask_up.T
        initial=rng.choice(N,size=SEEDS,replace=False)
        innovations={key:rng.random((T,N)) for key in ['inf','rec','diag','test','tie']}
        for ratio in RATIOS:
            adj=graph_from_uniforms(graph_u,ratio)
            for visibility in VIS:
                obs=adj*(mask_u<visibility)
                for policy in POLICIES:
                    s=simulate(adj,obs,initial,budget,policy,innovations)
                    rows.append(dict(rep=rep,ratio=ratio,visibility=visibility,
                        budget=budget,policy=policy,attack_rate=s['attack_rate'],
                        hidden_person_days=s['hidden_person_days'],positive_tests=s['positive_tests'],
                        tests=s['tests'],mean_true_degree=adj.sum()/N,
                        mean_observed_degree=obs.sum()/N))
        if (rep+1)%20==0:print('SBM sensitivity replicate',rep+1,'/',reps,flush=True)
    raw=pd.DataFrame(rows);raw.to_csv(R/'structure_replicates.csv',index=False)
    summary=(raw.groupby(['ratio','visibility','budget','policy'],sort=True)
               .agg(n_reps=('rep','size'),attack=('attack_rate','mean'),
                    se_attack=('attack_rate',lambda v:v.std(ddof=1)/np.sqrt(len(v))),
                    hidden=('hidden_person_days','mean'),
                    degree=('mean_true_degree','mean'),
                    observed_degree=('mean_observed_degree','mean')).reset_index())
    summary.to_csv(R/'structure_summary.csv',index=False)
    piv=raw.pivot(index=['rep','ratio','visibility'],columns='policy',values='attack_rate').reset_index()
    piv['adaptive_minus_contact']=piv['adaptive']-piv['contact']
    piv['adaptive_minus_random']=piv['adaptive']-piv['random']
    contrasts=[]
    for (ratio,visibility),group in piv.groupby(['ratio','visibility'],sort=True):
        for metric in ['adaptive_minus_contact','adaptive_minus_random']:
            contrasts.append(dict(ratio=ratio,visibility=visibility,contrast=metric,
                                  **paired_summary(group[metric])))
    contrasts=pd.DataFrame(contrasts);contrasts.to_csv(R/'structure_contrasts.csv',index=False)
    # Paired differences in policy advantage across network structures and observability.
    interactions=[]
    for vis in VIS:
        group=piv[piv.visibility==vis].pivot(index='rep',columns='ratio',values='adaptive_minus_contact')
        for ratio in RATIOS:
            interactions.append(dict(comparison='community_minus_r1',ratio=ratio,visibility=vis,
                                     **paired_summary(group[ratio]-group[1.])))
    for ratio in RATIOS:
        group=piv[piv.ratio==ratio].pivot(index='rep',columns='visibility',values='adaptive_minus_contact')
        for vis in VIS:
            interactions.append(dict(comparison='missingness_minus_full',ratio=ratio,visibility=vis,
                                     **paired_summary(group[vis]-group[1.])))
    pd.DataFrame(interactions).to_csv(R/'structure_interactions.csv',index=False)
    # Absolute attack-rate degradation for adaptive and contact, q versus full observation.
    missing=[]
    for ratio in RATIOS:
        for pol in ['adaptive','contact','degree']:
            group=raw[(raw.ratio==ratio)&(raw.policy==pol)].pivot(index='rep',columns='visibility',values='attack_rate')
            for vis in VIS:
                missing.append(dict(ratio=ratio,policy=pol,visibility=vis,
                                    **paired_summary(group[vis]-group[1.])))
    pd.DataFrame(missing).to_csv(R/'structure_missingness_effects.csv',index=False)

    # Paired advantage versus actual community-strength ratio.
    fig,ax=plt.subplots(figsize=(7.0,3.5))
    for vis in [1.,.25]:
        s=contrasts[(contrasts.visibility==vis)&(contrasts.contrast=='adaptive_minus_contact')].sort_values('ratio')
        ax.errorbar(s.ratio,100*s.estimate,yerr=196*s.se,marker='o',capsize=2.5,
                    label=f'{int(vis*100)}% edges observed')
    ax.axhline(0,ls='--',lw=1,color='#777777')
    ax.set_xticks(RATIOS,['1','1.5','2','4','8','12'])
    ax.set_xlabel(r'Community strength $p_{in}/p_{out}$')
    ax.set_ylabel('Adaptive - contact (percentage points)')
    ax.set_title('Incremental testing benefit across SBM structures')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(F/'community_sensitivity.png',dpi=190);plt.close(fig)
    # Show both relative policy benefits and total burden under missing contacts.
    fig,ax=plt.subplots(figsize=(7.0,3.6))
    for ratio in [1.5,12.]:
        for pol,sty in [('adaptive','-'),('contact','--')]:
            s=summary[(summary.ratio==ratio)&(summary.policy==pol)].sort_values('visibility')
            ax.errorbar(100*s.visibility,100*s.attack,yerr=196*s.se_attack,ls=sty,
                        marker='o',markersize=4,capsize=2,label=f'r={ratio:g} · {pol}')
    ax.set_xlabel('Observed true contact edges (%)')
    ax.set_ylabel('Infected by day 50 (%)')
    ax.set_title('Cost of incomplete contact observation')
    ax.set_xticks([25,50,60,75,100]);ax.legend(ncol=2,fontsize=8)
    fig.tight_layout();fig.savefig(F/'missingness_sensitivity.png',dpi=190);plt.close(fig)
    # Heatmap descriptive only; inferential uncertainty shown in table/CI plot.
    mat=(contrasts[contrasts.contrast=='adaptive_minus_contact'].pivot(index='ratio',columns='visibility',values='estimate')
         .loc[RATIOS,VIS])
    fig,ax=plt.subplots(figsize=(6.5,3.6));mesh=ax.imshow(100*mat.to_numpy(),cmap='RdBu_r',vmin=-5,vmax=5,aspect='auto')
    ax.set_xticks(np.arange(len(VIS)),[f'{int(x*100)}%' for x in VIS]);ax.set_yticks(np.arange(len(RATIOS)),[f'{x:g}' for x in RATIOS]);ax.set_xlabel('Observed edges');ax.set_ylabel(r'$p_{in}/p_{out}$')
    for i in range(len(RATIOS)):
        for j in range(len(VIS)):
            v=100*mat.iloc[i,j]
            ax.text(j,i,f'{v:+.1f}',ha='center',va='center',fontsize=8,color='black' if abs(v)<3 else 'white')
    cb=fig.colorbar(mesh,ax=ax);cb.set_label('Adaptive - contact (pp)')
    ax.set_title('Testing advantage: community × information')
    fig.tight_layout();fig.savefig(F/'structure_heatmap.png',dpi=190);plt.close(fig)
    print('Saved sensitivity results:',len(raw),'policy runs',flush=True)
    print(contrasts[(contrasts.visibility==1)&(contrasts.contrast=='adaptive_minus_contact')][['ratio','estimate','ci_low','ci_high']].to_string(index=False),flush=True)
    return raw

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--reps',type=int,default=100);ap.add_argument('--seed',type=int,default=2362026);ap.add_argument('--budget',type=int,default=20)
    args=ap.parse_args();run(args.reps,args.seed,args.budget)
