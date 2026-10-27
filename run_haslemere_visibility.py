"""Empirical contact-network missingness sensitivity; infections are simulated."""
import argparse
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulate_testing import ROOT,T,SEEDS,POLICIES,simulate
from run_haslemere import read_haslemere_graph, obtain_data
from run_structure_sensitivity import VIS,paired_summary
R=ROOT/'results';F=ROOT/'figures'

def run(csv,reps=100,seed=2362026,budget=20):
    path=obtain_data(csv)
    graph,meta=read_haslemere_graph(path)
    pd.DataFrame([meta]).to_csv(R/'haslemere_graph_metrics.csv',index=False)
    n=graph.shape[0];upper=sparse.triu(graph,k=1,format='coo')
    rows=[]
    for rep in range(reps):
        rng=np.random.default_rng(np.random.SeedSequence([seed,90,rep]))
        initial=rng.choice(n,SEEDS,replace=False)
        innovations={key:rng.random((T,n)) for key in ['inf','rec','diag','test','tie']}
        # Per-edge uniform retained for all visibility levels: nested observed graphs.
        uniform=rng.random(len(upper.data))
        for vis in VIS:
            keep=uniform<vis
            i,j=upper.row[keep],upper.col[keep]
            obs=sparse.csr_matrix((np.ones(2*len(i)),(np.r_[i,j],np.r_[j,i])),shape=graph.shape)
            for pol in POLICIES:
                o=simulate(graph,obs,initial,budget,pol,innovations)
                rows.append(dict(rep=rep,visibility=vis,budget=budget,policy=pol,
                    attack_rate=o['attack_rate'],hidden_person_days=o['hidden_person_days'],
                    tests=o['tests'],positive_tests=o['positive_tests'],
                    observed_edges=len(i)))
        if (rep+1)%20==0:print('Haslemere visibility replicate',rep+1,'/',reps,flush=True)
    raw=pd.DataFrame(rows);raw.to_csv(R/'haslemere_visibility_replicates.csv',index=False)
    summary=raw.groupby(['visibility','policy'],sort=True).agg(n_reps=('rep','size'),
       attack=('attack_rate','mean'),se_attack=('attack_rate',lambda x:x.std(ddof=1)/np.sqrt(len(x))),
       mean_hidden=('hidden_person_days','mean'),mean_edges=('observed_edges','mean')).reset_index()
    summary.to_csv(R/'haslemere_visibility_summary.csv',index=False)
    pv=raw.pivot(index=['rep','visibility'],columns='policy',values='attack_rate').reset_index()
    pv['adaptive_minus_contact']=pv['adaptive']-pv['contact'];pv['adaptive_minus_random']=pv['adaptive']-pv['random']
    out=[]
    for visibility,x in pv.groupby('visibility'):
        for comp in ['contact','random']:
            out.append(dict(visibility=visibility,contrast='adaptive - '+comp,
                            **paired_summary(x['adaptive_minus_'+comp])))
    paired=pd.DataFrame(out);paired.to_csv(R/'haslemere_visibility_contrasts.csv',index=False)
    effects=[]
    for pol in ['adaptive','contact','degree']:
        t=raw[raw.policy==pol].pivot(index='rep',columns='visibility',values='attack_rate')
        for q in VIS:
            effects.append(dict(visibility=q,policy=pol,**paired_summary(t[q]-t[1.])))
    pd.DataFrame(effects).to_csv(R/'haslemere_visibility_effects.csv',index=False)
    fig,ax=plt.subplots(figsize=(7.0,3.5))
    for pol in ['random','contact','degree','adaptive']:
        z=summary[summary.policy==pol].sort_values('visibility')
        ax.errorbar(100*z.visibility,100*z.attack,yerr=196*z.se_attack,
             marker='o',capsize=2.5,label=pol.title())
    ax.set_xticks([25,50,60,75,100]);ax.set_xlabel('Observed true contact edges (%)')
    ax.set_ylabel('Infected by day 50 (%)');ax.set_title('Haslemere topology: missing-edge sensitivity')
    ax.legend(ncol=2,fontsize=8);fig.tight_layout();fig.savefig(F/'haslemere_visibility.png',dpi=190);plt.close(fig)
    print(summary.to_string(index=False),flush=True)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--csv',default=str(ROOT/'data'/'Kissler_DataS1.csv'));p.add_argument('--reps',type=int,default=100);p.add_argument('--seed',type=int,default=2362026);p.add_argument('--budget',type=int,default=20)
    args=p.parse_args();run(args.csv,args.reps,args.seed,args.budget)
