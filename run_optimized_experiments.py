"""Paired SBM and Haslemere tests: random, degree, contact, heuristic, filter and fitted optimizer.
Full true adjacency transmits; only masked observed adjacency is visible to the policies.
"""
import argparse
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulate_testing import ROOT,N,T,SEEDS,simulate
from model_based_core import simulate_with_filter
from run_haslemere import read_haslemere_graph
from run_structure_sensitivity import graph_from_uniforms, RATIOS, VIS, paired_summary
R=ROOT/'results';F=ROOT/'figures'
POLICIES=['random','degree','contact','adaptive','opt_filter','opt_fitted']


def calc_stats(raw,network):
    keys=['ratio','visibility','policy'] if network=='SBM' else ['visibility','policy']
    summ=raw.groupby(keys,sort=True).agg(n_reps=('rep','size'),attack=('attack_rate','mean'),
                se_attack=('attack_rate',lambda x:x.std(ddof=1)/np.sqrt(len(x))),
                hidden=('hidden_person_days','mean'),tests=('tests','mean'),
                test_yield=('test_yield','mean')).reset_index()
    pidx=['rep','ratio','visibility'] if network=='SBM' else ['rep','visibility']
    pv=raw.pivot(index=pidx,columns='policy',values='attack_rate').reset_index()
    cons=[]
    gcols=['ratio','visibility'] if network=='SBM' else ['visibility']
    for idx,g in pv.groupby(gcols,sort=True):
        idx=(idx,) if not isinstance(idx,tuple) else idx
        for comparison in ['contact','adaptive','opt_filter','random']:
            delta=g['opt_fitted']-g[comparison]
            cons.append({**dict(zip(gcols,idx)),'contrast':f'opt_fitted - {comparison}',**paired_summary(delta)})
    contrast=pd.DataFrame(cons)
    stem='optimized_sbm' if network=='SBM' else 'optimized_haslemere'
    raw.to_csv(R/f'{stem}_replicates.csv',index=False)
    summ.to_csv(R/f'{stem}_summary.csv',index=False)
    contrast.to_csv(R/f'{stem}_contrasts.csv',index=False)
    missing=[]
    for idx,g in raw.groupby(['ratio','policy'] if network=='SBM' else ['policy'],sort=True):
        idx=(idx,) if not isinstance(idx,tuple) else idx
        panel=g.pivot(index='rep',columns='visibility',values='attack_rate')
        for q in VIS:
            missing.append({**dict(zip(['ratio','policy'] if network=='SBM' else ['policy'],idx)),
                            'visibility':q,**paired_summary(panel[q]-panel[1.])})
    pd.DataFrame(missing).to_csv(R/f'{stem}_missingness.csv',index=False)
    if network=='SBM':
        interactions=[]
        for q in VIS:
            panel=pv[pv.visibility==q].assign(diff=lambda z:z.opt_fitted-z.adaptive).pivot(index='rep',columns='ratio',values='diff')
            for ratio in RATIOS:
                interactions.append({'visibility':q,'ratio':ratio,'comparison':'community_minus_r1',
                                     **paired_summary(panel[ratio]-panel[1.])})
        pd.DataFrame(interactions).to_csv(R/'optimized_sbm_interactions.csv',index=False)
    return summ,contrast


def experiment_sbm(model,reps=30,seed=2362026,budget=20):
    prior=pd.read_csv(R/'structure_replicates.csv')
    assert prior.rep.nunique()>=reps and budget==20 and seed==2362026, 'Reused baselines require original seed 2362026, budget 20 and <=50 reps.'
    prior=prior[prior.rep<reps].copy()
    prior['test_yield']=prior.positive_tests/prior.tests
    rows=prior[['rep','ratio','visibility','policy','attack_rate','hidden_person_days','tests','test_yield']].to_dict('records')
    cache_path=R/'optimized_sbm_checkpoint.csv'
    cached=pd.read_csv(cache_path).to_dict('records') if cache_path.exists() else []
    complete={int(v['rep']) for v in cached}
    for rep in range(reps):
        if rep in complete:continue
        rng=np.random.default_rng(np.random.SeedSequence([seed,101,rep]))
        graph_u=rng.random((N,N))
        mask_up=np.triu(rng.random((N,N)),1)
        mask=mask_up+mask_up.T
        idx=rng.choice(N,SEEDS,replace=False)
        innovations={key:rng.random((T,N)) for key in ['inf','rec','diag','test','tie']}
        for ratio in RATIOS:
            adj=sparse.csr_matrix(graph_from_uniforms(graph_u,ratio))
            for q in VIS:
                obs=adj.multiply(mask<q).tocsr()
                for pol in ["opt_filter","opt_fitted"]:
                    out=simulate_with_filter(adj,obs,idx,budget,pol,innovations,risk_model=model)
                    cached.append(dict(rep=rep,ratio=ratio,visibility=q,policy=pol,
                                     attack_rate=out['attack_rate'],hidden_person_days=out['hidden_person_days'],
                                     tests=out['tests'],test_yield=out['test_yield']))
        pd.DataFrame(cached).to_csv(cache_path,index=False)
        if (rep+1)%5==0: print('SBM',rep+1,'/',reps,flush=True)
    rows.extend([v for v in cached if int(v['rep'])<reps])
    raw=pd.DataFrame(rows);return calc_stats(raw,'SBM')


def experiment_haslemere(model,csv,reps=30,seed=2362026,budget=20):
    graph,meta=read_haslemere_graph(csv)
    prior=pd.read_csv(R/'haslemere_visibility_replicates.csv')
    assert prior.rep.nunique()>=reps and budget==20 and seed==2362026, 'Reused baselines require original seed 2362026, budget 20 and <=50 reps.'
    prior=prior[prior.rep<reps].copy()
    prior['test_yield']=prior.positive_tests/prior.tests
    rows=prior[['rep','visibility','policy','attack_rate','hidden_person_days','tests','test_yield']].to_dict('records')
    n=graph.shape[0]
    upper=sparse.triu(graph,k=1,format='coo')
    cache_path=R/'optimized_haslemere_checkpoint.csv'
    cached=pd.read_csv(cache_path).to_dict('records') if cache_path.exists() else []
    complete={int(v['rep']) for v in cached}
    for rep in range(reps):
        if rep in complete:continue
        rng=np.random.default_rng(np.random.SeedSequence([seed,90,rep]))
        idx=rng.choice(n,SEEDS,replace=False)
        innovations={key:rng.random((T,n)) for key in ['inf','rec','diag','test','tie']}
        mask=rng.random(len(upper.data))
        for q in VIS:
            kept=mask<q
            i,j=upper.row[kept],upper.col[kept]
            obs=sparse.csr_matrix((np.ones(2*len(i)),(np.r_[i,j],np.r_[j,i])),shape=graph.shape)
            for pol in ["opt_filter","opt_fitted"]:
                out=simulate_with_filter(graph,obs,idx,budget,pol,innovations,risk_model=model)
                cached.append(dict(rep=rep,visibility=q,policy=pol,
                                 attack_rate=out['attack_rate'],hidden_person_days=out['hidden_person_days'],
                                 tests=out['tests'],test_yield=out['test_yield']))
        pd.DataFrame(cached).to_csv(cache_path,index=False)
        if (rep+1)%5==0: print('Haslemere',rep+1,'/',reps,flush=True)
    rows.extend([v for v in cached if int(v['rep'])<reps])
    pd.DataFrame([meta]).to_csv(R/'haslemere_graph_metrics.csv',index=False)
    raw=pd.DataFrame(rows);return calc_stats(raw,'Haslemere')


def plots():
    sm=pd.read_csv(R/'optimized_sbm_summary.csv')
    hs=pd.read_csv(R/'optimized_haslemere_summary.csv')
    cm=pd.read_csv(R/'optimized_sbm_contrasts.csv')
    # policy means on two representative SBM levels, full network
    fig,ax=plt.subplots(figsize=(8.1,4.1))
    for ratio in [1.,12.]:
        g=sm[(sm.ratio==ratio)&(sm.visibility==1)].set_index('policy').loc[POLICIES]
        ax.plot(np.arange(len(POLICIES)),100*g.attack,marker='o',label=f'Community ratio {ratio:g}')
    ax.set_xticks(np.arange(len(POLICIES)),['Random','Degree','Contact','Heuristic','MF optimizer','Fitted optimizer'],rotation=18,ha='right')
    ax.set_ylabel('Infected by day 50 (%)');ax.set_title('SBM: comparison of testing policies')
    ax.legend();fig.tight_layout();fig.savefig(F/'optimized_sbm_policies.png',dpi=190);plt.close(fig)
    # relative gain over heuristic; primary community structure question
    fig,ax=plt.subplots(figsize=(7.3,3.8))
    for q in [1.,.25]:
        p=cm[(cm.visibility==q)&(cm.contrast=='opt_fitted - adaptive')].sort_values('ratio')
        ax.errorbar(p.ratio,100*p.estimate,yerr=196*p.se,marker='o',capsize=2.5,label=f'{100*q:g}% edges observed')
    ax.axhline(0,color='grey',lw=1,ls='--')
    ax.set_xticks(RATIOS,['1','1.5','2','4','8','12']);ax.set_xlabel('SBM $p_{in}/p_{out}$')
    ax.set_ylabel('Fitted optimizer - heuristic (pp)')
    ax.set_title('Does community strength change optimization gains?');ax.legend()
    fig.tight_layout();fig.savefig(F/'optimized_community.png',dpi=190);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.4,3.8))
    for pol in ['contact','adaptive','opt_filter','opt_fitted']:
        g=hs[hs.policy==pol].sort_values('visibility')
        ax.errorbar(100*g.visibility,100*g.attack,yerr=196*g.se_attack,marker='o',capsize=2,
                    label={'opt_fitted':'Fitted optimizer','opt_filter':'MF optimizer','adaptive':'Heuristic','contact':'Recent contact'}[pol])
    ax.set_xlabel('Observed Haslemere edges (%)');ax.set_ylabel('Infected by day 50 (%)')
    ax.set_xticks([25,50,60,75,100]);ax.set_title('Empirical contact network: incomplete observation')
    ax.legend(ncol=2,fontsize=8);fig.tight_layout();fig.savefig(F/'optimized_haslemere_visibility.png',dpi=190);plt.close(fig)
    # empirical trade-off burden under full and partial network
    fig,ax=plt.subplots(figsize=(7.4,3.8))
    for ratio in [1.,12.]:
        for pol in ['adaptive','opt_fitted']:
            g=sm[(sm.ratio==ratio)&(sm.policy==pol)].sort_values('visibility')
            ax.plot(100*g.visibility,100*g.attack,marker='o',ls='-' if pol=='opt_fitted' else '--',label=f'r={ratio:g}, {pol}')
    ax.set_xlabel('Observed edges (%)');ax.set_ylabel('Infected by day 50 (%)');ax.set_xticks([25,50,60,75,100])
    ax.set_title('SBM: absolute benefit and loss from missing contacts');ax.legend(fontsize=8,ncol=2)
    fig.tight_layout();fig.savefig(F/'optimized_sbm_visibility.png',dpi=190);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--csv',default=str(ROOT/'data'/'Kissler_DataS1.csv'))
    p.add_argument('--reps',type=int,default=30)
    p.add_argument('--seed',type=int,default=2362026)
    p.add_argument('--skip-sbm',action='store_true');p.add_argument('--skip-haslemere',action='store_true')
    args=p.parse_args()
    model=joblib.load(ROOT/'models'/'risk_model.joblib')
    if not args.skip_sbm:experiment_sbm(model,args.reps,args.seed)
    if not args.skip_haslemere:experiment_haslemere(model,args.csv,args.reps,args.seed)
    if not (args.skip_sbm or args.skip_haslemere):plots()
