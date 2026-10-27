"""STAT 236: network-aware screening simulation. Python 3 + numpy/pandas/matplotlib.

Run: python simulate_testing.py --reps 50
Outputs CSV summary, paired differences, and figures. Uses fixed network/innovation
seeds across policies within a replicate for a paired design.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
FIGS = ROOT / 'figures'
RESULTS.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

N=180; K=3; T=50; T0=8; EXPECTED_DEGREE=8
BETA=0.055; GAMMA=0.12; DELTA=0.025; SENS=0.90
COOLDOWN=5; WINDOW=7; SEEDS=5
POLICIES=('random', 'degree', 'contact', 'adaptive')


def generate_sbm(rng, ratio):
    """K equal-sized blocks, mean expected degree fixed at EXPECTED_DEGREE."""
    size = N//K
    pin = EXPECTED_DEGREE / ((size-1)+(N-size)/ratio)
    pout=pin/ratio
    g=np.repeat(np.arange(K), size)
    probs=np.where(g[:,None]==g[None,:], pin,pout)
    u=rng.random((N,N))
    a=np.triu(u<probs,1).astype(np.float64)
    return a+a.T


def choose(policy, budget, eligible, obs, deg, confirm_day, day, u_random):
    n = len(eligible)
    inds=np.flatnonzero(eligible)
    if not len(inds) or budget <= 0:
        return np.empty(0,dtype=np.int64)
    if policy=='random':
        score=np.zeros(n)
    elif policy=='degree':
        score=deg
    else:
        confirmed_recent=np.where((confirm_day>=0)&((day-confirm_day)<WINDOW),
                                  np.exp(-np.maximum(0,day-confirm_day)/4),0.0)
        one=obs@confirmed_recent
        if policy=='contact':
            score=one
        elif policy=='adaptive':
            two=obs@one
            # A transparent risk *proxy*, not a calibrated infection probability.
            risk_proxy=1-np.exp(-(0.02 + 0.65*one + 0.10*two/(1+deg/5)))
            score=risk_proxy*(1+0.10*deg)
        else:
            raise ValueError(policy)
    # Randomized tie break; tiny perturbations only matter at numerical ties.
    ids=inds[np.lexsort((u_random[inds],-score[inds]))][:budget]
    return ids


def simulate(true_adj, obs_adj, seed_idx, budget, policy, innovations):
    n=true_adj.shape[0]
    state=np.zeros(n,dtype=np.int8)  # 0 S, 1 hidden, 2 confirmed/isolated, 3 removed
    state[seed_idx]=1
    ever_infected=np.zeros(n,dtype=bool)
    ever_infected[seed_idx]=True
    confirm_day=np.full(n,-1,dtype=np.int32)
    last_test=np.full(n,-999,dtype=np.int32)
    deg=np.asarray(obs_adj.sum(axis=1)).ravel()
    total_tests=0; positive_tests=0; hidden_days=0; isolated_days=0
    infections_daily=[]; hidden_daily=[]; test_daily=[]
    for d in range(T):
        # Background symptom-related/ordinary detection occurs before allocation.
        spontaneous=(state==1)&(innovations['diag'][d]<DELTA)
        state[spontaneous]=2
        confirm_day[spontaneous]=d

        tested=np.empty(0,dtype=np.int64)
        if d>=T0 and budget>0:
            # Eligibility only uses observable information (ever diagnosed, prior tests).
            eligible=(confirm_day<0)&((d-last_test)>=COOLDOWN)
            tested=choose(policy,budget,eligible,obs_adj,deg,confirm_day,d,innovations['tie'][d])
            last_test[tested]=d
            total_tests+=len(tested)
            positives=tested[(state[tested]==1)&(innovations['test'][d,tested]<SENS)]
            state[positives]=2
            confirm_day[positives]=d
            positive_tests+=len(positives)

        # Only undiagnosed infected people transmit; newly infected enter H at day end.
        infectors=(state==1).astype(float)
        force=BETA*(true_adj@infectors)
        susceptible=(state==0)
        new_inf=susceptible&(innovations['inf'][d]<(1-np.exp(-force)))
        # Recovery acts on previous H/C, not on new cases.
        rec=((state==1)|(state==2))&(innovations['rec'][d]<GAMMA)
        state[rec]=3
        state[new_inf]=1
        ever_infected[new_inf]=True
        hidden_days+=int(np.sum(state==1))
        isolated_days+=int(np.sum(state==2))
        infections_daily.append(int(ever_infected.sum()))
        hidden_daily.append(int((state==1).sum()))
        test_daily.append(len(tested))
    return {'attack_rate':ever_infected.mean(), 'cumulative_infections':int(ever_infected.sum()),
            'hidden_person_days':hidden_days,'isolation_person_days':isolated_days,
            'tests':total_tests,'positive_tests':positive_tests,
            'test_yield':positive_tests/total_tests if total_tests else np.nan,
            'infection_curve':infections_daily, 'hidden_curve':hidden_daily, 'test_curve':test_daily}


def run(reps=50, base_seed=2362026):
    rows=[]
    for structure,ratio in [('weak',1.5),('strong',12.0)]:
        for rep in range(reps):
            rng=np.random.default_rng(np.random.SeedSequence([base_seed, 1 if structure=='weak' else 2,rep]))
            adj=generate_sbm(rng,ratio)
            idx=rng.choice(N,size=SEEDS,replace=False)
            uniforms={key:rng.random((T,N)) for key in ['inf','rec','diag','test','tie']}
            visibility_uniform=rng.random((N,N))
            visibility_uniform=np.triu(visibility_uniform,1)
            visibility_uniform=visibility_uniform+visibility_uniform.T
            for vis in [1.0,0.60]:
                obs=adj*(visibility_uniform<vis)
                obs=(obs>0).astype(float)
                for budget in [0,8,20]:
                    pols=['none'] if budget==0 else POLICIES
                    for policy in pols:
                        x=simulate(adj,obs,idx,budget,policy,uniforms)
                        row={k:x[k] for k in ['attack_rate','cumulative_infections','hidden_person_days',
                                                 'isolation_person_days','tests','positive_tests','test_yield']}
                        row.update({'structure':structure,'rep':rep,'visibility':vis,
                                    'budget':budget,'policy':policy,
                                    'observed_degree':float(np.mean(obs.sum(axis=1))),
                                    'true_degree':float(np.mean(adj.sum(axis=1)))})
                        rows.append(row)
    df=pd.DataFrame(rows)
    df.to_csv(RESULTS/'replicate_results.csv',index=False)
    group=['structure','visibility','budget','policy']
    summary=(df.groupby(group,sort=False).agg(
        n_replicates=('attack_rate','size'),
        mean_attack=('attack_rate','mean'),se_attack=('attack_rate',lambda x:x.std(ddof=1)/np.sqrt(len(x))),
        mean_hidden=('hidden_person_days','mean'),
        mean_tests=('tests','mean'),mean_yield=('test_yield','mean'),
        mean_isolation=('isolation_person_days','mean')).reset_index())
    summary.to_csv(RESULTS/'summary.csv',index=False)
    pairs=[]
    for structure in ['weak','strong']:
        for vis in [1.0,0.6]:
            for budget in [8,20]:
                subset=df[(df.structure==structure)&(df.visibility==vis)&(df.budget==budget)]
                pivot=subset.pivot(index='rep',columns='policy',values='attack_rate')
                for comp in ['random','contact','degree']:
                    diff=pivot['adaptive']-pivot[comp]
                    se=diff.std(ddof=1)/np.sqrt(len(diff))
                    pairs.append({'structure':structure,'visibility':vis,'budget':budget,
                                  'contrast':'adaptive - '+comp,'mean_difference':diff.mean(),
                                  'ci_lower':diff.mean()-1.96*se,'ci_upper':diff.mean()+1.96*se,
                                  'n_replicates':len(diff)})
    pd.DataFrame(pairs).to_csv(RESULTS/'paired_contrasts.csv',index=False)
    # Plot 1: policy comparison at fixed budget, full observation.
    subset=summary[(summary.visibility==1.0)&(summary.budget==20)&(summary.policy!='none')]
    fig,ax=plt.subplots(figsize=(8.0,4.6))
    names=['random','contact','degree','adaptive']
    means=[]; errs=[]
    for structure in ['weak','strong']:
        temp=subset[subset.structure==structure].set_index('policy').loc[names]
        means.append(temp.mean_attack.to_numpy()); errs.append(1.96*temp.se_attack.to_numpy())
    x=np.arange(len(names)); w=0.35
    ax.bar(x-w/2,means[0],w,yerr=errs[0],label='Weak communities',capsize=3)
    ax.bar(x+w/2,means[1],w,yerr=errs[1],label='Strong communities',capsize=3)
    ax.set_xticks(x,names); ax.set_ylabel('Final infected fraction');ax.set_ylim(0,1.02)
    ax.set_title('Testing policy comparison | 20 tests/day | fully observed network')
    ax.legend(loc='upper right');fig.tight_layout();fig.savefig(FIGS/'policy_comparison.png',dpi=180);plt.close(fig)
    # Plot 2: robustness by edge observation, paired differences.
    paired=pd.read_csv(RESULTS/'paired_contrasts.csv')
    fig,ax=plt.subplots(figsize=(7.7,4.4))
    for structure in ['weak','strong']:
        s=paired[(paired.structure==structure)&(paired.budget==20)&
                 (paired.contrast=='adaptive - random')].sort_values('visibility')
        ax.errorbar(s.visibility, s.mean_difference,yerr=[s.mean_difference-s.ci_lower,s.ci_upper-s.mean_difference],
                    marker='o',capsize=4,label=structure.title()+' communities')
    ax.axhline(0,color='gray',lw=1,ls='--')
    ax.set_xlim(0.5,1.1);ax.set_xticks([0.6,1.0],['60%','100%'])
    ax.set_xlabel('Observed edge fraction');ax.set_ylabel('Adaptive minus random attack rate')
    ax.set_title('Effect of incomplete network observation | 20 tests/day')
    ax.legend();fig.tight_layout();fig.savefig(FIGS/'robustness.png',dpi=180);plt.close(fig)
    # Plot 3: mean attack versus budget for full observation
    fig,ax=plt.subplots(figsize=(7.7,4.4))
    for policy in ['none','random','contact','degree','adaptive']:
        s=summary[(summary.structure=='strong')&(summary.visibility==1.0)&(summary.policy==policy)]
        if policy=='none':
            ax.scatter(s.budget,s.mean_attack,label='No testing',marker='x',s=70)
        else:
            ax.plot(s.budget,s.mean_attack,marker='o',label=policy.title())
    ax.set_xticks([0,8,20]); ax.set_ylim(0,1.02);ax.set_xlabel('Daily testing budget')
    ax.set_ylabel('Final infected fraction');ax.set_title('Strong community structure | full observation')
    ax.legend(ncol=2);fig.tight_layout();fig.savefig(FIGS/'budget.png',dpi=180);plt.close(fig)
    print(summary.to_string(index=False,float_format=lambda v:f'{v:.3f}'))
    print('\nPaired contrasts (adaptive - random):\n',pd.DataFrame(pairs).query("contrast == 'adaptive - random'").to_string(index=False))
    print('\nSaved CSV and PNG files to',ROOT)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--reps',type=int,default=50)
    ap.add_argument('--seed',type=int,default=2362026)
    a=ap.parse_args()
    run(a.reps,a.seed)
