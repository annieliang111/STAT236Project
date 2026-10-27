"""50-day SHCR simulation on Haslemere 5-minute contacts extended by daily-block resampling.

Observed day blocks 1-3 are preserved exactly. Days 4..days are independent draws
from observed days {1,2,3}, sampled once per paired replicate and reused across
all policy and contact-visibility conditions. All infection outcomes simulated.

This code deliberately DOES NOT add an Exposed (E) state: S -> H is immediate.
Only recorded daytime contacts permit transmission; nighttime diagnoses and
recovery continue, but no nighttime transmission is simulated.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import os
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
import joblib
from temporal_haslemere import load_contacts, N, SLOTS_PER_DAY, INITIAL_HIDDEN, INITIAL_CONFIRMED

ROOT=Path(__file__).resolve().parent
DAYS=50
DAILY_TESTS=20
ROUNDS=4
ROUND_SLOTS=48
TESTS_PER_ROUND=DAILY_TESTS//ROUNDS
GAMMA_DAY=0.12
DELTA_DAY=0.14
TEST_SENS=0.90
BETA_CONTACT=0.06
GAMMA_STEP=1-(1-GAMMA_DAY)**(1/288)
DELTA_STEP=1-(1-DELTA_DAY)**(1/288)
GAMMA_NIGHT=1-(1-GAMMA_DAY)**(96/288)
DELTA_NIGHT=1-(1-DELTA_DAY)**(96/288)
RECENCY_EFOLD_HOURS=12
RECENCY_DECAY=np.exp(-1/(RECENCY_EFOLD_HOURS*12))
RECENCY_NIGHT=RECENCY_DECAY**96
FEATURES=['filter_logit_h','log_recent_confirmed','log_recent_twohop','log_recent_degree',
          'log_cumulative_degree','filter_susceptible','fraction_of_days','log_time_since_test']
POLICIES=['none','random','degree','contact','adaptive','opt_filter','opt_temporal_fitted','opt_cumulative_fitted']


def rng_for(seed, stage, rep, channel):
    return np.random.default_rng(np.random.SeedSequence([seed,stage,rep,channel]))


def sampled_day_blocks(seed, rep, days=DAYS):
    rng=rng_for(seed,100,rep,0)
    return np.r_[np.arange(3),rng.integers(0,3,size=days-3)]


def adjacency_hist(pi,pj,weights,values):
    a=np.bincount(pi,weights=weights*values[pj],minlength=N)
    b=np.bincount(pj,weights=weights*values[pi],minlength=N)
    return a+b


def feature_matrix(q, recent, cumulative, pi, pj, confirmed, last_test, k, days):
    ph=np.clip(q[:,1],1e-6,1-1e-6)
    c=(confirmed>=0).astype(float)
    direct=adjacency_hist(pi,pj,recent,c)
    two=adjacency_hist(pi,pj,recent,direct)
    degree=adjacency_hist(pi,pj,recent,np.ones(N))
    cumdeg=adjacency_hist(pi,pj,cumulative,np.ones(N))/max(1,k)
    since=np.where(last_test>=0,k-last_test,days*SLOTS_PER_DAY)
    X=np.column_stack([np.log(ph/(1-ph)), np.log1p(direct),
        np.log1p(two/(1+degree)), np.log1p(degree),np.log1p(cumdeg),
        q[:,0], np.full(N,k/(days*SLOTS_PER_DAY)),np.log1p(since)])
    return X,degree,direct,two


def filter_diagnosis(q,newly,previous,p_d):
    nmask=~newly & ~previous
    if nmask.any():
        p=q[nmask].copy();normalizer=np.maximum(1-p_d*p[:,1],1e-12)
        p[:,1]*=(1-p_d)
        q[nmask]=p/normalizer[:,None]
    q[newly]=[0,0,1,0]


def filter_tests(q,tested,positives):
    neg=np.setdiff1d(tested,positives,assume_unique=True)
    if len(neg):
        p=q[neg].copy();z=np.maximum(1-TEST_SENS*p[:,1],1e-12)
        p[:,1]*=(1-TEST_SENS)
        q[neg]=p/z[:,None]
    if len(positives):q[positives]=[0,0,1,0]


def filter_predict(q,i,j,prob_recover):
    eta=1-np.exp(-BETA_CONTACT)
    logs=np.log1p(-eta*q[:,1])
    logsurvive=np.bincount(i,weights=logs[j],minlength=N)+np.bincount(j,weights=logs[i],minlength=N)
    p_inf=-np.expm1(logsurvive)
    old=q.copy()
    q[:,0]=old[:,0]*(1-p_inf)
    q[:,1]=old[:,1]*(1-prob_recover)+old[:,0]*p_inf
    q[:,2]=old[:,2]*(1-prob_recover)
    q[:,3]=old[:,3]+prob_recover*(old[:,1]+old[:,2])
    q/=np.maximum(q.sum(1,keepdims=True),1e-12)


def run_episode(pi,pj,slot_ids,day_blocks,vis_mask,init_h,init_c,policy,
                seed=90236,rep=0,model=None,days=DAYS,collect=False,
                max_rows_per_round=60):
    """Run one episode. Stream-based CRN is paired across policies even with differing actions."""
    inf_rng=rng_for(seed,200,rep,1);rec_rng=rng_for(seed,200,rep,2)
    diag_rng=rng_for(seed,200,rep,3);test_rng=rng_for(seed,200,rep,4)
    tie_rng=rng_for(seed,200,rep,5);sample_rng=rng_for(seed,200,rep,6)
    m=len(pi)
    state=np.zeros(N,np.int8);state[init_h]=1;state[init_c]=2
    ever=np.zeros(N,bool);ever[init_h]=True;ever[init_c]=True
    confirmed=np.full(N,-1,int);confirmed[init_c]=0
    last_test=np.full(N,-1,int)
    q=np.zeros((N,4),float)
    free=np.ones(N,bool);free[init_c]=False
    q[free,0]=1-INITIAL_HIDDEN/(N-INITIAL_CONFIRMED)
    q[free,1]=INITIAL_HIDDEN/(N-INITIAL_CONFIRMED)
    q[init_c,2]=1
    recent=np.zeros(m,float);cumul=np.zeros(m,float)
    X_rows=[];y_rows=[]
    total_tests=positive_tests=new_infections=0
    hidden_person_hours=0.
    daily_infected=[]
    for day in range(days):
        orig=int(day_blocks[day])
        for s in range(SLOTS_PER_DAY):
            k=day*SLOTS_PER_DAY+s
            ids=slot_ids[orig*SLOTS_PER_DAY+s]
            i=pi[ids];j=pj[ids]
            previous=confirmed>=0
            new_diag=(state==1)&(diag_rng.random(N)<DELTA_STEP)
            state[new_diag]=2;confirmed[new_diag]=k
            filter_diagnosis(q,new_diag,previous,DELTA_STEP)
            if s%ROUND_SLOTS==0:
                eligible=(confirmed<0)&((last_test<0)|(k-last_test>=SLOTS_PER_DAY))
                X,degree,direct,two=feature_matrix(q,recent,cumul,pi,pj,confirmed,last_test,k,days)
                if collect:
                    pool=np.flatnonzero(eligible)
                    if len(pool)>max_rows_per_round:
                        # Unbiased sampling from eligible persons; training from separate episodes.
                        pool=sample_rng.choice(pool,size=max_rows_per_round,replace=False)
                    X_rows.append(X[pool]);y_rows.append((state[pool]==1).astype(int))
                if policy=='none':chosen=np.empty(0,dtype=int)
                else:
                    if policy=='random':scores=np.zeros(N)
                    elif policy=='degree':scores=degree
                    elif policy=='contact':scores=direct
                    elif policy=='adaptive':
                        risk=1-np.exp(-(0.02+0.65*direct+0.10*two/(1+degree/5)))
                        scores=risk*(1+0.10*degree)
                    elif policy.startswith('opt_'):
                        if policy=='opt_filter':ph=q[:,1]
                        else:
                            if model is None:raise ValueError('Fitted policies require trained risk model')
                            ph=model.predict_proba(X)[:,1]
                        if policy=='opt_cumulative_fitted':
                            exposure=adjacency_hist(pi,pj,cumul,q[:,0])*ROUND_SLOTS/max(k,1)
                        else:
                            exposure=adjacency_hist(pi,pj,recent,q[:,0])*ROUND_SLOTS*(1-RECENCY_DECAY)
                        scores=ph*TEST_SENS*(1-np.exp(-BETA_CONTACT))*exposure
                    else:raise ValueError(policy)
                    ix=np.flatnonzero(eligible)
                    jitter=tie_rng.random(N)
                    chosen=ix[np.lexsort((jitter[ix],-scores[ix]))[:TESTS_PER_ROUND]]
                    last_test[chosen]=k;total_tests+=len(chosen)
                    test_draws=test_rng.random(N)
                    pos=chosen[(state[chosen]==1)&(test_draws[chosen]<TEST_SENS)]
                    positive_tests+=len(pos)
                    state[pos]=2;confirmed[pos]=k
                    filter_tests(q,chosen,pos)
            # Transmission on the actual current temporal contacts, *not* the policy's thinned view.
            infectious=(state==1).astype(float)
            force=BETA_CONTACT*(np.bincount(i,weights=infectious[j],minlength=N)+
                                np.bincount(j,weights=infectious[i],minlength=N))
            newly=(state==0)&(inf_rng.random(N)<-np.expm1(-force))
            recovered=((state==1)|(state==2))&(rec_rng.random(N)<GAMMA_STEP)
            state[recovered]=3
            state[newly]=1;ever[newly]=True;new_infections+=int(newly.sum())
            hidden_person_hours+=((state==1).sum())/12
            obs_ids=ids[vis_mask[ids]]
            filter_predict(q,pi[obs_ids],pj[obs_ids],GAMMA_STEP)
            recent*=RECENCY_DECAY
            recent[obs_ids]+=1.
            cumul[obs_ids]+=1.
        daily_infected.append(int(ever.sum()))
        if day<days-1:
            # No observed nighttime contacts; no modeled transmission, but biological states evolve.
            previous=confirmed>=0
            nd=(state==1)&(diag_rng.random(N)<DELTA_NIGHT)
            state[nd]=2;confirmed[nd]=(day+1)*SLOTS_PER_DAY-1
            filter_diagnosis(q,nd,previous,DELTA_NIGHT)
            rr=((state==1)|(state==2))&(rec_rng.random(N)<GAMMA_NIGHT)
            state[rr]=3
            filter_predict(q,np.empty(0,int),np.empty(0,int),GAMMA_NIGHT)
            recent*=RECENCY_NIGHT
    out=dict(new_infections=new_infections,ever_infected=int(ever.sum()),
             attack_rate=ever.mean(),hidden_person_hours=hidden_person_hours,
             tests=total_tests,positive_tests=positive_tests,
             day3_infected=daily_infected[2],day50_infected=daily_infected[-1])
    if collect:
        out['X']=np.vstack(X_rows);out['y']=np.concatenate(y_rows)
    return out


def fit_model(pi,pj,slot_ids,days=DAYS,train=24,heldout=10,seed=340237):
    dataset={}
    for name,n,stage in [('train',train,710),('heldout',heldout,720)]:
        xx=[];yy=[]
        for rep in range(n):
            sample_rng=rng_for(seed,stage,rep,0)
            starts=sample_rng.choice(N,INITIAL_HIDDEN+INITIAL_CONFIRMED,replace=False)
            keep=sample_rng.random(len(pi))<(.5 if rep%2 else 1.)
            blocks=sampled_day_blocks(seed+stage,rep,days)
            pol=('random' if rep%2 else 'contact')
            obj=run_episode(pi,pj,slot_ids,blocks,keep,starts[:INITIAL_HIDDEN],starts[INITIAL_HIDDEN:],
                            pol,seed=seed+stage,rep=rep,days=days,collect=True)
            xx.append(obj['X']);yy.append(obj['y'])
        dataset[name]=(np.vstack(xx),np.concatenate(yy))
    X,y=dataset['train'];model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=250,solver='lbfgs'))
    model.fit(X,y)
    metrics=[]
    for name,(X,y) in dataset.items():
        for method,p in [('filter',expit(X[:,0])),('fitted',model.predict_proba(X)[:,1])]:
            metrics.append({'split':name,'method':method,'n_rows':len(y),'prevalence':float(y.mean()),
                'brier':brier_score_loss(y,p),'logloss':log_loss(y,p,labels=[0,1]),
                'auc':roc_auc_score(y,p) if len(np.unique(y))>1 else np.nan})
    (ROOT/'models').mkdir(exist_ok=True);(ROOT/'results').mkdir(exist_ok=True)
    joblib.dump(model,ROOT/'models'/'extended_temporal_risk_model.joblib')
    pd.DataFrame(metrics).to_csv(ROOT/'results'/'extended_risk_diagnostics.csv',index=False)
    pd.DataFrame({'feature':FEATURES,'coefficient_standardized':model.named_steps['logisticregression'].coef_.ravel()}).to_csv(ROOT/'results'/'extended_risk_coefficients.csv',index=False)
    print(pd.DataFrame(metrics).to_string(index=False),flush=True)
    return model


def run_evaluation(pi,pj,slot_ids,model,days=DAYS,reps=12,seed=202638,visibilities=(1.,.5),
                   policies=POLICIES):
    rows=[]
    for rep in range(reps):
        rng=rng_for(seed,900,rep,0)
        start=rng.choice(N,INITIAL_HIDDEN+INITIAL_CONFIRMED,replace=False)
        mask_draw=rng.random(len(pi))
        blocks=sampled_day_blocks(seed,rep,days)
        for vis in visibilities:
            for policy in policies:
                val=run_episode(pi,pj,slot_ids,blocks,mask_draw<vis,
                    start[:INITIAL_HIDDEN],start[INITIAL_HIDDEN:],policy,
                    seed=seed,rep=rep,model=model,days=days)
                rows.append({'rep':rep,'visibility':vis,'policy':policy,'days':days,
                             'block_sequence':','.join(str(v+1) for v in blocks),**val})
        print('Completed extended replicate',rep+1,'of',reps,flush=True)
    return pd.DataFrame(rows)


def summarize(df,days):
    r=ROOT/'results';r.mkdir(exist_ok=True)
    df.to_csv(r/'extended_haslemere_replicates.csv',index=False)
    summary=df.groupby(['visibility','policy'],as_index=False).agg(n=('rep','size'),
        attack=('attack_rate','mean'),attack_sd=('attack_rate','std'),
        day3=('day3_infected','mean'),new_infections=('new_infections','mean'),
        tests=('tests','mean'),positive_tests=('positive_tests','mean'))
    summary.to_csv(r/'extended_haslemere_summary.csv',index=False)
    contrasts=[]
    for v in sorted(df.visibility.unique()):
        p=df[df.visibility==v].pivot(index='rep',columns='policy',values='attack_rate')
        for baseline in ['random','contact','adaptive','opt_filter','opt_cumulative_fitted']:
            diff=(p['opt_temporal_fitted']-p[baseline])*100
            mean=float(diff.mean());sd=float(diff.std(ddof=1)) if len(diff)>1 else np.nan
            se=sd/np.sqrt(len(diff))
            contrasts.append({'visibility':v,'contrast':'temporal fitted - '+baseline,'diff_pp':mean,
               'ci_low':mean-1.96*se,'ci_high':mean+1.96*se,'n_reps':len(diff)})
    con=pd.DataFrame(contrasts);con.to_csv(r/'extended_haslemere_contrasts.csv',index=False)
    full=summary[np.isclose(summary.visibility,1)].set_index('policy')
    display=['none','random','degree','contact','adaptive','opt_filter','opt_temporal_fitted','opt_cumulative_fitted']
    names={'none':'No testing','random':'Random','degree':'Degree','contact':'Recent confirmed contacts',
         'adaptive':'Adaptive heuristic','opt_filter':'Mean-field optimizer',
         'opt_temporal_fitted':'Temporal fitted optimizer',
         'opt_cumulative_fitted':'Cumulative-history fitted optimizer'}
    table=['\\begin{tabular}{lrrr}\\toprule','Policy & Ever infected (\\%) & New infections & Tests used \\\\ \\midrule']
    for name in display:
        x=full.loc[name];table.append(f'{names[name]} & {x.attack*100:.2f} & {x.new_infections:.1f} & {x.tests:.1f} \\\\')
    table.append('\\bottomrule\\end{tabular}')
    (r/'extended_policy_table.tex').write_text('\n'.join(table))
    table=['\\begin{tabular}{rrrr}\\toprule','Visible dyads (\\%) & Temporal fitted & Cumulative fitted & Heuristic \\\\ \\midrule']
    for vis in sorted(df.visibility.unique()):
        x=summary[np.isclose(summary.visibility,vis)].set_index('policy').attack
        table.append(f'{int(vis*100)} & {x["opt_temporal_fitted"]*100:.2f} & {x["opt_cumulative_fitted"]*100:.2f} & {x["adaptive"]*100:.2f} \\\\')
    table.append('\\bottomrule\\end{tabular}')
    (r/'extended_visibility_table.tex').write_text('\n'.join(table))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (ROOT/'figures').mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(6.8,4.0))
    for name in ['random','contact','adaptive','opt_temporal_fitted','opt_cumulative_fitted']:
        sub=summary[summary.policy==name].sort_values('visibility')
        ax.plot(sub.visibility*100,sub.attack*100,marker='o',label=names[name])
    ax.set_xlabel('Contact pairs visible to policy (%)')
    ax.set_ylabel(f'Ever infected by day {days} (%)')
    ax.set_title('Extended dynamic Haslemere (day-block resampling)')
    ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout()
    fig.savefig(ROOT/'figures'/'extended_haslemere_visibility.png',dpi=180);plt.close(fig)
    return summary,con


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--csv',default=str(ROOT/'data'/'Kissler_DataS1.csv'))
    ap.add_argument('--days',type=int,default=DAYS)
    ap.add_argument('--train',type=int,default=24)
    ap.add_argument('--heldout',type=int,default=10)
    ap.add_argument('--reps',type=int,default=12)
    ap.add_argument('--skip-fit',action='store_true')
    ap.add_argument('--only-check',action='store_true')
    args=ap.parse_args()
    _,pairs,slot_ids,meta=load_contacts(args.csv)
    pi,pj=pairs.T
    if args.only_check:
        rng=rng_for(202638,900,0,0)
        init=rng.choice(N,INITIAL_HIDDEN+INITIAL_CONFIRMED,replace=False)
        v=run_episode(pi,pj,slot_ids,sampled_day_blocks(202638,0,args.days),
            np.ones(len(pi),bool),init[:INITIAL_HIDDEN],init[INITIAL_HIDDEN:],
            'random',days=args.days)
        print('SMOKE',v);return
    model=joblib.load(ROOT/'models'/'extended_temporal_risk_model.joblib') if args.skip_fit else fit_model(pi,pj,slot_ids,args.days,args.train,args.heldout)
    df=run_evaluation(pi,pj,slot_ids,model,args.days,args.reps)
    print('SUMMARY\n',summarize(df,args.days)[0].to_string(index=False),flush=True)

if __name__=='__main__':main()
