"""Causal three-day Haslemere temporal-network epidemic and screening experiments.

Measured within-4m contacts drive the infection process at every 5-minute slot.
Testing decisions are made at 07:00, 11:00, 15:00, 19:00 daily (5 per round,
20 per day); only preceding contacts/test results enter the decision.
Nighttime has state progression but zero *modeled* transmission: unrecorded
contact is not incorrectly treated as observed absence, and this is a limitation.
All epidemic outcomes are simulated; no empirical infection labels exist.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
import joblib

ROOT=Path(__file__).resolve().parent
N=469; SLOTS_PER_DAY=192; DAYS=3; T=576; SLOT_HOURS=1/12
ROUNDS_PER_DAY=4; ROUND=48; TESTS_PER_ROUND=5
# Per-day progression probabilities for 24-hour time, converted to five minutes.
GAMMA_DAY=.12; DELTA_DAY=.14; TEST_SENS=.90
GAMMA_STEP=1-(1-GAMMA_DAY)**(1/288)
DELTA_STEP=1-(1-DELTA_DAY)**(1/288)
# Contact-specific transmission hazard per observed five-minute encounter.
BETA_CONTACT=.060
INITIAL_HIDDEN=25; INITIAL_CONFIRMED=4
LAMBDA_RECENT=np.exp(-1/24)  # two-hour exponential memory scale (e-folding)
NIGHT_SLOTS=96
CUTOFF=4
VISIBILITIES=(1.0,.75,.5,.25)
POLICIES=('none','random','degree','contact','adaptive','opt_filter','opt_temporal_fitted','opt_cumulative_fitted')
EPS=1e-9


def load_contacts(path):
    """Read Kissler_DataS1.csv (headerless: slot, person 1, person 2, meters)."""
    x=pd.read_csv(path,header=None,names=['step','i','j','distance'])
    for key in x.columns:x[key]=pd.to_numeric(x[key],errors='raise')
    if len(x)!=102831 or int(x.step.min())!=1 or int(x.step.max())!=576:
        raise ValueError('Expected 102831 headerless records with steps 1..576')
    if set(x.i).union(x.j)!=set(range(1,470)):
        raise ValueError('Expected participant IDs 1..469')
    y=x.loc[(x.distance<=CUTOFF)&(x.i!=x.j),['step','i','j']].copy()
    y['step']=y.step.astype(int)-1; y['i']=y.i.astype(int)-1;y['j']=y.j.astype(int)-1
    if y.duplicated(['step','i','j']).any():raise ValueError('Duplicate contact entries')
    by=[([],[]) for _ in range(T)]
    for r in y.itertuples(index=False):
        by[r.step][0].append(r.i); by[r.step][1].append(r.j)
    mats=[]
    for left,right in by:
        a=np.asarray(left,dtype=int);b=np.asarray(right,dtype=int)
        z=sparse.csr_matrix((np.ones(2*len(a)),(np.r_[a,b],np.r_[b,a])),shape=(N,N))
        z.data[:]=1.
        mats.append(z)
    pairs=y[['i','j']].drop_duplicates().to_numpy(dtype=int)
    counts=np.asarray([m.nnz//2 for m in mats],dtype=int)
    edge_id={(int(a),int(b)):k for k,(a,b) in enumerate(pairs)}
    slot_ids=[np.asarray([edge_id[(int(a),int(b))] for a,b in zip(left,right)],dtype=int) for left,right in by]
    return mats,pairs,slot_ids,dict(participants=N,recorded_rows=len(x),close_rows=len(y),unique_pairs=len(pairs),
                           steps=T,mean_pairs_per_step=float(counts.mean()),
                           day_contact_obs=','.join(map(str,[int(sum(counts[d*SLOTS_PER_DAY:(d+1)*SLOTS_PER_DAY])) for d in range(DAYS)])))


def masked_contacts(mats,pairs,visibility,rng):
    # Retain/delete each *pair*, not each repeated occurrence (edge missingness).
    chosen=np.zeros((N,N),dtype=np.float64)
    keep=rng.random(len(pairs))<visibility
    p=pairs[keep]
    chosen[p[:,0],p[:,1]]=1;chosen[p[:,1],p[:,0]]=1
    result=[m.multiply(chosen).tocsr() for m in mats]
    return result


class TemporalFilter:
    """Factorized SHCR Bayesian approximation with conditional diagnoses/tests."""
    def __init__(self,initial_confirmed):
        self.q=np.zeros((N,4),float)
        non=np.ones(N,bool);non[initial_confirmed]=False
        self.q[non,0]=1-INITIAL_HIDDEN/(N-INITIAL_CONFIRMED)
        self.q[non,1]=INITIAL_HIDDEN/(N-INITIAL_CONFIRMED)
        self.q[initial_confirmed,2]=1.

    def diagnosis(self,newly,already,p_d):
        noncase=~already & ~newly
        if noncase.any():
            p=self.q[noncase];z=np.maximum(1-p_d*p[:,1],EPS)
            p[:,0]/=z;p[:,1]*=(1-p_d)/z;p[:,2]/=z;p[:,3]/=z
            self.q[noncase]=p
        self.q[newly]=[0,0,1,0]

    def test(self,tested,positives):
        neg=np.setdiff1d(tested,positives,assume_unique=True)
        if len(neg):
            p=self.q[neg];z=np.maximum(1-TEST_SENS*p[:,1],EPS)
            p[:,0]/=z;p[:,1]*=(1-TEST_SENS)/z;p[:,2]/=z;p[:,3]/=z
            self.q[neg]=p
        if len(positives):self.q[positives]=[0,0,1,0]

    def predict(self,obs_mat,p_recover):
        old=self.q
        # Neighbor independence approximation; log product is exact for independent Bernoulli beliefs.
        log_survive=np.asarray(obs_mat@np.log1p(-(1-np.exp(-BETA_CONTACT))*old[:,1])).ravel()
        pinf=-np.expm1(log_survive)
        q=np.empty_like(old)
        q[:,0]=old[:,0]*(1-pinf)
        q[:,1]=old[:,1]*(1-p_recover)+old[:,0]*pinf
        q[:,2]=old[:,2]*(1-p_recover)
        q[:,3]=old[:,3]+p_recover*(old[:,1]+old[:,2])
        self.q=q/np.maximum(q.sum(axis=1,keepdims=True),EPS)


FEATURES=('filter_logit_h','log_recent_confirmed','log_recent_twohop','log_recent_degree',
          'log_cumulative_degree','filter_susceptible','day_fraction','time_since_test')


def features(flt,recent,cumulative,confirmed,last_test,slot):
    rh=np.clip(flt.q[:,1],1e-5,1-1e-5)
    c=(confirmed>=0).astype(float)
    direct=recent@c
    two=recent@direct
    deg=np.asarray(recent.sum(1)).ravel()
    cumdeg=np.asarray(cumulative.sum(1)).ravel()/(slot+1)
    since=np.where(last_test>=0,slot-last_test,T)
    return np.column_stack((np.log(rh/(1-rh)),np.log1p(direct),
                            np.log1p(two/(1+deg)),np.log1p(deg),
                            np.log1p(cumdeg),flt.q[:,0],
                            np.full(N,slot/T),np.log1p(since)))


def ranked(eligible,score,random_tie,k):
    idx=np.flatnonzero(eligible)
    if not len(idx) or k<=0:return np.empty(0,int)
    return idx[np.lexsort((random_tie[idx],-score[idx]))][:k]


def run_episode(mats,observed,slot_ids,pairs,edge_mask,init_h,init_c,innovation,policy,model=None,collect=False):
    state=np.zeros(N,np.int8)
    state[init_h]=1;state[init_c]=2
    ever=np.zeros(N,bool);ever[init_h]=True;ever[init_c]=True
    confirm=np.full(N,-1,int);confirm[init_c]=0
    last=np.full(N,-1,int)
    flt=TemporalFilter(init_c)
    # Keep dynamic history only on the 1262 actually observed dyads.
    rec_weights=np.zeros(len(pairs),np.float32)
    cum_weights=np.zeros(len(pairs),np.float32)
    pi,pj=pairs[:,0],pairs[:,1]
    observed_ids=[ids[edge_mask[ids]] for ids in slot_ids]
    total_tests=positive_tests=0; hidden_time=0.;new_infections=0
    training_X=[];training_y=[]
    # cumulative uses all observed past contact periods; temporal uses recent exponential weights.
    # At slot t, rec and cumul contain only slots < t.
    infection_curve=[]
    for t in range(T):
        old_confirm=confirm>=0
        new_diag=(state==1)&(innovation['diag'][t]<DELTA_STEP)
        state[new_diag]=2;confirm[new_diag]=t
        flt.diagnosis(new_diag,old_confirm,DELTA_STEP)
        if t%ROUND==0 and (policy!='none' or collect):
            rec=sparse.csr_matrix((np.r_[rec_weights,rec_weights],(np.r_[pi,pj],np.r_[pj,pi])),shape=(N,N))
            cumul=sparse.csr_matrix((np.r_[cum_weights,cum_weights],(np.r_[pi,pj],np.r_[pj,pi])),shape=(N,N))
            eligible=(confirm<0)&((last<0)|(t-last>=SLOTS_PER_DAY))
            if collect:
                z=features(flt,rec,cumul,confirm,last,t)
                training_X.append(z[eligible]);training_y.append((state[eligible]==1).astype(int))
            c=(confirm>=0).astype(float)
            rec_degree=np.asarray(rec.sum(1)).ravel()
            recent_direct=rec@c
            if policy=='random':score=np.zeros(N)
            elif policy=='degree':score=rec_degree
            elif policy=='contact':score=recent_direct
            elif policy=='adaptive':
                two=rec@recent_direct
                risk_proxy=1-np.exp(-(0.02+0.65*recent_direct + .10*two/(1+rec_degree/5)))
                score=risk_proxy*(1+.10*rec_degree)
            elif policy in ('opt_filter','opt_temporal_fitted','opt_cumulative_fitted'):
                if policy=='opt_filter':ph=flt.q[:,1]
                else:
                    if model is None:raise ValueError('Need separately trained temporal risk model')
                    ph=model.predict_proba(features(flt,rec,cumul,confirm,last,t))[:,1]
                # Additive one-round surrogate. Expected forward contact volume based on
                # previously observed intervals: recent exponential mean vs cumulative mean.
                if policy=='opt_cumulative_fitted':
                    # expected contacts in next 4-hour decision block
                    influence=(cumul@flt.q[:,0])*ROUND/max(1,t)
                else:
                    influence=(rec@flt.q[:,0])*ROUND*(1-LAMBDA_RECENT)
                score=ph*TEST_SENS*BETA_CONTACT*influence
            else:raise ValueError(policy)
            tested=ranked(eligible,score,innovation['tie'][t],TESTS_PER_ROUND)
            last[tested]=t;total_tests+=len(tested)
            pos=tested[(state[tested]==1)&(innovation['test'][t,tested]<TEST_SENS)]
            state[pos]=2;confirm[pos]=t;positive_tests+=len(pos)
            flt.test(tested,pos)
        # True temporal edges drive the acquisition process; all contacts after isolation
        # are treated as non-transmitting even though fixed observed movement is reused.
        force=BETA_CONTACT*np.asarray(mats[t]@(state==1).astype(float)).ravel()
        new=(state==0)&(innovation['inf'][t]<-np.expm1(-force))
        recovers=((state==1)|(state==2))&(innovation['rec'][t]<GAMMA_STEP)
        state[recovers]=3
        state[new]=1;ever[new]=True
        new_infections+=int(new.sum())
        hidden_time+=float((state==1).sum())*5/60
        flt.predict(observed[t],GAMMA_STEP)
        # contact observations acquired after decision and the interval's transmission event
        rec_weights*=LAMBDA_RECENT
        rec_weights[observed_ids[t]]+=1.
        cum_weights[observed_ids[t]]+=1.
        infection_curve.append(int(ever.sum()))
        # 8h overnight: progression with no modeled transmission (contacts unavailable).
        if t in (191,383):
            old_confirm=(confirm>=0)
            dxp=1-(1-DELTA_DAY)**(NIGHT_SLOTS/288)
            recp=1-(1-GAMMA_DAY)**(NIGHT_SLOTS/288)
            nd=(state==1)&(innovation['night_diag'][t//192]<dxp)
            state[nd]=2;confirm[nd]=t
            flt.diagnosis(nd,old_confirm,dxp)
            rr=((state==1)|(state==2))&(innovation['night_rec'][t//192]<recp)
            state[rr]=3
            flt.predict(sparse.csr_matrix((N,N)),recp)
            # temporal memory decays through unavailable night hours
            rec_weights*=LAMBDA_RECENT**NIGHT_SLOTS
    ans=dict(new_infections=new_infections,attack_rate=ever.mean(),ever_infected=int(ever.sum()),
             hidden_person_hours=hidden_time,tests=total_tests,positive_tests=positive_tests,
             yield_positive=positive_tests/total_tests if total_tests else np.nan,
             infections_day1=infection_curve[191],infections_day2=infection_curve[383],infections_day3=infection_curve[575])
    if collect:
        ans['X']=np.vstack(training_X);ans['y']=np.concatenate(training_y)
    return ans


def innovations(rng):
    return {s:rng.random((T,N)) for s in ('inf','rec','diag','test','tie')} | {
        'night_diag':rng.random((2,N)),'night_rec':rng.random((2,N))}


def episode_seed(base_seed,stage,rep):
    return np.random.default_rng(np.random.SeedSequence([base_seed,stage,rep]))


def paired_ci(values):
    x=np.asarray(values,float);mu=x.mean();se=x.std(ddof=1)/np.sqrt(len(x))
    return mu,mu-1.96*se,mu+1.96*se


def fit_temporal_model(mats,pairs,slot_ids,train=80,heldout=24,seed=340236):
    dsets={}
    for name,n,marker in [('train',train,810),('heldout',heldout,820)]:
        X=[];y=[]
        for rep in range(n):
            rng=episode_seed(seed,marker,rep)
            vis=VISIBILITIES[rep%len(VISIBILITIES)]
            edge_mask=rng.random(len(pairs))<vis
            mask=np.zeros((N,N),np.float64)
            kept=pairs[edge_mask]
            mask[kept[:,0],kept[:,1]]=1;mask[kept[:,1],kept[:,0]]=1
            obs=[m.multiply(mask).tocsr() for m in mats]
            idx=rng.choice(N,INITIAL_HIDDEN+INITIAL_CONFIRMED,replace=False)
            pol='contact' if rep%2 else 'random'
            out=run_episode(mats,obs,slot_ids,pairs,edge_mask,idx[:INITIAL_HIDDEN],idx[INITIAL_HIDDEN:],innovations(rng),pol,collect=True)
            X.append(out['X']);y.append(out['y'])
        dsets[name]=(np.vstack(X),np.concatenate(y))
    x,y=dsets['train']; xh,yh=dsets['heldout']
    model=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=400,solver='lbfgs'))
    model.fit(x,y)
    rows=[]
    for split,(xx,yy) in dsets.items():
        baseline=np.clip(expit(xx[:,0]),1e-7,1-1e-7)
        preds=model.predict_proba(xx)[:,1]
        for label,p in [('meanfield',baseline),('fitted',preds)]:
            rows.append(dict(split=split,method=label,n=len(yy),prevalence=float(yy.mean()),
                             brier=brier_score_loss(yy,p),logloss=log_loss(yy,p,labels=[0,1]),
                             auc=roc_auc_score(yy,p) if len(np.unique(yy))>1 else np.nan))
    out=ROOT/'results';out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'temporal_risk_diagnostics.csv',index=False)
    pd.DataFrame({'feature':FEATURES,'coefficient_standardized':model.named_steps['logisticregression'].coef_.ravel()}).to_csv(out/'temporal_risk_coefficients.csv',index=False)
    (ROOT/'models').mkdir(exist_ok=True)
    joblib.dump(model,ROOT/'models'/'temporal_risk_model.joblib')
    print('Temporal risk model:',pd.DataFrame(rows).to_string(index=False),flush=True)
    return model


def evaluate(mats,pairs,slot_ids,model,reps=50,seed=250236,start_rep=0):
    rows=[]
    # Paired masks: nested across visibility levels, same epidemic innovations across policies.
    for rep in range(start_rep,start_rep+reps):
        rng=episode_seed(seed,900,rep)
        idx=rng.choice(N,INITIAL_HIDDEN+INITIAL_CONFIRMED,replace=False)
        init_h=idx[:INITIAL_HIDDEN];init_c=idx[INITIAL_HIDDEN:]
        u=innovations(rng)
        # nested missing-pair uniforms across visibility conditions
        edge_u=rng.random(len(pairs))
        for vis in VISIBILITIES:
            keep=pairs[edge_u<vis]
            mask=np.zeros((N,N),np.float64)
            mask[keep[:,0],keep[:,1]]=1;mask[keep[:,1],keep[:,0]]=1
            obs=[m.multiply(mask).tocsr() for m in mats]
            for pol in POLICIES:
                val=run_episode(mats,obs,slot_ids,pairs,edge_u<vis,init_h,init_c,u,pol,model)
                rows.append({'rep':rep,'visibility':vis,'policy':pol,**val})
        if (rep+1)%10==0:print('evaluation',rep+1,'/',reps,flush=True)
    frame=pd.DataFrame(rows)
    summary,contrasts=summarize(frame)
    return frame,summary,contrasts


def summarize(frame):
    out=ROOT/'results';out.mkdir(exist_ok=True)
    frame.to_csv(out/'temporal_haslemere_replicates.csv',index=False)
    summary=frame.groupby(['visibility','policy']).agg(n=('rep','size'),
        attack=('attack_rate','mean'),attack_sd=('attack_rate','std'),
        new_infections=('new_infections','mean'),hidden_person_hours=('hidden_person_hours','mean'),
        tests=('tests','mean'),positive_tests=('positive_tests','mean')).reset_index()
    summary.to_csv(out/'temporal_haslemere_summary.csv',index=False)
    contrasts=[]
    for vis in VISIBILITIES:
        p=frame[frame.visibility==vis].pivot(index='rep',columns='policy',values='attack_rate')
        for ref in ('random','contact','adaptive','opt_filter','opt_cumulative_fitted'):
            vals=(p['opt_temporal_fitted']-p[ref])*100
            mu,lo,hi=paired_ci(vals)
            contrasts.append(dict(visibility=vis,contrast='temporal fitted - '+ref,
                                  diff_pp=mu,ci_low=lo,ci_high=hi,n_reps=len(vals)))
    pd.DataFrame(contrasts).to_csv(out/'temporal_haslemere_contrasts.csv',index=False)
    return summary,pd.DataFrame(contrasts)


def figures_and_latex(summary,contrasts,meta):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fdir=ROOT/'figures';fdir.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(7.4,4.4))
    names={'random':'Random','contact':'Recent positive contact','adaptive':'Adaptive heuristic',
           'opt_filter':'Filtered optimizer','opt_temporal_fitted':'Temporal fitted optimizer',
           'opt_cumulative_fitted':'Cumulative-history optimizer'}
    for p,label in names.items():
        s=summary[summary.policy==p].sort_values('visibility')
        ax.plot(s.visibility*100,s.attack*100,marker='o',label=label)
    ax.set_xlabel('Fraction of contact pairs observed (%)');ax.set_ylabel('Ever infected after 3 recorded days (%)')
    ax.set_title('Three-day temporal Haslemere epidemic | 20 tests/day')
    ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2);fig.tight_layout()
    fig.savefig(fdir/'temporal_haslemere_visibility.png',dpi=170);plt.close(fig)
    s=summary[(summary.visibility==1.)&(summary.policy.isin(names))].copy()
    s=s.set_index('policy').reindex(list(names))
    fig,ax=plt.subplots(figsize=(7.7,4.1));ax.bar(np.arange(len(s)),s.attack.values*100)
    ax.set_xticks(np.arange(len(s)),['Random','Contact','Heuristic','Filter','Temporal fit','Cumulative fit'],rotation=20,ha='right')
    ax.set_ylabel('Ever infected (%)');ax.set_title('Full-observation temporal Haslemere simulation')
    fig.tight_layout();fig.savefig(fdir/'temporal_haslemere_policy.png',dpi=170);plt.close(fig)
    r=ROOT/'results';pd.DataFrame([meta]).to_csv(r/'temporal_network_stats.csv',index=False)
    order=['none','random','contact','adaptive','opt_filter','opt_temporal_fitted','opt_cumulative_fitted']
    s=summary[summary.visibility==1.].set_index('policy').reindex(order)
    lines=['\\begin{tabular}{lrr}\\toprule','Policy & Infected (\\%) & New infections \\\\ \\midrule']
    names2={'none':'No testing','random':'Random','contact':'Contact','adaptive':'Adaptive heuristic',
            'opt_filter':'Mean-field optimizer','opt_temporal_fitted':'Temporal fitted optimizer','opt_cumulative_fitted':'Cumulative-history fitted optimizer'}
    for p,v in s.iterrows():
        lines.append(f'{names2[p]} & {100*v.attack:.2f} & {v.new_infections:.1f} \\\\')
    lines+=['\\bottomrule\\end{tabular}']
    (r/'temporal_policy_table.tex').write_text('\n'.join(lines))
    s=summary[summary.policy.isin(('opt_temporal_fitted','opt_cumulative_fitted','adaptive'))]
    lines=['\\begin{tabular}{rrrr}\\toprule','Observed pairs (\\%) & Temporal optimizer & Cumulative-history optimizer & Heuristic \\\\ \\midrule']
    for vis in sorted(VISIBILITIES):
        v=s[s.visibility==vis].set_index('policy').attack
        lines.append(f'{int(vis*100)} & {100*v["opt_temporal_fitted"]:.2f} & {100*v["opt_cumulative_fitted"]:.2f} & {100*v["adaptive"]:.2f} \\\\')
    lines+=['\\bottomrule\\end{tabular}']
    (r/'temporal_visibility_table.tex').write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--csv',default=str(ROOT/'data'/'Kissler_DataS1.csv'))
    p.add_argument('--train',type=int,default=80)
    p.add_argument('--heldout',type=int,default=24)
    p.add_argument('--reps',type=int,default=50)
    p.add_argument('--skip-fit',action='store_true')
    p.add_argument('--batch-start',type=int,default=-1)
    p.add_argument('--batch-count',type=int,default=3)
    p.add_argument('--aggregate-batches',action='store_true')
    a=p.parse_args()
    mats,pairs,slot_ids,meta=load_contacts(a.csv)
    if a.aggregate_batches:
        from glob import glob
        files=sorted(glob(str(ROOT/'results'/'temporal_batch_*.csv')))
        if not files:raise ValueError('No temporal batch files')
        df=pd.concat([pd.read_csv(f) for f in files],ignore_index=True)
        if df.duplicated(['rep','visibility','policy']).any():raise ValueError('Duplicate replicate/policy')
        s,c=summarize(df)
        figures_and_latex(s,c,meta)
        print('Aggregated',len(df),'rows,',df.rep.nunique(),'replicates from',len(files),'files')
        return
    model=joblib.load(ROOT/'models'/'temporal_risk_model.joblib') if a.skip_fit else fit_temporal_model(mats,pairs,slot_ids,a.train,a.heldout)
    if a.batch_start>=0:
        df,s,c=evaluate(mats,pairs,slot_ids,model,a.batch_count,start_rep=a.batch_start)
        dst=ROOT/'results'/f'temporal_batch_{a.batch_start:03d}.csv'
        df.to_csv(dst,index=False)
        print('Saved',dst,flush=True)
        return
    df,s,c=evaluate(mats,pairs,slot_ids,model,a.reps)
    figures_and_latex(s,c,meta)
    print('Full observation:',s[s.visibility==1][['policy','attack','new_infections']].to_string(index=False))
    print('Paired contrasts:',c[c.visibility==1].to_string(index=False))

if __name__=='__main__':main()
