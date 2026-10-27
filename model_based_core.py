"""Observable-data-only mean-field SHCR filter, calibrated risk features, and myopic allocation.

The simulator's latent epidemic states are used ONLY to simulate outcomes / create
independent training labels, never to select tests. Diagnostic and test results,
contact observations and prior test times are the policy's full information set.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.special import expit
from simulate_testing import T, T0, BETA, GAMMA, DELTA, SENS, COOLDOWN, WINDOW, SEEDS, choose

EPS=1e-7

class MeanFieldSHCR:
    """Factorized filtering approximation, q_i=(P[S],P[H],P[C],P[R]).

    Diagnosis: H->C with rate DELTA, no diagnosis is negative evidence.
    Tests: H->C with sensitivity SENS and perfect specificity; negatives update
    H via Bayes' rule. Prediction uses factorized neighbor infection beliefs.
    This is NOT an exact posterior on the joint 4^n epidemic state space.
    """
    def __init__(self, n, init_prevalence):
        self.q=np.zeros((n,4),dtype=float)
        self.q[:,0]=1-init_prevalence
        self.q[:,1]=init_prevalence

    def diagnosis_update(self, newly_confirmed, already_confirmed):
        eligible=~already_confirmed
        noncase=eligible & (~newly_confirmed)
        p=self.q[noncase]
        z=np.maximum(1-DELTA*p[:,1],EPS)
        p[:,0]/=z
        p[:,1]*=(1-DELTA)/z
        p[:,2]/=z
        p[:,3]/=z
        self.q[noncase]=p
        self.q[newly_confirmed,:]=[0,0,1,0]

    def test_update(self, tested, positive):
        if len(tested)==0:return
        negative=np.setdiff1d(tested,positive,assume_unique=True)
        if len(negative):
            p=self.q[negative]
            z=np.maximum(1-SENS*p[:,1],EPS)
            p[:,0]/=z; p[:,1]*=(1-SENS)/z; p[:,2]/=z; p[:,3]/=z
            self.q[negative]=p
        if len(positive):self.q[positive,:]=[0,0,1,0]

    def predict(self,observed_adj):
        old=self.q.copy()
        pi=1-np.exp(-BETA)
        # Conditional mean-field probability of avoiding infection from neighbors.
        log_survive=observed_adj@np.log1p(-pi*old[:,1])
        pinf=-np.expm1(log_survive)
        q=np.empty_like(old)
        q[:,0]=old[:,0]*(1-pinf)
        q[:,1]=(1-GAMMA)*old[:,1]+old[:,0]*pinf
        q[:,2]=(1-GAMMA)*old[:,2]
        q[:,3]=old[:,3]+GAMMA*(old[:,1]+old[:,2])
        self.q=np.maximum(q,0)
        self.q/=self.q.sum(axis=1)[:,None]


def risk_features(flt, obs, deg, confirmed_day, last_test, d):
    """Observables / inferred beliefs only; no true state or hidden labels."""
    w=np.where((confirmed_day>=0)&((d-confirmed_day)<WINDOW),
               np.exp(-np.maximum(0,d-confirmed_day)/4),0.)
    direct=np.asarray(obs@w).ravel()
    two=np.asarray(obs@direct).ravel()/(1+deg)
    ph=np.clip(flt.q[:,1],1e-5,1-1e-5)
    last_negative=np.clip(d-last_test,0,50)
    x=np.column_stack([np.log(ph/(1-ph)),np.log1p(direct),np.log1p(two),
                       np.log1p(deg),d/50*np.ones(len(deg)),
                       np.log1p(last_negative),
                       flt.q[:,0]])
    return x

FEATURE_NAMES=['filter_logit_H','log1p_recent_contacts','log1p_twohop_per_degree',
               'log1p_observed_degree','day_scaled','log1p_days_since_test',
               'filter_prob_susceptible']


def optimized_values(p_hidden, p_susceptible, obs):
    """One-day *additive surrogate* for expected infections a positive test prevents.

    E_i[prevented] ≈ P_i(H)*sensitivity*sum_j Aobs_ij*P_j(S)*(1-exp(-tau)).
    Ignores multiple-source competition, joint uncertainty, and future days.
    """
    exposure=np.asarray(obs@p_susceptible).ravel()
    return p_hidden*SENS*(1-np.exp(-BETA))*exposure


def simulate_with_filter(true_adj, obs_adj, seed_idx, budget, policy, innovations,
                         risk_model=None, collect_training=False):
    """Same event order as simulate_testing.simulate; uses no oracle latent states for ranking.

    Supported policies: original four baselines, opt_filter, opt_fitted.
    collect_training logs eligible individual features and true H labels solely
    for fitting on separately seeded epidemic trajectories.
    """
    n=true_adj.shape[0]
    state=np.zeros(n,dtype=np.int8)
    state[seed_idx]=1
    ever_inf=np.zeros(n,dtype=bool); ever_inf[seed_idx]=True
    confirmed_day=np.full(n,-1,dtype=np.int32)
    last_test=np.full(n,-999,dtype=np.int32)
    deg=np.asarray(obs_adj.sum(axis=1)).ravel()
    filt=MeanFieldSHCR(n, SEEDS/n)
    total_tests=positive_tests=hidden_days=isolation_days=0
    xs=[];ys=[];days=[]
    if policy=='opt_fitted':
        scaler=risk_model.named_steps['standardscaler']
        lr=risk_model.named_steps['logisticregression']
        coeff=lr.coef_.ravel()/scaler.scale_
        intercept=float(lr.intercept_[0]-scaler.mean_@coeff)
    for d in range(T):
        spontaneous=(state==1)&(innovations['diag'][d]<DELTA)
        previously=(confirmed_day>=0)
        state[spontaneous]=2
        confirmed_day[spontaneous]=d
        filt.diagnosis_update(spontaneous,previously)
        elig=(confirmed_day<0)&((d-last_test)>=COOLDOWN)
        tested=np.empty(0,dtype=np.int64)
        if d>=T0 and budget>0:
            if collect_training:
                f=risk_features(filt,obs_adj,deg,confirmed_day,last_test,d)
                # Labels only enter the independent training dataset, never policy input.
                xs.append(f[elig]);ys.append((state[elig]==1).astype(int));days.append(d)
            if policy in ('opt_filter','opt_fitted'):
                if policy=='opt_filter':
                    pred_h=filt.q[:,1].copy()
                    pred_s=filt.q[:,0].copy()
                else:
                    if risk_model is None:raise ValueError('opt_fitted requires trained model')
                    features=risk_features(filt,obs_adj,deg,confirmed_day,last_test,d)
                    pred_h=expit(features@coeff+intercept)
                    # Calibrated P(H) and mean-field P(S) retain different roles.
                    pred_s=filt.q[:,0].copy()
                v=optimized_values(pred_h,pred_s,obs_adj)
                idx=np.flatnonzero(elig)
                tested=idx[np.lexsort((innovations['tie'][d,idx],-v[idx]))][:budget]
            else:
                tested=choose(policy,budget,elig,obs_adj,deg,confirmed_day,d,innovations['tie'][d])
            last_test[tested]=d;total_tests+=len(tested)
            positives=tested[(state[tested]==1)&(innovations['test'][d,tested]<SENS)]
            state[positives]=2
            confirmed_day[positives]=d
            positive_tests+=len(positives)
            filt.test_update(tested,positives)
        force=BETA*np.asarray(true_adj@(state==1).astype(float)).ravel()
        new=(state==0)&(innovations['inf'][d]<-np.expm1(-force))
        recover=((state==1)|(state==2))&(innovations['rec'][d]<GAMMA)
        state[recover]=3;state[new]=1;ever_inf[new]=True
        filt.predict(obs_adj)
        hidden_days+=int(np.sum(state==1));isolation_days+=int(np.sum(state==2))
    result=dict(attack_rate=float(ever_inf.mean()),cumulative_infections=int(ever_inf.sum()),
                hidden_person_days=hidden_days,isolation_person_days=isolation_days,
                tests=total_tests,positive_tests=positive_tests,
                test_yield=positive_tests/total_tests if total_tests else np.nan)
    if collect_training:
        result['X']=np.vstack(xs) if xs else np.empty((0,len(FEATURE_NAMES)))
        result['y']=np.concatenate(ys) if ys else np.empty(0,dtype=int)
    return result
