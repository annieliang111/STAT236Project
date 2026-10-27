"""Fit eligible-person hidden-infection risk from independent simulated trajectories.

Fit and evaluation outbreaks are disjoint by random seed, not merely by row.
Features contain only observations available before tests. Targets are hidden
infection states in the *training simulator*, never exposed in evaluation policies.
"""
import argparse, json
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
import joblib
from simulate_testing import ROOT, N, T, SEEDS, generate_sbm
from model_based_core import simulate_with_filter, FEATURE_NAMES
from run_haslemere import read_haslemere_graph

MODEL_DIR=ROOT/'models'; MODEL_DIR.mkdir(exist_ok=True)
VIS=[.25,.5,.6,.75,1.0]
RATIOS=[1.,1.5,2.,4.,8.,12.]


def sample_episode(rng,rep,haslemere,stage):
    if haslemere is not None and rep%3==0:
        graph=haslemere; population='Haslemere'
    else:
        ratio=RATIOS[rep%len(RATIOS)]
        graph=sparse.csr_matrix(generate_sbm(rng,ratio)); population=f'SBM_r{ratio:g}'
    n=graph.shape[0]
    visibility=VIS[(rep*7+3)%len(VIS)]
    # Same Bernoulli missing-contact model as the evaluation experiments.
    upper=sparse.triu(graph,k=1,format='coo')
    keep=rng.random(len(upper.data))<visibility
    i,j=upper.row[keep],upper.col[keep]
    obs=sparse.csr_matrix((np.ones(2*len(i)),(np.r_[i,j],np.r_[j,i])),shape=graph.shape)
    seeds=rng.choice(n,SEEDS,replace=False)
    innovations={name:rng.random((T,n)) for name in ['inf','rec','diag','test','tie']}
    policy='contact' if rep%2==0 else 'random'
    out=simulate_with_filter(graph,obs,seeds,20,policy,innovations,collect_training=True)
    return out['X'],out['y'],population,visibility


def run(csv,train_reps=90,valid_reps=24,seed=339236):
    haslemere,_=read_haslemere_graph(csv)
    # Train and validation use disjoint SeedSequence prefixes and epidemics.
    arrays={}
    for stage,reps,segment in [('train',train_reps,771),('valid',valid_reps,881)]:
        Xs=[];ys=[];meta=[]
        for rep in range(reps):
            rng=np.random.default_rng(np.random.SeedSequence([seed,segment,rep]))
            X,y,pop,q=sample_episode(rng,rep,haslemere,stage)
            Xs.append(X);ys.append(y);meta.append({'population':pop,'q':q,'rows':len(y)})
            if (rep+1)%30==0:print(stage,rep+1,'/',reps,flush=True)
        arrays[stage]=(np.vstack(Xs),np.concatenate(ys),meta)
    X,y,_=arrays['train']; Xv,yv,_=arrays['valid']
    model=make_pipeline(StandardScaler(), LogisticRegression(C=1.0,solver='lbfgs',max_iter=450))
    model.fit(X,y)
    out=[]
    for label,xs,ys in [('train',X,y),('holdout',Xv,yv)]:
        p=model.predict_proba(xs)[:,1]
        mf=1/(1+np.exp(-np.clip(xs[:,0],-30,30)))
        for name,pred in [('fitted',p),('meanfield',mf),('constant',np.full(len(ys),np.mean(y)))]:
            out.append(dict(split=label,model=name,n_rows=len(ys),prevalence=np.mean(ys),
                            brier=brier_score_loss(ys,pred),logloss=log_loss(ys,pred,labels=[0,1]),
                            auc=roc_auc_score(ys,pred) if np.unique(ys).size==2 else np.nan))
    pd.DataFrame(out).to_csv(ROOT/'results'/'risk_model_diagnostics.csv',index=False)
    estimator=model.named_steps['logisticregression']
    pd.DataFrame({'feature':FEATURE_NAMES,'coef_standardized':estimator.coef_.ravel()}).to_csv(ROOT/'results'/'risk_model_coefficients.csv',index=False)
    joblib.dump(model,MODEL_DIR/'risk_model.joblib')
    with open(MODEL_DIR/'fit_metadata.json','w') as f:
        json.dump({'seed':seed,'train_trajectories':train_reps,'holdout_trajectories':valid_reps,
                   'train_rows':len(y),'holdout_rows':len(yv),
                   'features':FEATURE_NAMES,'policies_in_training':['random','contact'],
                   'comment':'Hidden labels used only for synthetic training/holdout; no observed disease outcomes.'},f,indent=2)
    print(pd.DataFrame(out).to_string(index=False,float_format=lambda v:f'{v:.5f}'),flush=True)
    print('Saved fitted model to',MODEL_DIR/'risk_model.joblib')
    return model

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--csv',default=str(ROOT/'data'/'Kissler_DataS1.csv'))
    parser.add_argument('--train-reps',type=int,default=90)
    parser.add_argument('--valid-reps',type=int,default=24)
    parser.add_argument('--seed',type=int,default=339236)
    args=parser.parse_args()
    run(args.csv,args.train_reps,args.valid_reps,args.seed)
