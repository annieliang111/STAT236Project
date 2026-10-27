"""Fast nonclinical regression checks for inference, action eligibility, and reproducibility."""
import numpy as np
import joblib
from scipy import sparse
from simulate_testing import ROOT,generate_sbm,N,T,SEEDS,simulate
from model_based_core import MeanFieldSHCR,simulate_with_filter,optimized_values,risk_features

def run_checks():
    n=7
    f=MeanFieldSHCR(n,.2)
    assert np.allclose(f.q.sum(axis=1),1)
    h0=f.q[:,1].copy()
    diag=np.zeros(n,dtype=bool)
    prev=np.zeros(n,dtype=bool)
    f.diagnosis_update(diag,prev)
    assert np.all(f.q[:,1]<h0)
    ht=f.q[0,1]
    f.test_update(np.array([0]),np.array([],dtype=int))
    assert f.q[0,1]<ht
    f.test_update(np.array([1]),np.array([1]))
    assert np.array_equal(f.q[1],[0,0,1,0])
    g=sparse.csr_matrix(np.ones((n,n))-np.eye(n))
    f.predict(g)
    assert np.allclose(f.q.sum(axis=1),1) and np.isfinite(f.q).all()
    assert (f.q>=0).all() and (f.q<=1).all()
    rng=np.random.default_rng(885)
    adj=sparse.csr_matrix(generate_sbm(rng,4.))
    idx=rng.choice(N,SEEDS,replace=False)
    innovations={k:rng.random((T,N)) for k in ['inf','rec','diag','test','tie']}
    for pol in ['random','degree','contact','adaptive']:
        old=simulate(adj,adj,idx,20,pol,innovations)
        new=simulate_with_filter(adj,adj,idx,20,pol,innovations)
        assert old['cumulative_infections']==new['cumulative_infections']
        assert old['positive_tests']==new['positive_tests']
    model=joblib.load(ROOT/'models'/'risk_model.joblib')
    for pol in ['opt_filter','opt_fitted']:
        a=simulate_with_filter(adj,adj,idx,20,pol,innovations,model)
        assert 0<=a['tests']<=20*(T-8)
        assert 0<=a['positive_tests']<=a['tests']
        assert 0<=a['attack_rate']<=1
    # Each choice uses the value of its own hidden infection risk, not truth.
    assert np.allclose(optimized_values(np.zeros(n),np.ones(n),g),0)
    print('PASS: posterior normalization, Bayesian negative evidence, baseline equivalence, budget, and score.')

if __name__=='__main__': run_checks()
