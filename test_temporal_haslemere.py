"""Fast temporal contact, posterior, reproducibility, and leakage checks."""
import numpy as np
from scipy import sparse
from temporal_haslemere import (load_contacts,TemporalFilter,run_episode,innovations,
    INITIAL_HIDDEN,INITIAL_CONFIRMED,N,GAMMA_STEP)


def run(csv='data/Kissler_DataS1.csv'):
    mats,pairs,ids,meta=load_contacts(csv)
    assert len(mats)==576 and meta['participants']==469 and meta['close_rows']==17363
    assert sum(len(x) for x in ids)==17363
    f=TemporalFilter(np.array([0,1,2,3]))
    f.q[10]=[.5,.5,0,0]
    f.test(np.array([10]),np.array([],int))
    assert f.q[10,1]<.5
    for i in range(5):f.predict(mats[i],GAMMA_STEP)
    assert np.allclose(f.q.sum(axis=1),1)
    rng=np.random.default_rng(33)
    chosen=rng.choice(N,INITIAL_HIDDEN+INITIAL_CONFIRMED,replace=False)
    inn=innovations(rng)
    all_eligible=np.ones(len(pairs),bool)
    result=run_episode(mats,mats,ids,pairs,all_eligible,chosen[:INITIAL_HIDDEN],chosen[INITIAL_HIDDEN:],inn,'random')
    assert result['tests']==60
    assert result['ever_infected']==INITIAL_HIDDEN+INITIAL_CONFIRMED+result['new_infections']
    assert result['infections_day1']<=result['infections_day2']<=result['infections_day3']
    assert result['infections_day3']==result['ever_infected']
    m=np.zeros((N,N),float)
    blank=[sparse.csr_matrix((N,N)) for _ in mats]
    r2=run_episode(mats,blank,ids,pairs,np.zeros(len(pairs),bool),chosen[:INITIAL_HIDDEN],chosen[INITIAL_HIDDEN:],inn,'random')
    # Random allocation uses no contact information; missing edges must not alter it.
    for x in ('new_infections','ever_infected','positive_tests','tests'):
        assert result[x]==r2[x],(x,result[x],r2[x])
    print('Temporal checks passed:',meta)

if __name__=='__main__':run()
