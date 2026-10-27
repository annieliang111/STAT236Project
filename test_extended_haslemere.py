"""Sanity tests for the extended 50-day SHCR/day-block model."""
import numpy as np
from extended_temporal_haslemere import (
  load_contacts,rng_for,sampled_day_blocks,run_episode,N,INITIAL_HIDDEN,INITIAL_CONFIRMED,DAYS)

def main():
    _,pairs,ids,meta=load_contacts('data/Kissler_DataS1.csv')
    assert meta['participants']==469 and meta['close_rows']==17363
    pi,pj=pairs.T
    a=sampled_day_blocks(202638,0,DAYS)
    assert len(a)==DAYS and np.array_equal(a[:3],np.arange(3))
    assert np.all((a[3:]>=0)&(a[3:]<=2))
    rng=rng_for(202638,900,0,0)
    start=rng.choice(N,INITIAL_HIDDEN+INITIAL_CONFIRMED,replace=False)
    full=np.ones(len(pi),bool)
    obs=np.zeros(len(pi),bool)
    x=run_episode(pi,pj,ids,a,full,start[:INITIAL_HIDDEN],start[INITIAL_HIDDEN:],
                  'random',days=6,seed=202638,rep=9)
    y=run_episode(pi,pj,ids,a,obs,start[:INITIAL_HIDDEN],start[INITIAL_HIDDEN:],
                  'random',days=6,seed=202638,rep=9)
    assert x==y, 'Random policy must be invariant to the contact observation mask'
    assert x['tests']==120
    assert x['new_infections']+INITIAL_HIDDEN+INITIAL_CONFIRMED==x['ever_infected']
    z=run_episode(pi,pj,ids,a,full,start[:INITIAL_HIDDEN],start[INITIAL_HIDDEN:],
                  'none',days=6,seed=202638,rep=9)
    assert z['tests']==0
    b=a.copy();b[3:]=2
    c=run_episode(pi,pj,ids,a,full,start[:INITIAL_HIDDEN],start[INITIAL_HIDDEN:],
                  'contact',days=6,seed=202638,rep=9)
    d=run_episode(pi,pj,ids,b,full,start[:INITIAL_HIDDEN],start[INITIAL_HIDDEN:],
                  'contact',days=6,seed=202638,rep=9)
    assert c['day3_infected']==d['day3_infected'], 'Day 4+ contacts cannot affect Day 1-3 results'
    print('Passed: source blocks, 4-state SHCR counts, budget, no-testing, observation-mask and future-block invariance.')

if __name__=='__main__':main()
