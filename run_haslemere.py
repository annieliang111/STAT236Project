"""Empirical-contact-network extension to STAT 236 screening study.

Uses the REAL 2017 Haslemere proximity measurements as graph input, and simulates
H/C/R infection states and interventions. Does NOT use observed COVID infections.

Download source yourself once or invoke --download on a network-connected computer:
  python run_haslemere.py --download --reps 50
  python run_haslemere.py --csv data/HaslemereProximity.csv --reps 50

Licensed original data are not redistributed with this project. Source:
https://github.com/skissler/haslemere (Kissler_DataS1.csv) and
https://github.com/skissler/HaslemereDataset (HaslemereProximity.csv).
"""
import argparse
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulate_testing import simulate, ROOT, T, SEEDS, POLICIES

DATA_DIR = ROOT / 'data'
DATA_DIR.mkdir(exist_ok=True)
RESULTS=ROOT/'results'; FIGS=ROOT/'figures'
URL='https://raw.githubusercontent.com/skissler/HaslemereDataset/main/data/HaslemereProximity.csv'


def obtain_data(path, download=False):
    path=Path(path)
    if not path.exists():
        if not download:
            raise FileNotFoundError(
                f'Haslemere data missing at {path}. Download from {URL} and save there, '
                'or run with --download on a network-connected machine. No substitute graph is used.')
        print('Downloading public source:',URL,flush=True)
        path.parent.mkdir(parents=True,exist_ok=True)
        try:
            urllib.request.urlretrieve(URL,str(path))
        except Exception as e:
            raise RuntimeError('Public data download failed. Download manually from: '+URL) from e
    return path


def read_haslemere_graph(path, threshold=4.0, expected_n=469):
    """Static binary union of <=threshold-meter encounters over all 3 recorded days.

    Data columns: time_step, user1_id, user2_id, distance_m.
    Includes persons with no qualifying contact among IDs 1,...,469.
    Observational content is exclusively the contact topology; infections simulated.
    """
    # Kissler_DataS1.csv is headerless; HaslemereProximity.csv uses column names.
    # Both contain the same original proximity observations in four-column form.
    cols=['time_step','user1_id','user2_id','distance_m']
    with open(path, encoding='utf-8-sig') as fh:
        first=fh.readline().strip()
    header=(first.lower().startswith('time_step,'))
    d=pd.read_csv(path) if header else pd.read_csv(path,header=None,names=cols)
    req=set(cols)
    if not req.issubset(d.columns):
        raise ValueError(f'Expected columns {req}, got {list(d.columns)}')
    for c in req:
        d[c]=pd.to_numeric(d[c],errors='raise')
    ids=np.concatenate([d.user1_id.values,d.user2_id.values])
    if len(d)!=102831 or not (ids.min()==1 and ids.max()==expected_n):
        raise ValueError('Unexpected record count or participant ID range. Official CSV has 102831 observations and IDs 1..469; check the file and source version.')
    if not (d.time_step.min()==1 and d.time_step.max()==576 and d.time_step.nunique()==576):
        raise ValueError('Data must contain all 576 recorded five-minute intervals (three days).')
    sub=d[(d.distance_m<=threshold)&(d.distance_m>=0)&(d.user1_id!=d.user2_id)].copy()
    source=np.minimum(sub.user1_id.to_numpy(dtype=int),sub.user2_id.to_numpy(dtype=int))-1
    target=np.maximum(sub.user1_id.to_numpy(dtype=int),sub.user2_id.to_numpy(dtype=int))-1
    pairs=np.unique(np.stack([source,target],axis=1),axis=0)
    i,j=pairs[:,0],pairs[:,1]
    matrix=sparse.csr_matrix((np.ones(2*len(i),dtype=np.float64),(np.r_[i,j],np.r_[j,i])),shape=(expected_n,expected_n))
    matrix.data[:]=1.0
    matrix.eliminate_zeros()
    deg=np.asarray(matrix.sum(axis=1)).ravel()
    metadata={'data_rows':len(d),'time_points':int(d.time_step.nunique()),'nodes':expected_n,
              'contact_threshold_m':threshold, 'unique_edges':int(len(i)),
              'mean_degree':float(deg.mean()),'median_degree':float(np.median(deg)),
              'isolates':int(np.sum(deg==0)),
              'largest_component':int(max(np.bincount(sparse.csgraph.connected_components(matrix,directed=False,return_labels=True)[1])))}
    return matrix,metadata


def observed_adj(a,rng,visibility):
    if visibility==1.0:
        return a.copy()
    upper=sparse.triu(a,k=1,format='coo')
    retain=rng.random(len(upper.data))<visibility
    i,j=upper.row[retain],upper.col[retain]
    return sparse.csr_matrix((np.ones(2*len(i)),(np.r_[i,j],np.r_[j,i])),shape=a.shape)


def run(reps=50,seed=2362026,csv=None,download=False):
    path=obtain_data(csv or DATA_DIR/'HaslemereProximity.csv',download=download)
    a,metadata=read_haslemere_graph(path)
    print('Observed Haslemere network after source-format validation:',metadata,flush=True)
    pd.DataFrame([metadata]).to_csv(RESULTS/'haslemere_graph_metrics.csv',index=False)
    rows=[]
    n=a.shape[0]
    for rep in range(reps):
        rng=np.random.default_rng(np.random.SeedSequence([seed,90,rep]))
        seeds=rng.choice(n,SEEDS,replace=False)
        innovations={key:rng.random((T,n)) for key in ['inf','rec','diag','test','tie']}
        for visibility in [1.,.60]:
            obs=observed_adj(a,rng,visibility)
            for budget in [0,8,20,52]:
                policies=['none'] if budget==0 else POLICIES
                for pol in policies:
                    x=simulate(a,obs,seeds,budget,pol,innovations)
                    row={k:x[k] for k in ['attack_rate','cumulative_infections',
                       'hidden_person_days','isolation_person_days','tests','positive_tests','test_yield']}
                    row.update({'network':'Haslemere','rep':rep,'visibility':visibility,
                                'budget':budget,'policy':pol,'nodes':n})
                    rows.append(row)
        if (rep+1)%10==0: print('Finished replicate',rep+1,flush=True)
    raw=pd.DataFrame(rows)
    raw.to_csv(RESULTS/'haslemere_replicates.csv',index=False)
    summary=raw.groupby(['network','visibility','budget','policy'],sort=False).agg(
        n_replicates=('attack_rate','size'),mean_attack=('attack_rate','mean'),
        se_attack=('attack_rate',lambda x: x.std(ddof=1)/np.sqrt(len(x))),
        mean_hidden=('hidden_person_days','mean'),mean_tests=('tests','mean'),
        mean_yield=('test_yield','mean'),mean_isolation=('isolation_person_days','mean')).reset_index()
    summary.to_csv(RESULTS/'haslemere_summary.csv',index=False)
    pairs=[]
    for vis in [1.,.60]:
        for budget in [8,20,52]:
            sub=raw[(raw.visibility==vis)&(raw.budget==budget)]
            pv=sub.pivot(index='rep',columns='policy',values='attack_rate')
            for comp in ['random','contact','degree']:
                vals=pv['adaptive']-pv[comp]
                se=vals.std(ddof=1)/np.sqrt(len(vals))
                pairs.append({'network':'Haslemere','visibility':vis,'budget':budget,
                              'contrast':'adaptive - '+comp,'mean_difference':vals.mean(),
                              'ci_lower':vals.mean()-1.96*se,'ci_upper':vals.mean()+1.96*se,
                              'n_replicates':len(vals)})
    paired=pd.DataFrame(pairs)
    paired.to_csv(RESULTS/'haslemere_paired.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,4.4))
    subs=summary[(summary.visibility==1)&(summary.budget==20)&(summary.policy!='none')]
    names=['random','degree','contact','adaptive']
    ss=subs.set_index('policy').loc[names]
    ax.bar(names,ss.mean_attack,yerr=1.96*ss.se_attack,capsize=4)
    ax.set_ylim(0, min(1.,max(.4,float((ss.mean_attack+1.96*ss.se_attack).max())*1.35)))
    ax.set_ylabel('Final infected fraction')
    ax.set_title('Haslemere contact graph | 20 tests/day | 100% graph observed')
    fig.tight_layout();fig.savefig(FIGS/'haslemere_policy.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.7,4.3))
    for pol in POLICIES:
        sub=summary[(summary.visibility==1)&(summary.policy==pol)].sort_values('budget')
        ax.plot(sub.budget,sub.mean_attack,marker='o',label=pol.title())
    baseline=summary[(summary.visibility==1)&(summary.budget==0)]
    ax.scatter([0],baseline.mean_attack,label='No testing',marker='x',s=70)
    ax.set_xticks([0,8,20,52]);ax.set_ylim(0,1);ax.set_xlabel('Tests per day')
    ax.set_ylabel('Final infected fraction');ax.legend(ncol=2)
    ax.set_title('Haslemere contact network | full observation')
    fig.tight_layout();fig.savefig(FIGS/'haslemere_budget.png',dpi=180);plt.close(fig)
    print('Saved empirical-network results; infection outcomes remain simulated.',flush=True)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--csv',default=None,help='Path to either original Kissler_DataS1.csv (no header) or HaslemereProximity.csv (header)')
    p.add_argument('--download',action='store_true',help='Download official CSV from public GitHub on your machine')
    p.add_argument('--reps',type=int,default=50)
    p.add_argument('--seed',type=int,default=2362026)
    args=p.parse_args()
    run(args.reps,args.seed,args.csv,args.download)
