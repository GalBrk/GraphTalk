"""Independent audit of saved 40-node Qwen3 runs. No existing outputs are read.

Run from repository root: python superseded/paper/compare/independent/analyze.py
Graph draws and golds are reconstructed here independently of graphqa.py.
Only the repository's answer extractor is reused, with a post-think sensitivity.
"""
from pathlib import Path
import collections, hashlib, itertools, json, random, re, sys
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from graphtalk import scoring

OUT = Path(__file__).resolve().parent / 'results'
OUT.mkdir(exist_ok=True)
ARMS = ['qwen3-1.7b','qwen3-1.7b-think','qwen3-4b','qwen3-4b-think']
CONDS = ['none','filler','components','clustering','rwse','degree','all']
TASKS = ['node_degree','connected_nodes','edge_existence','edge_count']
SEED = 20260906

def nodes(s):
    return set() if s is None or 'no nodes' in s.lower() else set(map(int,re.findall(r'-?\d+',s)))

def make_graph(p,i):
    seed=SEED+1000000*(1+int(round(p*1000)))+40000+i
    rng=random.Random(seed)
    a=np.zeros((40,40),dtype=np.int64)
    for u,v in itertools.combinations(range(40),2):
        if rng.random()<p: a[u,v]=a[v,u]=1
    d=a.sum(axis=1); m=int(d.sum()//2)
    visited=set(); ncomp=0
    for u in range(40):
        if u in visited: continue
        ncomp+=1; todo=[u]; visited.add(u)
        while todo:
            v=todo.pop()
            for w in np.flatnonzero(a[v]):
                if int(w) not in visited: visited.add(int(w)); todo.append(int(w))
    triangles=np.diag(a@a@a)/2
    c=np.divide(2*triangles,d*(d-1),out=np.zeros(40),where=d>1)
    trans=np.divide(a,d[:,None],out=np.zeros((40,40)),where=d[:,None]>0)
    rw2=np.diag(trans@trans); rw3=np.diag(trans@trans@trans)
    # Match the renderer's documented six-decimal pre-round then two decimals.
    fmt=lambda x: format(round(float(x),6),'.2f')
    rw=[(fmt(x),fmt(y)) for x,y in zip(rw2,rw3)]
    counts=collections.Counter(rw)
    v=random.Random(seed).choice(list(range(40)))
    u,w=random.Random(seed).sample(list(range(40)),2)
    topo=dict(p=p,index=i,graph_id=f'p{p:g}/{i}',n_edges=m,mean_degree=2*m/40,
              components=ncomp,triangles=int(triangles.sum()/3),mean_clustering=c.mean(),
              target=v,target_degree=int(d[v]),target_clustering=c[v],edge_u=u,edge_v=w,
              rwse_unique=len(counts),rwse_modal_share=max(counts.values())/40,
              rwse_unique_4dp=len(set((format(round(float(x),6),'.4f'),format(round(float(y),6),'.4f')) for x,y in zip(rw2,rw3))),
              rwse_unique_6dp=len(set((round(float(x),6),round(float(y),6)) for x,y in zip(rw2,rw3))),
              degree_unique=len(set(d)),clustering_unique=len(set(map(fmt,c))))
    gold={'node_count':'40','edge_count':str(m),'node_degree':str(d[v]),
          'connected_nodes':', '.join(map(str,np.flatnonzero(a[v]))) if d[v] else 'No nodes',
          'edge_existence':'Yes' if a[u,w] else 'No',
          'cycle_check':'Yes' if m>40-ncomp else 'No'}
    return topo,gold

def equal(pred,gold,task):
    if pred is None: return False
    if task=='connected_nodes': return nodes(pred)==nodes(gold)
    if task in ('cycle_check','edge_existence'):
        a=re.search(r'\b(yes|no)\b',pred,re.I); b=re.search(r'\b(yes|no)\b',gold,re.I)
        return bool(a and b and a.group(1).lower()==b.group(1).lower())
    return pred.strip().rstrip('.').lower()==gold.strip().rstrip('.').lower()

def ci_diff(x,y,p,seed=1947,reps=4000):
    delta=np.asarray(y,dtype=float)-np.asarray(x,dtype=float)
    rng=np.random.default_rng(seed)
    draws=np.zeros(reps)
    for density in sorted(set(p)):
        arr=delta[np.asarray(p)==density]
        draws+=arr[rng.integers(0,len(arr),(reps,len(arr)))].sum(axis=1)/len(delta)
    return np.quantile(draws,[.025,.975])*100

def bh(values):
    values=np.asarray(values); order=np.argsort(values); n=len(values)
    adj=np.minimum.accumulate((values[order]*n/np.arange(1,n+1))[::-1])[::-1]
    out=np.empty(n); out[order]=np.minimum(1,adj); return out

def contrast(a,b,metric='success'):
    z=a[['graph_id','p',metric]].merge(b[['graph_id',metric]],on='graph_id',validate='one_to_one',suffixes=('_a','_b'))
    x=z[metric+'_a'].astype(int).to_numpy(); y=z[metric+'_b'].astype(int).to_numpy()
    fixed=int(((x==0)&(y==1)).sum()); broken=int(((x==1)&(y==0)).sum())
    lo,hi=ci_diff(x,y,z.p.to_numpy())
    return dict(n=len(z),baseline=100*x.mean(),treatment=100*y.mean(),delta=100*(y-x).mean(),
                lo=lo,hi=hi,fixed=fixed,broken=broken,
                pvalue=binomtest(fixed,fixed+broken,.5).pvalue if fixed+broken else 1.)

def main():
    graphs={}; rows=[]; files=[]; seen={}; diffs=[]; duplicates=[]
    for arm in ARMS:
        for tag in ['densfull40','densfull40hi']:
            paths=sorted((ROOT/'runs').glob(f'{arm}.{tag}.shard*.jsonl'))
            assert paths,(arm,tag)
            for path in paths:
                payload=path.read_bytes()
                files.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload)))
                for lineno,line in enumerate(payload.splitlines(),1):
                    r=json.loads(line); assert r['model']==arm
                    key=(arm,r['instance_id'],r['condition'],r['style'])
                    signature=hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest()
                    if key in seen:
                        assert seen[key]==signature,('conflicting duplicate',key)
                        duplicates.append(dict(key=key,path=path.relative_to(ROOT).as_posix(),line=lineno))
                        continue
                    seen[key]=signature
                    task,n,p,i=re.fullmatch(r'([^/]+)/size(\d+)/p([\d.]+)/(\d+)',r['instance_id']).groups()
                    assert int(n)==40 and r['style']=='zero_shot' and task==r['task']
                    p=float(p); i=int(i)
                    if (p,i) not in graphs: graphs[p,i]=make_graph(p,i)
                    topo,golds=graphs[p,i]
                    assert equal(str(r['gold']),golds[task],task),('GOLD MISMATCH',key,golds[task],r['gold'])
                    response=r['response'] or ''
                    pred=scoring.extract_answer(response,task)
                    post=response.rsplit('</think>',1)[-1]
                    pred_post=scoring.extract_answer(post,task)
                    cap=bool(r['hit_cap']); exact=equal(pred,golds[task],task)
                    if pred!=pred_post:
                        diffs.append(dict(arm=arm,task=task,condition=r['condition'],instance_id=r['instance_id'],cap=cap,
                                          pred=pred,post=pred_post,response=response))
                    ps=nodes(pred) if task=='connected_nodes' and pred is not None else set()
                    gs=nodes(golds[task]) if task=='connected_nodes' else set()
                    row=dict(arm=arm,tag=tag,task=task,condition=r['condition'],p=p,index=i,
                             graph_id=topo['graph_id'],instance_id=r['instance_id'],gold=golds[task],pred=pred,
                             exact=int(exact),success=int(exact and not cap),cap=int(cap),parsed=int(pred is not None),
                             tokens=r['n_new_tokens'],success_post=int(equal(pred_post,golds[task],task) and not cap),
                             pred_set_size=len(ps) if task=='connected_nodes' and pred is not None else np.nan,
                             missing=len(gs-ps) if task=='connected_nodes' else np.nan,
                             extra=len(ps-gs) if task=='connected_nodes' else np.nan,
                             valid_set=int(ps<=set(range(40))) if task=='connected_nodes' and pred is not None else np.nan,
                             f1=scoring.set_f1(pred,golds[task]) if task=='connected_nodes' else np.nan,
                             source=path.relative_to(ROOT).as_posix(),line=lineno)
                    rows.append(row)
    df=pd.DataFrame(rows); tf=pd.DataFrame([x[0] for x in graphs.values()])
    assert len(df)==84000 and len(tf)==700
    sizes=df.groupby(['arm','task','condition','p']).size(); assert (sizes==100).all()
    df.to_csv(OUT/'responses.csv',index=False); tf.to_csv(OUT/'topology.csv',index=False)
    pd.DataFrame(diffs).to_json(OUT/'extraction_differences.jsonl',orient='records',lines=True)
    summary=df.groupby(['arm','task','condition','p']).agg(n=('success','size'),success=('success','mean'),
        exact=('exact','mean'),cap=('cap','mean'),parsed=('parsed','mean'),tokens=('tokens','median'),f1=('f1','mean')).reset_index()
    summary.to_csv(OUT/'by_density.csv',index=False)
    main_df=df[df.p<=.5].copy()
    pooled=main_df.groupby(['arm','task','condition']).agg(n=('success','size'),success=('success','mean'),
        exact=('exact','mean'),cap=('cap','mean'),f1=('f1','mean')).reset_index()
    pooled.to_csv(OUT/'pooled.csv',index=False)
    effects=[]
    for (arm,task),g in main_df.groupby(['arm','task']):
        for control in ['none','filler']:
            group=[]
            for cond in CONDS:
                if cond==control: continue
                result=contrast(g[g.condition==control],g[g.condition==cond])
                group.append(dict(arm=arm,task=task,control=control,condition=cond,**result))
            for row,q in zip(group,bh([x['pvalue'] for x in group])): row['qvalue']=q
            effects.extend(group)
    pd.DataFrame(effects).to_csv(OUT/'paired_effects.csv',index=False)
    # New joint task probe: identical graph and target, independent questions.
    nd=main_df[main_df.task=='node_degree']; cn=main_df[main_df.task=='connected_nodes']
    joint=nd.merge(cn,on=['arm','condition','graph_id','p','index'],suffixes=('_degree','_neighbors'),validate='one_to_one')
    assert all(int(a)==len(nodes(b)) for a,b in zip(joint.gold_degree,joint.gold_neighbors))
    degree_num=pd.to_numeric(joint.pred_degree,errors='coerce')
    joint['both_complete']=(joint.cap_degree==0)&(joint.cap_neighbors==0)
    joint['consistent']=joint.both_complete&(joint.parsed_degree==1)&(joint.parsed_neighbors==1)&(joint.valid_set_neighbors==1)&(degree_num==joint.pred_set_size_neighbors)
    joint['joint_success']=(joint.success_degree==1)&(joint.success_neighbors==1)
    joint['consistent_wrong']=joint.consistent&~joint.joint_success
    joint.to_csv(OUT/'joint_rows.csv',index=False)
    joint.groupby(['arm','condition']).agg(n=('consistent','size'),consistent=('consistent','mean'),
       joint_success=('joint_success','mean'),consistent_wrong=('consistent_wrong','mean'),both_complete=('both_complete','mean')).reset_index().to_csv(OUT/'consistency.csv',index=False)
    je=[]
    for arm,g in joint.groupby('arm'):
        for metric in ['consistent','joint_success','consistent_wrong']:
            group=[]
            for cond in CONDS[1:]:
                group.append(dict(arm=arm,condition=cond,metric=metric,**contrast(g[g.condition=='none'],g[g.condition==cond],metric)))
            for row,q in zip(group,bh([x['pvalue'] for x in group])): row['qvalue']=q
            je.extend(group)
    pd.DataFrame(je).to_csv(OUT/'joint_effects.csv',index=False)
    ee=df[df.task=='edge_existence']; er=[]
    for (arm,cond,p),g in ee.groupby(['arm','condition','p']):
        yes=g.gold.str.lower().eq('yes'); py=g.pred.fillna('').str.lower().eq('yes')
        tpr=g.loc[yes,'success'].mean(); tnr=g.loc[~yes,'success'].mean()
        er.append(dict(arm=arm,condition=cond,p=p,prevalence=yes.mean(),pred_yes=py.mean(),accuracy=g.success.mean(),
                       tpr=tpr,tnr=tnr,balanced=(tpr+tnr)/2,n_positive=int(yes.sum()),n_negative=int((~yes).sum())))
    pd.DataFrame(er).to_csv(OUT/'edge_balance.csv',index=False)
    caps=df[df.cap==1].groupby(['arm','tag']).tokens.agg(['min','max','count']).reset_index()
    caps.to_csv(OUT/'observed_caps.csv',index=False)
    manifest=dict(source_commit='4b45c2435abe5913a4d28fa19e49b42d9bf62ca0',rows=len(df),graphs=len(tf),files=files,
                  cells=len(sizes),all_gold_verified=True,identical_duplicates=duplicates,extract_differences=len(diffs),
                  uncapped_extract_differences=sum(not x['cap'] for x in diffs),
                  success_changes_after_think=int((df.success!=df.success_post).sum()),
                  python=sys.version,numpy=np.__version__,pandas=pd.__version__)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k!='files'},indent=2))
    print(caps.to_string(index=False))

if __name__=='__main__': main()
