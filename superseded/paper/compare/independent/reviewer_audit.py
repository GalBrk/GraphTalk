"""Reproduce reviewer samples and a narrowly scoped manual sensitivity audit.

Does not change repository scoring. Review established that all but one of the
24 uncapped singleton extractions with multi-node golds are extraction failures.
The correction is restricted to those manually inspected records. It is NOT a
general-purpose new scorer or an exhaustive extraction validation.
"""
from pathlib import Path
import json, re
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent / 'results'
r = pd.read_csv(OUT/'responses.csv')
cache = {}
def raw(row):
    path = row.source
    if path not in cache:
        cache[path] = (ROOT/path).read_text().splitlines()
    return json.loads(cache[path][int(row.line)-1])
def nodes(x):
    return set() if pd.isna(x) or x == 'No nodes' else set(map(int,re.findall(r'\d+',x)))

# One independently inspected uncapped neighborhood output per arm/condition.
sample = r[(r.task=='connected_nodes') & (r.cap==0)].groupby(
    ['arm','condition'],group_keys=False).sample(n=1,random_state=472).copy()
sample['response'] = [raw(x)['response'] for _,x in sample.iterrows()]
sample.to_json(OUT/'reviewer_sample28.jsonl',orient='records',lines=True)

# Expanded targeted inspection; each was visually reviewed before this script.
x = r[(r.task=='connected_nodes') & (r.cap==0) &
      (r.pred_set_size<=1) & r.gold.str.contains(',')].copy()
assert len(x)==24
records=[]
for index,a in x.iterrows():
    response=raw(a)['response']; post=response.rsplit('</think>',1)[-1]
    true_singleton=(a.arm=='qwen3-4b-think' and a.condition=='components' and
                    a.instance_id=='connected_nodes/size40/p0.1/21')
    if true_singleton:
        corrected='5'
    else:
        normalized=re.sub(r'\bnode\s+', '',post,flags=re.I).replace('**','')
        lists=re.findall(r'\d+(?:\s*(?:,\s*and\s+|,\s*|\s+and\s+)\d+)+',normalized)
        # The longest list corresponds to the stated final answer in these
        # reviewed cases. No gold data selects the list or its contents.
        corrected=max(lists,key=lambda z:len(re.findall(r'\d+',z)))
    old_set=nodes(a.pred); new_set=nodes(corrected); gold_set=nodes(a.gold)
    rec=dict(arm=a.arm,condition=a.condition,instance_id=a.instance_id,
             source=a.source,line=int(a.line),pred=a.pred,reviewed_pred=corrected,
             gold=a.gold,extraction_error=not true_singleton,
             old_success=old_set==gold_set,new_success=new_set==gold_set,
             response=response)
    records.append(rec)
    r.loc[index,'pred']=corrected
    r.loc[index,'pred_set_size']=len(new_set)
    r.loc[index,'success']=int(new_set==gold_set)
pd.DataFrame(records).to_json(OUT/'reviewer_singleton24.jsonl',orient='records',lines=True)

nd=r[(r.p<=.5)&(r.task=='node_degree')]
cn=r[(r.p<=.5)&(r.task=='connected_nodes')]
j=nd.merge(cn,on=['arm','condition','graph_id','p','index'],suffixes=('_degree','_neighbors'),validate='one_to_one')
j['consistent']=(j.cap_degree==0)&(j.cap_neighbors==0)&(j.parsed_degree==1)&(j.parsed_neighbors==1)&(j.valid_set_neighbors==1)&(pd.to_numeric(j.pred_degree,errors='coerce')==j.pred_set_size_neighbors)
j['joint_success']=(j.success_degree==1)&(j.success_neighbors==1)
j['consistent_wrong']=j.consistent & ~j.joint_success
summary=j.groupby(['arm','condition']).agg(consistent=('consistent','mean'),joint_success=('joint_success','mean'),consistent_wrong=('consistent_wrong','mean')).reset_index()
old=pd.read_csv(OUT/'consistency.csv')
comparison=old.merge(summary,on=['arm','condition'],suffixes=('_primary','_reviewed'))
for col in ['consistent','joint_success','consistent_wrong']:
    comparison[col+'_change_pp']=100*(comparison[col+'_reviewed']-comparison[col+'_primary'])
comparison.to_csv(OUT/'reviewer_consistency_sensitivity.csv',index=False)
print(pd.DataFrame(records).groupby(['arm','condition']).agg(n=('extraction_error','size'),errors=('extraction_error','sum'),old=('old_success','sum'),reviewed=('new_success','sum')).to_string())
print(comparison[comparison.consistent_change_pp.abs()+comparison.joint_success_change_pp.abs()>0].to_string(index=False))
