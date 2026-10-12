"""Standalone entity-star spans, without adding the Bernstein seed family."""
from experiment_core import *
from native_rank import FastRank
from pathlib import Path

def run(N,d):
    sp=BernsteinSpace.build(Mesh.build(N),d);A,_=sp.curl_constraints();pr=ordering(sp);ea=sparse_mod_rank(A,priority=pr)
    free=np.array([j for j in range(A.shape[1]) if j not in ea.rows]);fmap=np.full(A.shape[1],-1,dtype=np.int64);fmap[free]=np.arange(len(free))
    cache={};records=[]
    for dim,label in [(3,'cell'),(2,'face'),(1,'edge'),(0,'vertex')]:
        span=FastRank(len(free),priority=pr[free]);count=0
        for ent,patch in patch_family(sp.mesh,dim):
            key=tuple(sorted(patch))
            if key not in cache:cache[key]=restricted_kernel(A,vector_columns(sp.allowed_nodes(patch)),pr)
            K=cache[key];insert_projected(span,K,fmap);count+=K.basis.shape[0]
        records.append(dict(N=N,potential_degree=d,family=label,span_rank=span.rank,global_dimension=len(free),defect=len(free)-span.rank,generators=count))
    return records
if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]/'results'
    rows=[]
    for N,d in [(2,6),(3,6)]:
        r=run(N,d);rows.extend(r);print(json.dumps(r),flush=True)
    (root/'standalone_entity_spans.json').write_text(json.dumps(rows,indent=2))
