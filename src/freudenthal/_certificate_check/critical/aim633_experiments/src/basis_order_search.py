"""Search global coefficient-elimination bases for locality and coefficient size.
These are alternative coordinate bases, not new candidate subspaces.
"""
from experiment_core import *
from pathlib import Path
from itertools import permutations


def inspect(sp,A,perm,reverse,component_reverse,row_reverse,seed=None):
    start=time.time();n=A.shape[1];rng=np.random.default_rng(seed or 0)
    if seed is None:
        key=lambda c:(-len(sp.node_entity[c//3]),tuple((-1 if reverse else 1)*int(sp.nodes[c//3,j]) for j in perm),(-1 if component_reverse else 1)*(c%3))
    else:
        randomkeys=rng.random(n);key=lambda c:(-len(sp.node_entity[c//3]),randomkeys[c])
    order=sorted(range(n),key=key);pr=np.argsort(order)
    mat=A[::-1].tocsr() if row_reverse else A
    e=sparse_mod_rank(mat,priority=pr);mod=e.nullspace_rows();integer,info=lift_rows(mod,1000003);B=coo_from_rows(integer,n)
    residual=A@B.T;residual.eliminate_zeros();assert residual.nnz==0
    local=0;support_sizes=[];nonlocal_entity=[];free=[j for j in range(n) if j not in e.rows]
    for f,row in zip(free,integer):
        owners=None;active=set()
        for j in row:
            own=sp.node_owner_vertices[j//3];owners=own.copy() if owners is None else owners&own
            active.update(sp.node_tets[j//3])
        local+=bool(owners);support_sizes.append(len(active))
        if not owners:nonlocal_entity.append(len(sp.node_entity[f//3])-1)
    return dict(N=sp.mesh.N,potential_degree=sp.d,axis_order=list(perm),reverse_coordinates=reverse,reverse_components=component_reverse,
                reverse_rows=row_reverse,random_seed=seed,dimension=B.shape[0],local_vectors=local,nonlocal_vectors=B.shape[0]-local,
                max_tetrahedra_support=max(support_sizes),median_tetrahedra_support=float(np.median(support_sizes)),
                max_vector_nnz=max(np.diff(B.indptr)),total_basis_nnz=B.nnz,exact_residual_nnz=0,seconds=time.time()-start,**info)

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]/'results';out=[];sp=BernsteinSpace.build(Mesh.build(2),6);A,_=sp.curl_constraints()
    configs=[(perm,rev,False,rr,None) for perm in permutations(range(3)) for rev in [False,True] for rr in [False,True]]
    configs += [((0,1,2),False,False,False,s) for s in range(1,7)]
    with (root/'basis_order_search.jsonl').open('w') as f:
        for c in configs:
            r=inspect(sp,A,*c);out.append(r);line=json.dumps(r,default=int);f.write(line+'\n');f.flush();print(line,flush=True)
        best=sorted(out,key=lambda r:(r['nonlocal_vectors'],r['max_abs_integer_coefficient'],r['total_basis_nnz']))[:3]
        sp=BernsteinSpace.build(Mesh.build(3),6);A,_=sp.curl_constraints()
        for b in best:
            r=inspect(sp,A,b['axis_order'],b['reverse_coordinates'],b['reverse_components'],b['reverse_rows'],b['random_seed'])
            line=json.dumps(r,default=int);f.write(line+'\n');f.flush();print(line,flush=True)
