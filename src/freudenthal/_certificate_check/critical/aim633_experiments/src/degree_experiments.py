"""Test whether affine-polynomial multiples of lower-degree local potentials
span the next-degree local space. Positive ranks + exact residuals certify
the tested finite-dimensional equality; no induction beyond tested degrees.
"""
from experiment_core import *
from transformations import *
from pathlib import Path
from native_rank import FastRank


def representative_vertices(mesh):
    groups={}
    for v,x in enumerate(mesh.vertices):
        a=tuple(sorted(int(i) for i in x));b=tuple(sorted(mesh.N-int(i) for i in x))
        key=min(a,b);groups.setdefault(key,[]).append(v)
    return [(vs[0],len(vs),key) for key,vs in sorted(groups.items())]


def degree_step(N,d,p=1000003):
    mesh=Mesh.build(N);lo=BernsteinSpace.build(mesh,d);hi=BernsteinSpace.build(mesh,d+1)
    Al,_=lo.curl_constraints();Ah,_=hi.curl_constraints();pl=ordering(lo);ph=ordering(hi)
    maps=[sparse.kron(M,sparse.eye(3,dtype=np.int64),format='csr') for M in degree_multipliers(lo,hi)]
    records=[]
    for v,mult,key in representative_vertices(mesh):
        start=time.time();lc=vector_columns(lo.vertex_nodes(v));hc=vector_columns(hi.vertex_nodes(v))
        KL=restricted_kernel(Al,lc,pl,p);KH=restricted_kernel(Ah,hc,ph,p)
        rr=Ah[:,hc].tocsr();rr=rr[np.diff(rr.indptr)>0];eh=sparse_mod_rank(rr,p,ph[hc])
        free=np.array([j for j in range(len(hc)) if j not in eh.rows],dtype=np.int64)
        span=FastRank(len(free),p,ph[hc[free]])
        stage=[]
        for f,M in zip(['1','x','y','z'],maps):
            product=(M[:,lc]@KL.basis.T).T.tocsr();product.eliminate_zeros()
            residual=Ah@product.T;residual.eliminate_zeros();assert residual.nnz==0
            permitted=set(int(c) for c in hc);assert all(int(c) in permitted for c in product.indices)
            span.add_matrix_rows(product[:,hc[free]].tocsr())
            stage.append(dict(factor=f,rank=span.rank,defect=KH.basis.shape[0]-span.rank))
        records.append(dict(N=N,low_potential_degree=d,high_potential_degree=d+1,vertex=v,coordinate=mesh.vertices[v].tolist(),orbit_size=mult,
                    low_dimension=KL.basis.shape[0],high_dimension=KH.basis.shape[0],affine_product_rank=span.rank,
                    new_generator_defect=KH.basis.shape[0]-span.rank,exact_residual_nnz=0,stages=stage,seconds=time.time()-start))
    return records

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]/'results';outfile=root/'degree_steps.jsonl'
    done=set()
    if outfile.exists():
        for line in outfile.read_text().splitlines():
            r=json.loads(line);done.add((r['N'],r['low_potential_degree']))
    with outfile.open('a') as f:
        for N in (2,3,4):
            for d in (3,4,5,6,7,8):
                if (N,d) in done:continue
                rr=degree_step(N,d)
                for r in rr:f.write(json.dumps(r)+'\n')
                f.flush()
                print(json.dumps(dict(N=N,degree_step=[d,d+1],representative_patches=len(rr),max_defect=max(r['new_generator_defect'] for r in rr),
                                      zero_defect_patches=sum(r['new_generator_defect']==0 for r in rr),seconds=sum(r['seconds'] for r in rr))),flush=True)
