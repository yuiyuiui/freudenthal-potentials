"""Test a global basis candidate built from degree-five local potentials,
affine multipliers, and (only if needed) the three degree-six interior templates.
"""
from experiment_core import *
from transformations import *
from native_rank import FastRank
from pathlib import Path


def template_at_vertex(sp,v,data):
    xyz=data['vertices'];tets=data['tetrahedra'];coef=data['coefficients']
    lookup={tuple(xyz[t].ravel()):j for j,t in enumerate(tets)};rows=[{} for _ in range(3)]
    for t in sp.mesh.stars[v]:
        key=tuple((sp.mesh.vertices[sp.mesh.tets[t]]-sp.mesh.vertices[v]).ravel());j=lookup[key]
        for a,g in enumerate(sp.local_to_global[t]):
            for comp in range(3):
                c=3*int(g)+comp
                for i in range(3):
                    z=int(coef[i,j,a,comp])
                    if c in rows[i]:assert rows[i][c]==z
                    rows[i][c]=z
    return coo_from_rows(rows,3*sp.nscalar)


def run(N):
    start=time.time();mesh=Mesh.build(N);lo=BernsteinSpace.build(mesh,5);hi=BernsteinSpace.build(mesh,6)
    Al,_=lo.curl_constraints();Ah,_=hi.curl_constraints();pl=ordering(lo);ph=ordering(hi);eh=sparse_mod_rank(Ah,priority=ph)
    free=np.array([j for j in range(Ah.shape[1]) if j not in eh.rows]);span=FastRank(len(free),priority=ph[free]);maps=[sparse.kron(M,sparse.eye(3,dtype=np.int64),format='csr') for M in degree_multipliers(lo,hi)]
    vertices=sorted(range(len(mesh.vertices)),key=lambda v:(sum(c in (0,N) for c in mesh.vertices[v]),tuple(mesh.vertices[v])))
    checkpoints=[]
    for v in vertices:
        cols=vector_columns(lo.vertex_nodes(v));K=restricted_kernel(Al,cols,pl);before=span.rank
        for j,M in enumerate(maps):
            centered=M if j==0 else M-int(mesh.vertices[v,j-1])*maps[0]
            B=(centered[:,cols]@K.basis.T).T.tocsr();B.eliminate_zeros()
            R=Ah@B.T;R.eliminate_zeros();assert R.nnz==0
            span.add_matrix_rows(B[:,free].tocsr())
        checkpoints.append(dict(vertex=int(v),coordinate=mesh.vertices[v].tolist(),degree5_dimension=K.basis.shape[0],rank_gain=span.rank-before,rank=span.rank))
    unenhanced=span.rank;root=Path(__file__).resolve().parents[1]/'results';data=np.load(root/'interior_star_degree6'/'three_gradient_primitive_templates.npz')
    enrichment=[]
    for v in vertices:
        if all(2<=int(c)<=N-2 for c in mesh.vertices[v]):
            B=template_at_vertex(hi,v,data);R=Ah@B.T;R.eliminate_zeros();assert R.nnz==0
            before=span.rank;span.add_matrix_rows(B[:,free].tocsr());enrichment.append(dict(vertex=int(v),rank_gain=span.rank-before))
    res=dict(N=N,potential_degree=6,global_dimension=len(free),affine_degree5_seed_rank=unenhanced,affine_degree5_seed_defect=len(free)-unenhanced,
             enriched_rank=span.rank,enriched_defect=len(free)-span.rank,exact_residual_nnz=0,seconds=time.time()-start,
             checkpoints=checkpoints,enrichment=enrichment)
    (root/f'global_polynomial_seeds_N{N}_d6.json').write_text(json.dumps(res,indent=2));return res
if __name__=='__main__':
    for N in [4]:
        r=run(N);print(json.dumps({k:v for k,v in r.items() if k not in ['checkpoints','enrichment']}),flush=True)
