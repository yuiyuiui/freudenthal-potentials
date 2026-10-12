"""Compare genuinely different candidate subspaces, not just equivalent bases."""
from experiment_core import *
from transformations import *
from pathlib import Path
from native_rank import FastRank


def add_full_rows(span,B,free):
    old=span.rank;P=B[:,free].tocsr();span.add_matrix_rows(P);return span.rank-old


def benchmark(N,d,p=1000003):
    start=time.time();mesh=Mesh.build(N);sp=BernsteinSpace.build(mesh,d);A,_=sp.curl_constraints();pr=ordering(sp)
    glob=sparse_mod_rank(A,p,pr);free=np.array([j for j in range(A.shape[1]) if j not in glob.rows],dtype=np.int64)
    fmap=np.full(A.shape[1],-1,dtype=np.int64);fmap[free]=np.arange(len(free))
    def newspan():return FastRank(len(free),p,pr[free])
    stages=[];span=newspan();cache={};patch_checkpoints=[]
    zero=np.flatnonzero(np.asarray(abs(A).sum(axis=0)).ravel()==0)
    for c in zero:span.add({int(fmap[c]):1})
    stages.append(dict(family='individually_admissible_Bernstein_vectors',span_rank=span.rank,defect=len(free)-span.rank,candidate_count=len(zero)))
    for dim,label in [(3,'cell_stars'),(2,'face_stars'),(1,'edge_stars'),(0,'vertex_stars')]:
        before=span.rank;count=0;calls=0
        for ent,patch in patch_family(mesh,dim):
            key=tuple(sorted(patch))
            if key not in cache:
                cols=vector_columns(sp.allowed_nodes(patch));cache[key]=restricted_kernel(A,cols,pr,p)
                calls+=1
            K=cache[key];gain=insert_projected(span,K,fmap);count+=K.basis.shape[0]
            patch_checkpoints.append(dict(family=label,entity=list(ent),tetrahedra=len(patch),rank_gain=gain,**K.checkpoint))
        stages.append(dict(family=label,span_rank=span.rank,rank_gain=span.rank-before,defect=len(free)-span.rank,candidate_count=count,new_kernel_solves=calls))
    # Full C1 vector potentials and gradients of C1 scalar potentials.
    scalar_high=BernsteinSpace.build(mesh,d+1);S_high,_=scalar_high.c1_constraints();S_same,_=sp.c1_constraints()
    grad=gradient_map(scalar_high,sp);sph=ordering(scalar_high,False);sps=ordering(sp,False)
    gradient_span=newspan();c1_span=newspan();union_span=newspan();special=[]
    for v in range(len(mesh.vertices)):
        kh=restricted_kernel(S_high,scalar_high.vertex_nodes(v),sph,p)
        Bg=(grad[:,kh.columns]@kh.basis.T).T.tocsr();Bg.eliminate_zeros()
        check=A@Bg.T;check.eliminate_zeros();assert check.nnz==0
        assert all(set(sp.node_tets[int(c)//3])<=mesh.stars[v] for c in Bg.indices)
        add_full_rows(gradient_span,Bg,free);add_full_rows(union_span,Bg,free)
        ks=restricted_kernel(S_same,sp.vertex_nodes(v),sps,p)
        scalarB=embed(ks,sp.nscalar)
        Bc=sparse.kron(scalarB,sparse.eye(3,dtype=np.int64),format='csr')
        check=A@Bc.T;check.eliminate_zeros();assert check.nnz==0
        add_full_rows(c1_span,Bc,free);add_full_rows(union_span,Bc,free)
        special.append(dict(vertex=v,coordinate=mesh.vertices[v].tolist(),scalar_C1_d1_dimension=kh.basis.shape[0],scalar_C1_d_dimension=ks.basis.shape[0],
                            gradient_cumulative_rank=gradient_span.rank,C1_vector_cumulative_rank=c1_span.rank,combined_cumulative_rank=union_span.rank))
    rg=sparse_mod_rank(S_high,p,sph).rank;rs=sparse_mod_rank(S_same,p,sps).rank
    zero_curl_dim=scalar_high.nscalar-rg-1
    stages += [dict(family='gradients_of_vertex_local_C1_scalar',span_rank=gradient_span.rank,defect=len(free)-gradient_span.rank,
                    global_gradient_dimension=zero_curl_dim,gradient_generation_defect=zero_curl_dim-gradient_span.rank),
               dict(family='vertex_local_C1_vector',span_rank=c1_span.rank,defect=len(free)-c1_span.rank,global_C1_vector_dimension=3*(sp.nscalar-rs)),
               dict(family='gradient_plus_C1_vector',span_rank=union_span.rank,defect=len(free)-union_span.rank)]
    return dict(N=N,k=d-1,potential_degree=d,prime=p,global_dimension=len(free),global_gradient_dimension=zero_curl_dim,
                global_curl_image_dimension=len(free)-zero_curl_dim,seconds=time.time()-start,stages=stages,entity_patches=patch_checkpoints,special_patches=special)

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]/'results'
    for N,d in [(2,5),(2,6),(2,7),(3,5),(3,6)]:
        if (root/f'families_N{N}_d{d}.json').exists():continue
        r=benchmark(N,d);(root/f'families_N{N}_d{d}.json').write_text(json.dumps(r,indent=2))
        print(json.dumps({k:v for k,v in r.items() if k not in ('entity_patches','special_patches')}),flush=True)
