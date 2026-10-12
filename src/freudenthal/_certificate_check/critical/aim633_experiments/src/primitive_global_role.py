"""Are the three interior-star primitives globally essential at N=4?
All other vertex spaces remain complete; replace the central star by its
exact 543-dimensional affine degree-five product subspace, then enrich it.
"""
from experiment_core import *
from native_rank import FastRank
from pathlib import Path

root=Path(__file__).resolve().parents[1]/'results';N=4;sp=BernsteinSpace.build(Mesh.build(N),6);A,_=sp.curl_constraints();pr=ordering(sp);ea=sparse_mod_rank(A,priority=pr)
free=np.array([j for j in range(A.shape[1]) if j not in ea.rows]);fmap=np.full(A.shape[1],-1,dtype=np.int64);fmap[free]=np.arange(len(free));e=FastRank(len(free),priority=pr[free]);center=next(v for v,x in enumerate(sp.mesh.vertices) if list(x)==[2,2,2]);checkpoints=[]
for v in range(len(sp.mesh.vertices)):
 if v==center:continue
 K=restricted_kernel(A,vector_columns(sp.vertex_nodes(v)),pr);insert_projected(e,K,fmap);checkpoints.append(dict(vertex=v,rank=e.rank))
others=e.rank
maps=np.load(root/'interior_star_degree6/coefficient_maps.npz');hc=maps['high_allowed_columns'];P=sparse.load_npz(root/'interior_star_degree6/affine_products.npz')
K=KernelResult(hc,P,0,{});insert_projected(e,K,fmap);with_products=e.rank
G=sparse.load_npz(root/'interior_star_degree6/three_gradient_primitive_rows.npz');insert_projected(e,KernelResult(hc,G,0,{}),fmap)
res=dict(N=4,potential_degree=6,global_dimension=len(free),without_central_vertex_space_rank=others,with_central_affine_product_subspace_rank=with_products,
         product_subspace_defect=len(free)-with_products,with_three_gradients_rank=e.rank,enriched_defect=len(free)-e.rank,checkpoints=checkpoints)
(root/'interior_primitives_global_role.json').write_text(json.dumps(res,indent=2));print(json.dumps({k:v for k,v in res.items() if k!='checkpoints'}),flush=True)
