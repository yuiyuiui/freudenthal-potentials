"""Resolve the new degree-six generators on the canonical interior vertex star."""
from experiment_core import *
from transformations import *
from pathlib import Path


def main():
 root=Path(__file__).resolve().parents[1]/'results'/'interior_star_degree6';root.mkdir(exist_ok=True)
 mesh=Mesh.build(4);v=next(i for i,x in enumerate(mesh.vertices) if list(x)==[2,2,2])
 low=BernsteinSpace.build(mesh,5);high=BernsteinSpace.build(mesh,6);Al,_=low.curl_constraints();Ah,_=high.curl_constraints()
 lc=vector_columns(low.vertex_nodes(v));hc=vector_columns(high.vertex_nodes(v));pl=ordering(low);ph=ordering(high)
 KL=restricted_kernel(Al,lc,pl);KH=restricted_kernel(Ah,hc,ph)
 R=Ah[:,hc].tocsr();R=R[np.diff(R.indptr)>0];eh=sparse_mod_rank(R,priority=ph[hc]);free=np.array([j for j in range(len(hc)) if j not in eh.rows])
 products=[]
 for M in degree_multipliers(low,high):
  MM=sparse.kron(M,sparse.eye(3,dtype=np.int64),format='csr');products.append((MM[hc][:,lc]@KL.basis.T).T.tocsr())
 P=sparse.vstack(products,format='csr');P.eliminate_zeros();check=R@P.T;check.eliminate_zeros();assert check.nnz==0
 ep=sparse_mod_rank(P[:,free].tocsr(),priority=ph[hc[free]])
 lift,info=lift_rows(ep.nullspace_rows(),1000003);D=coo_from_rows(lift,len(free))
 residual=P[:,free]@D.T;residual.eliminate_zeros();assert residual.nnz==0
 pairing=KH.basis[:,free]@D.T;select=sparse_mod_rank(pairing.tocsr());chosen=np.array(select.independent_inputs)
 prim=KH.basis[chosen];assert len(chosen)==3
 # Test whether the primitive quotient can be represented by local gradients.
 sc=BernsteinSpace.build(mesh,7);S,_=sc.c1_constraints();KS=restricted_kernel(S,sc.vertex_nodes(v),ordering(sc,False));G=gradient_map(sc,high)
 gradients=(G[hc][:,KS.columns]@KS.basis.T).T.tocsr();pairG=gradients[:,free]@D.T;eg=sparse_mod_rank(pairG.tocsr())
 curl=high.curl_map();Cprim=curl[:,hc]@prim.T;Cprim.eliminate_zeros()
 # Export standalone broken Bernstein coefficients on this 24-tetrahedron star.
 support=sorted(mesh.stars[v]);verts=sorted(set(mesh.tets[support].ravel()));vmap={int(g):i for i,g in enumerate(verts)}
 loc_tets=np.array([[vmap[int(g)] for g in mesh.tets[t]] for t in support]);xyz=mesh.vertices[verts]-mesh.vertices[v]
 full=sparse.csr_matrix((3,3*high.nscalar),dtype=np.int64).tolil();full[:,hc]=prim;full=full.tocsr()
 broken=np.stack([full[:,vector_columns(high.local_to_global[t])].toarray().reshape(3,-1,3) for t in support],axis=1)
 np.savez(root/'three_primitive_templates.npz',vertices=xyz,tetrahedra=loc_tets,bernstein_multiindices=high.alpha,
          coefficients=broken,potential_degree=6,center_vertex=vmap[v])
 R5=Al[:,lc].tocsr();R5=R5[np.diff(R5.indptr)>0];sparse.save_npz(root/'degree5_constraints.npz',R5)
 sparse.save_npz(root/'degree5_local_basis.npz',KL.basis);sparse.save_npz(root/'degree6_local_basis.npz',KH.basis)
 sparse.save_npz(root/'degree6_constraints.npz',R);sparse.save_npz(root/'affine_products.npz',P)
 sparse.save_npz(root/'quotient_duals.npz',D);sparse.save_npz(root/'three_primitive_rows.npz',prim)
 np.savez(root/'coefficient_maps.npz',low_allowed_columns=lc,high_allowed_columns=hc,high_free_local_columns=free,
           low_nodes=low.nodes,high_nodes=high.nodes,chosen_rows=chosen)
 result=dict(potential_degree=6,star_tetrahedra=24,star_vertices=len(verts),degree5_dimension=KL.basis.shape[0],degree6_dimension=KH.basis.shape[0],
   affine_product_rank=ep.rank,new_primitive_dimension=D.shape[0],primitive_pairing_rank=select.rank,
   gradient_source_dimension=KS.basis.shape[0],gradient_projection_rank=eg.rank,
   chosen_generators_curl_nnz=[int(Cprim[:,j].nnz) for j in range(3)],
   primitive_coefficient_nnz=[int(prim.getrow(j).nnz) for j in range(3)],
   exact_product_kernel_residual_nnz=0,exact_product_annihilation_nnz=0,dual_lift=info)
 if eg.rank==3:
  scalar_choices=np.array(eg.independent_inputs);phi=KS.basis[scalar_choices]
  scalar_full=sparse.csr_matrix((3,sc.nscalar),dtype=np.int64).tolil();scalar_full[:,KS.columns]=phi;scalar_full=scalar_full.tocsr()
  scalar_broken=np.stack([scalar_full[:,sc.local_to_global[t]].toarray() for t in support],axis=1)
  np.savez(root/'three_scalar_potentials_degree7.npz',vertices=xyz,tetrahedra=loc_tets,bernstein_multiindices=sc.alpha,coefficients=scalar_broken,potential_degree=7,center_vertex=vmap[v])
  gr=gradients[np.array(eg.independent_inputs)];sparse.save_npz(root/'three_gradient_primitive_rows.npz',gr)
  cg=curl[:,hc]@gr.T;cg.eliminate_zeros();assert cg.nnz==0
  full=sparse.csr_matrix((3,3*high.nscalar),dtype=np.int64).tolil();full[:,hc]=gr;full=full.tocsr()
  broken=np.stack([full[:,vector_columns(high.local_to_global[t])].toarray().reshape(3,-1,3) for t in support],axis=1)
  np.savez(root/'three_gradient_primitive_templates.npz',vertices=xyz,tetrahedra=loc_tets,bernstein_multiindices=high.alpha,coefficients=broken,potential_degree=6,center_vertex=vmap[v])
  result['gradient_primitive_exact_curl_residual_nnz']=0
 (root/'verified_local_statement.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
