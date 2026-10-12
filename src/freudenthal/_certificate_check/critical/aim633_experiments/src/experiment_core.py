from __future__ import annotations
from geometry import *
from algebra import *
from dataclasses import dataclass
from itertools import combinations
import json,time,hashlib


def ordering(space:BernsteinSpace,vector:bool=True)->np.ndarray:
    mult=3 if vector else 1
    order=sorted(range(mult*space.nscalar),key=lambda c:(-len(space.node_entity[c//mult]),tuple(space.nodes[c//mult]),c%mult))
    return np.argsort(order)

@dataclass
class KernelResult:
    columns:np.ndarray
    basis:sparse.csr_matrix  # rows in restricted coordinates, exact integer lift
    rank:int
    checkpoint:dict


def restricted_kernel(A:sparse.csr_matrix,cols:np.ndarray,priority:np.ndarray,p:int=1000003,verify:bool=True)->KernelResult:
    start=time.time();R=A[:,cols].tocsr();R=R[np.diff(R.indptr)>0]
    ep=sparse_mod_rank(R,p=p,priority=priority[cols])
    modular=ep.nullspace_rows()
    lifted,info=lift_rows(modular,p)
    B=coo_from_rows(lifted,len(cols))
    if verify:
        # Int64 is safe only after this explicit bound.
        bound=int(np.max(np.abs(R.data),initial=0))*int(np.max(np.diff(R.indptr),initial=0))*int(np.max(np.abs(B.data),initial=0))
        if bound>=np.iinfo(np.int64).max:raise OverflowError('exact product bound exceeds int64')
        residual=R@B.T;residual.eliminate_zeros()
        if residual.nnz:raise AssertionError(f'kernel lift residual {residual.nnz}')
    else:bound=None
    # Free-variable identity in modular construction gives full row independence.
    ck=dict(variables=len(cols),constraint_rows=R.shape[0],constraint_rank_mod_p=ep.rank,dimension=B.shape[0],basis_nnz=B.nnz,
            max_elimination_row_nnz=ep.max_nnz,exact_kernel_residual_nnz=0 if verify else None,
            product_bound=bound,seconds=time.time()-start,**info)
    return KernelResult(cols,B,ep.rank,ck)


def insert_embedded(e:SparseEchelon,K:KernelResult)->int:
    old=e.rank
    for i in range(K.basis.shape[0]):
        sl=slice(K.basis.indptr[i],K.basis.indptr[i+1])
        row={int(K.columns[j]):int(z) for j,z in zip(K.basis.indices[sl],K.basis.data[sl])}
        e.add(row)
    return e.rank-old


def insert_projected(e,K:KernelResult,colmap:np.ndarray)->int:
    old=e.rank
    if old==e.ncols:
        # Preserve row indices used by certificate selection in Python echelon.
        if hasattr(e,'input_count'):e.input_count+=K.basis.shape[0]
        return 0
    mapped=colmap[K.columns];B=K.basis.tocoo();keep=mapped[B.col]>=0
    P=sparse.coo_matrix((B.data[keep],(B.row[keep],mapped[B.col[keep]])),shape=(B.shape[0],e.ncols)).tocsr()
    e.add_matrix_rows(P)
    return e.rank-old


def patch_family(mesh:Mesh,dimension:int):
    ent={}
    for t,tet in enumerate(mesh.tets):
        for e in combinations(sorted(int(v) for v in tet),dimension+1):ent.setdefault(e,set()).add(t)
    return list(ent.items())


def run_vertex(N:int,d:int,p:int=1000003,artifact_path:str|None=None)->dict:
    start=time.time();mesh=Mesh.build(N);sp=BernsteinSpace.build(mesh,d);A,_=sp.curl_constraints();priority=ordering(sp)
    global_e=sparse_mod_rank(A,p,priority);glob_dim=A.shape[1]-global_e.rank
    free=np.array([j for j in range(A.shape[1]) if j not in global_e.rows],dtype=np.int64)
    free_map=np.full(A.shape[1],-1,dtype=np.int64);free_map[free]=np.arange(len(free))
    span=SparseEchelon(len(free),p,priority[free]);patches=[];basis_pieces=[]
    vertices=sorted(range(len(mesh.vertices)),key=lambda v:(sum(c in (0,N) for c in mesh.vertices[v]),tuple(mesh.vertices[v])))
    for v in vertices:
        cols=vector_columns(sp.vertex_nodes(v));K=restricted_kernel(A,cols,priority,p)
        gain=insert_projected(span,K,free_map)
        rec=dict(vertex=int(v),coordinate=mesh.vertices[v].tolist(),tetrahedra=len(mesh.stars[v]),rank_gain=gain,cumulative_span=span.rank,**K.checkpoint)
        patches.append(rec)
        if artifact_path:
            B=K.basis.tocoo();basis_pieces.append(sparse.coo_matrix((B.data,(B.row,cols[B.col])),shape=(B.shape[0],A.shape[1])).tocsr())
    result=dict(N=N,k=d-1,potential_degree=d,prime=p,tetrahedra=len(mesh.tets),vertices=len(mesh.vertices),variables=A.shape[1],
        constraint_rows=A.shape[0],constraint_nnz=A.nnz,constraint_rank=global_e.rank,global_dimension=glob_dim,
        sum_local_dimensions=sum(r['dimension'] for r in patches),local_span_rank=span.rank,defect=glob_dim-span.rank,
        max_global_elimination_row_nnz=global_e.max_nnz,max_span_elimination_row_nnz=span.max_nnz,
        seconds=time.time()-start,patches=patches)
    if artifact_path:
        import os
        os.makedirs(artifact_path,exist_ok=True)
        sparse.save_npz(artifact_path+'/constraints.npz',A)
        BB=sparse.vstack(basis_pieces,format='csr');sparse.save_npz(artifact_path+'/local_generators.npz',BB)
        np.save(artifact_path+'/global_free_columns.npy',free)
        np.save(artifact_path+'/selected_basis_rows.npy',np.array(span.independent_inputs,dtype=np.int64))
        np.savez(artifact_path+'/geometry.npz',vertices=mesh.vertices,tets=mesh.tets,coefficient_nodes=sp.nodes,local_to_global=sp.local_to_global,degree=d,N=N)
        with open(artifact_path+'/checkpoints.json','w') as f:json.dump(result,f,indent=2)
    return result

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--N',type=int,default=2);ap.add_argument('--degree',type=int,default=6);ap.add_argument('--prime',type=int,default=1000003);ap.add_argument('--save')
 args=ap.parse_args();r=run_vertex(args.N,args.degree,args.prime,args.save)
 print(json.dumps({k:v for k,v in r.items() if k!='patches'}),flush=True)
