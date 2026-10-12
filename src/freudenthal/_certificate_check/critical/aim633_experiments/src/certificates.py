"""Finite-instance algebraic certificates, including an exact k=4 control.

For a positive result, integer B satisfies A B^T=0, and modular ranks of A
and B sum to the ambient dimension. Since reduction cannot increase rank,
these facts prove the finite-dimensional equality over Q (hence R).
This does NOT certify all meshes/degrees in AIM 633.
"""
from experiment_core import *
from pathlib import Path
import hashlib


def digest(A):
    A=A.tocsr();h=hashlib.sha256()
    for a in (np.array(A.shape,dtype=np.int64),A.indptr,A.indices,A.data):h.update(a.tobytes())
    return h.hexdigest()


def determinant_by_prescribed_pivots(A:sparse.csr_matrix,row_ids,col_ids,p):
    """Independent verifier: no rank search, only the prescribed square minor.
    Eliminates with previously verified pivots in their insertion order.
    Returns its determinant modulo p (rows/columns in the given orders).
    """
    positions={int(c):i for i,c in enumerate(col_ids)};pivots=[];det=1
    for step,r in enumerate(row_ids):
        sl=slice(A.indptr[r],A.indptr[r+1])
        row={positions[int(c)]:int(v)%p for c,v in zip(A.indices[sl],A.data[sl]) if int(c) in positions and int(v)%p}
        while row and min(row)<step:
            j=min(row);z=row.pop(j)
            for k,v in pivots[j].items():
                if k==j:continue
                nv=(row.get(k,0)-z*v)%p
                if nv:row[k]=nv
                elif k in row:del row[k]
        assert row.get(step,0)!=0,f'vanishing prescribed pivot {step}'
        z=row[step];det=det*z%p;inv=pow(z,-1,p)
        pivots.append({j:v*inv%p for j,v in row.items()})
    return det


def positive_certificate(folder,p=1000003):
    folder=Path(folder);ck=json.load(open(folder/'checkpoints.json'));assert ck['defect']==0
    A=sparse.load_npz(folder/'constraints.npz');B=sparse.load_npz(folder/'local_generators.npz')
    geo=np.load(folder/'geometry.npz');sp=BernsteinSpace.build(Mesh.build(int(geo['N'])),int(geo['degree']));pr=ordering(sp)
    ea=sparse_mod_rank(A,p,pr);free=np.array([j for j in range(A.shape[1]) if j not in ea.rows],dtype=np.int64)
    # Only verify the selected basis; the generator pool was verified patchwise.
    row_sel=np.load(folder/'selected_basis_rows.npy');BB=B[row_sel].tocsr()
    residual=A@BB.T;residual.eliminate_zeros();assert residual.nnz==0
    eb=sparse_mod_rank(BB[:,free].tocsr(),p,pr[free]);assert ea.rank+eb.rank==A.shape[1]
    ar=np.array(ea.independent_inputs,dtype=np.int64);ac=np.array(list(ea.rows),dtype=np.int64)
    br=np.array(eb.independent_inputs,dtype=np.int64);bc=np.array(list(eb.rows),dtype=np.int64)
    da=determinant_by_prescribed_pivots(A,ar,ac,p)
    db=determinant_by_prescribed_pivots(BB[:,free].tocsr(),br,bc,p)
    np.savez(folder/'minor_certificate.npz',prime=p,A_rows=ar,A_cols=ac,B_rows=br,B_projected_cols=bc,free_cols=free)
    sparse.save_npz(folder/'selected_local_basis.npz',BB)
    res=dict(N=ck['N'],k=ck['k'],variables=A.shape[1],constraint_rank=ea.rank,basis_dimension=eb.rank,
            exact_kernel_residual_nnz=0,constraint_minor_determinant_mod_p=int(da),basis_minor_determinant_mod_p=int(db),prime=p,
            matrix_sha256=digest(A),basis_sha256=digest(BB),passed=True)
    (folder/'verified_certificate.json').write_text(json.dumps(res,indent=2));return res


def negative_control(N=3,d=5,p=1000003):
    root=Path(__file__).resolve().parents[1]/'results'/f'N{N}_d{d}_negative'
    result=run_vertex(N,d,p,str(root));assert result['defect']>0
    mesh=Mesh.build(N);sp=BernsteinSpace.build(mesh,d);A,_=sp.curl_constraints();pr=ordering(sp)
    ea=sparse_mod_rank(A,p,pr);free=np.array([j for j in range(A.shape[1]) if j not in ea.rows],dtype=np.int64)
    modular=ea.nullspace_rows();lift,info=lift_rows(modular,p);G=coo_from_rows(lift,A.shape[1])
    AG=A@G.T;AG.eliminate_zeros();assert AG.nnz==0
    B=sparse.load_npz(root/'local_generators.npz');P=B[:,free].tocsr()
    el=sparse_mod_rank(P,p,pr[free]);dual,dualinfo=lift_rows(el.nullspace_rows(),p);D=coo_from_rows(dual,len(free))
    annih=P@D.T;annih.eliminate_zeros();assert annih.nnz==0
    # Exact dual independence and nonzero pairings establish the missing modes.
    assert sparse_mod_rank(D,p).rank==result['defect']
    pair=D@G[:,free].T;pair.eliminate_zeros();assert sparse_mod_rank(pair,p).rank==D.shape[0]
    # One explicit witness, selected by the first nonzero dual pairing.
    first=pair.getrow(0);gidx=int(first.indices[0]);w=G.getrow(gidx)
    sparse.save_npz(root/'global_kernel_basis.npz',G);sparse.save_npz(root/'annihilators_on_free_coefficients.npz',D)
    sparse.save_npz(root/'explicit_nonlocalizable_potential.npz',w)
    sparse.save_npz(root/'dual_pairings.npz',pair)
    # Describe where annihilating linear functionals inspect coefficients.
    dual_meta=[]
    for i in range(D.shape[0]):
        row=D.getrow(i);variables=free[row.indices];entities={len(sp.node_entity[int(c)//3])-1 for c in variables}
        coords=sp.nodes[variables//3]
        dual_meta.append(dict(index=i,nnz=row.nnz,coefficient_max=int(max(abs(row.data))),entity_dimensions=sorted(entities),
                              coefficient_grid_bbox=[coords.min(axis=0).tolist(),coords.max(axis=0).tolist()]))
    verify=dict(N=N,k=d-1,global_dimension=G.shape[0],local_span_dimension=el.rank,exact_defect=D.shape[0],
                global_kernel_exact_residual_nnz=0,local_span_exact_annihilation_nnz=0,dual_pairing_rank=D.shape[0],
                witness_global_kernel_row=gidx,witness_dual_pairing=int(first.data[0]),
                global_lift=info,dual_lift=dualinfo,dual_metadata=dual_meta,passed=True)
    (root/'verified_negative_certificate.json').write_text(json.dumps(verify,indent=2))
    return verify

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]/'results'
    for case in ['N2_d6','N3_d6']:
        print(json.dumps(positive_certificate(root/case)),flush=True)
    print(json.dumps(negative_control()),flush=True)
