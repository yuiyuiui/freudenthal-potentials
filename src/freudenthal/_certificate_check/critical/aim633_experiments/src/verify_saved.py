"""Verify stored finite-case certificates without searching for a new basis.
Usage: python src/verify_saved.py results/N2_d6 results/N3_d6
"""
from experiment_core import *
from certificates import determinant_by_prescribed_pivots,digest
from pathlib import Path
import sys

def verify(folder):
    folder=Path(folder);geo=np.load(folder/'geometry.npz');minor=np.load(folder/'minor_certificate.npz')
    sp=BernsteinSpace.build(Mesh.build(int(geo['N'])),int(geo['degree']));A,_=sp.curl_constraints()
    saved=sparse.load_npz(folder/'constraints.npz');difference=A-saved;difference.eliminate_zeros();assert difference.nnz==0
    B=sparse.load_npz(folder/'selected_local_basis.npz');p=int(minor['prime'])
    R=A@B.T;R.eliminate_zeros();assert R.nnz==0
    da=determinant_by_prescribed_pivots(A,minor['A_rows'],minor['A_cols'],p)
    db=determinant_by_prescribed_pivots(B[:,minor['free_cols']].tocsr(),minor['B_rows'],minor['B_projected_cols'],p)
    assert da!=0 and db!=0 and len(minor['A_rows'])+len(minor['B_rows'])==A.shape[1]
    assert B.shape[0]==len(minor['B_rows'])
    for i in range(B.shape[0]):
        owners=None
        for c in B.indices[B.indptr[i]:B.indptr[i+1]]:
            possible=sp.node_owner_vertices[int(c)//3]
            owners=possible.copy() if owners is None else owners&possible
        assert owners,'basis function has no single-star support'
    return dict(case=folder.name,passed=True,dimension=B.shape[0],exact_residual_nnz=0,A_minor_mod_p=int(da),B_minor_mod_p=int(db),prime=p)
if __name__=='__main__':
    folders=sys.argv[1:] or [str(Path(__file__).resolve().parents[1]/'results/N2_d6')]
    for f in folders:print(json.dumps(verify(f)),flush=True)
