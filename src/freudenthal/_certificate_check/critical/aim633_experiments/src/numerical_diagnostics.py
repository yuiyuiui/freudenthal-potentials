"""Conditioning and reconstruction in the Bernstein COEFFICIENT norm only.
No statement about physical H1 norms or uniform Schwarz stability is made.
"""
from experiment_core import *
from pathlib import Path
from scipy.sparse.linalg import eigsh,splu
from scipy.linalg import svdvals
import warnings


def diagnostics(N,d):
    root=Path(__file__).resolve().parents[1]/'results'/f'N{N}_d{d}'
    sp=BernsteinSpace.build(Mesh.build(N),d);A,_=sp.curl_constraints()
    B=sparse.load_npz(root/'selected_local_basis.npz').astype(float)
    rows=[];eig=[];rng=np.random.default_rng(633)
    glob=restricted_kernel(A,np.arange(A.shape[1]),ordering(sp))
    targets=[]
    for j in range(3):
        w=np.zeros((sp.nscalar,3));w[:,j]=1;targets.append((f'constant_{j}',w.ravel()))
    for j in range(10):
        idx=rng.choice(glob.basis.shape[0],size=min(80,glob.basis.shape[0]),replace=False)
        c=rng.integers(-3,4,size=len(idx));w=np.asarray(c@glob.basis[idx]).ravel();targets.append((f'independent_global_kernel_seed_{j}',w))
    C=sp.curl_map().astype(float)
    for scaling in ['raw','unit_coefficient_norm']:
        norms=np.sqrt(np.asarray(B.multiply(B).sum(axis=1)).ravel())
        BB=B if scaling=='raw' else sparse.diags(1/norms)@B
        gram=(BB@BB.T).tocsc();t=time.time()
        lo=float(eigsh(gram,k=1,which='SA',tol=1e-9,maxiter=20000,return_eigenvectors=False)[0])
        hi=float(eigsh(gram,k=1,which='LA',tol=1e-9,maxiter=20000,return_eigenvectors=False)[0])
        solver=splu(gram)
        eig.append(dict(scaling=scaling,min_gram_eigenvalue=lo,max_gram_eigenvalue=hi,basis_singular_condition=(hi/lo)**.5,
                        gram_nnz=gram.nnz,seconds=time.time()-t))
        for label,w in targets:
            c=solver.solve(np.asarray(BB@w).ravel());fit=np.asarray(BB.T@c).ravel();err=fit-w
            rows.append(dict(target=label,scaling=scaling,relative_coefficient_error=float(np.linalg.norm(err)/np.linalg.norm(w)),
                             continuity_residual_norm=float(np.linalg.norm(A@fit)),relative_curl_error=float(np.linalg.norm(C@err)/max(1,np.linalg.norm(C@w))),
                             coefficient_amplification=float(np.linalg.norm(c)/np.linalg.norm(w))))
    # One interior-vertex patch: floating singular values vs exact modular rank.
    v=next(v for v,x in enumerate(sp.mesh.vertices) if np.array_equal(x,np.ones(3,dtype=int)))
    cols=vector_columns(sp.vertex_nodes(v));R=A[:,cols].tocsr();R=R[np.diff(R.indptr)>0];s=svdvals(R.toarray().astype(float))
    er=sparse_mod_rank(R).rank
    spectral=dict(vertex_coordinate=sp.mesh.vertices[v].tolist(),shape=R.shape,exact_rank=er,
                  smallest_positive_singular_value=float(s[er-1]),largest_singular_value=float(s[0]),
                  largest_zero_singular_value=float(s[er]) if er<len(s) else None,
                  ranks_by_relative_tolerance={str(t):int(sum(s>t*s[0])) for t in [1e-6,1e-8,1e-10,1e-12]})
    result=dict(N=N,potential_degree=d,norm='Euclidean norm of global C0 Bernstein coefficient vectors; NOT an H1 norm',
                basis_spectra=eig,local_constraint_spectrum=spectral,reconstructions=rows)
    (root/'numerical_diagnostics.json').write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
    r=diagnostics(2,6);print(json.dumps(r,indent=2),flush=True)
