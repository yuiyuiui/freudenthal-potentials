"""Independent exact polynomial check of the three interior gradient templates."""
from geometry import *
from algebra import *
from pathlib import Path
import json

def main():
    root=Path(__file__).resolve().parents[1]/'results/interior_star_degree6'
    scal=np.load(root/'three_scalar_potentials_degree7.npz');vec=np.load(root/'three_gradient_primitive_templates.npz')
    xyz=scal['vertices'];tets=scal['tetrahedra'];phi=scal['coefficients'];w=vec['coefficients']
    amap={a:i for i,a in enumerate(compositions(7,4))};bmap={a:i for i,a in enumerate(compositions(6,4))}
    grads=[];faces={}
    for t,tet in enumerate(tets):
        M=np.vstack([np.ones(4,dtype=np.int64),xyz[tet].T]);inv=np.rint(np.linalg.inv(M)).astype(np.int64)
        assert np.array_equal(inv@M,np.eye(4,dtype=np.int64));grads.append(inv[:,1:])
        for opp in range(4):
            f=tuple(sorted(int(v) for i,v in enumerate(tet) if i!=opp));faces.setdefault(f,[]).append((t,opp))
        for j,b in enumerate(compositions(6,4)):
            expected=np.zeros((3,3),dtype=np.int64)
            for i in range(4):
                a=list(b);a[i]+=1;expected+=7*phi[:,t,amap[tuple(a)]][:,None]*inv[i,1:][None,:]
            assert np.array_equal(expected,w[:,t,j,:])
    face_modes=0
    for face,inc in faces.items():
        for deg,data,lookup in [(7,phi,amap),(6,w,bmap)]:
            for beta3 in compositions(deg,3):
                traces=[]
                for t,opp in inc:
                    beta=[0]*4;tet=list(tets[t])
                    for v,b in zip(face,beta3):beta[tet.index(v)]=b
                    traces.append(data[:,t,lookup[tuple(beta)]])
                if len(traces)==1:assert not np.any(traces[0])
                else:assert np.array_equal(traces[0],traces[1])
                face_modes+=1
    A=sparse.load_npz(root/'degree6_constraints.npz');K=sparse.load_npz(root/'degree6_local_basis.npz')
    P=sparse.load_npz(root/'affine_products.npz');D=sparse.load_npz(root/'quotient_duals.npz');G=sparse.load_npz(root/'three_gradient_primitive_rows.npz')
    maps=np.load(root/'coefficient_maps.npz');free=maps['high_free_local_columns']
    for X in [K,P,G]:
        R=A@X.T;R.eliminate_zeros();assert R.nnz==0
    R=P[:,free]@D.T;R.eliminate_zeros();assert R.nnz==0
    dim=A.shape[1]-sparse_mod_rank(A).rank
    rankP=sparse_mod_rank(P[:,free].tocsr()).rank;rankD=sparse_mod_rank(D).rank;rankPair=sparse_mod_rank((G[:,free]@D.T).tocsr()).rank
    assert dim==546 and rankP==543 and rankD==3 and rankPair==3
    A5=sparse.load_npz(root/'degree5_constraints.npz');K5=sparse.load_npz(root/'degree5_local_basis.npz')
    R=A5@K5.T;R.eliminate_zeros();assert R.nnz==0
    assert A5.shape[1]-sparse_mod_rank(A5).rank==K5.shape[0]==175
    # Connect polynomial templates to the rows used in the algebraic certificate.
    from experiment_core import Mesh,BernsteinSpace,vector_columns
    mesh=Mesh.build(4);sp=BernsteinSpace.build(mesh,6);center=np.array([2,2,2]);v=next(i for i,x in enumerate(mesh.vertices) if np.array_equal(x,center))
    coo=G.tocoo();GG=sparse.coo_matrix((coo.data,(coo.row,maps['high_allowed_columns'][coo.col])),shape=(3,3*sp.nscalar)).tocsr()
    lookup={tuple(xyz[t].ravel()):j for j,t in enumerate(tets)}
    for t in mesh.stars[v]:
        j=lookup[tuple((mesh.vertices[mesh.tets[t]]-center).ravel())]
        broken=GG[:,vector_columns(sp.local_to_global[t])].toarray().reshape(3,-1,3)
        assert np.array_equal(broken,w[:,j])
    result=dict(passed=True,exact_gradient_identity=True,exact_scalar_and_gradient_face_modes_checked=face_modes,
                zero_extension_on_all_outer_faces=True,degree5_dimension=175,degree6_dimension=546,affine_product_dimension=543,new_gradient_generators=3)
    (root/'independent_template_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
