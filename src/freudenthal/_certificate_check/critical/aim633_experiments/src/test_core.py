from geometry import *
from algebra import *
from experiment_core import *
from transformations import *
import sympy as sy
from fractions import Fraction
from math import factorial
import json,time

def check_zero(M):
    M.eliminate_zeros()
    assert M.nnz==0

def broken_constraints(space):
    mesh=space.mesh;d=space.d;alpha=compositions(d,4);amap={a:i for i,a in enumerate(alpha)}
    r=[];c=[];v=[];row=0;nloc=len(alpha)
    for face,inc in mesh.faces.items():
        if len(inc)!=2:continue
        for mode in ['value','curl']:
            for beta3 in compositions(d if mode=='value' else d-1,3):
                for comp in range(3):
                    terms={}
                    for sign,(t,opp) in zip((1,-1),inc):
                        tet=list(mesh.tets[t]);beta=[0]*4
                        for vert,b in zip(face,beta3):beta[tet.index(vert)]=b
                        if mode=='value':
                            col=3*(t*nloc+amap[tuple(beta)])+comp
                            terms[col]=terms.get(col,0)+sign
                        else:
                            for i,g in enumerate(mesh.gradients[t]):
                                a=beta.copy();a[i]+=1
                                cross=((0,-g[2],g[1]),(g[2],0,-g[0]),(-g[1],g[0],0))
                                for j,z in enumerate(cross[comp]):
                                    col=3*(t*nloc+amap[tuple(a)])+j
                                    terms[col]=terms.get(col,0)+sign*int(z)
                    for col,z in terms.items():
                        if z:r.append(row);c.append(col);v.append(z)
                    row+=1
    M=sparse.coo_matrix((np.array(v,dtype=np.int64),(r,c)),shape=(row,3*len(mesh.tets)*nloc)).tocsr()
    M.eliminate_zeros();return M

def bernstein_eval(coeff,degree,lambdas):
    out=[0,0,0]
    for alpha,row in zip(compositions(degree,4),coeff):
        b=Fraction(factorial(degree),1)
        for a,l in zip(alpha,lambdas):b*=l**a/factorial(a)
        for j,z in enumerate(row):out[j]+=b*int(z)
    return out

def eval_tet(space,t,vector,point):
    # SymPy exact inversion is independent of the assembly's integer-rounding path.
    vertices=space.mesh.vertices[space.mesh.tets[t]]
    M=sy.Matrix([[1,*map(int,x)] for x in vertices])
    C=M.inv()  # lambda_i(x)=sum_j C[j,i] [1,x]_j
    lam=[]
    for i in range(4):
        val=Fraction(int(C[0,i]))+sum(Fraction(int(C[j+1,i]))*point[j] for j in range(3))
        lam.append(val)
    local=vector.reshape(-1,3)[space.local_to_global[t]]
    value=bernstein_eval(local,space.d,lam)
    amap={tuple(a):i for i,a in enumerate(space.alpha)}
    deriv=[]
    for beta in compositions(space.d-1,4):
        D=[[0]*3 for _ in range(3)]
        for i in range(4):
            a=list(beta);a[i]+=1;coeff=local[amap[tuple(a)]]
            for j in range(3):
                for k in range(3):D[j][k]+=space.d*int(C[k+1,i])*int(coeff[j])
        deriv.append([D[2][1]-D[1][2],D[0][2]-D[2][0],D[1][0]-D[0][1]])
    curl=bernstein_eval(np.array(deriv,dtype=np.int64),space.d-1,lam)
    return value,curl


def main():
    checks=[];start=time.time()
    for N,d in [(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]:
        sp=BernsteinSpace.build(Mesh.build(N),d);A,_=sp.curl_constraints();full,_=sp.curl_constraints(True);broken=broken_constraints(sp)
        r=sparse_mod_rank(A).rank;rf=sparse_mod_rank(full).rank;rb=sparse_mod_rank(broken).rank
        assert r==rf
        assert A.shape[1]-r==broken.shape[1]-rb
        if N==1 and d<=2:assert sy.Matrix(A.toarray()).rank()==r
        checks.append(dict(test='reduced_vs_full_vs_broken',N=N,d=d,dimension=A.shape[1]-r,passed=True))
    for N,d in [(1,2),(2,3),(2,6)]:
        mesh=Mesh.build(N);sp=BernsteinSpace.build(mesh,d);A,_=sp.curl_constraints()
        for j in range(3):
            vec=np.zeros((sp.nscalar,3),dtype=np.int64);vec[:,j]=1;assert not np.any(A@vec.ravel())
            for axis in range(3):
                vec[:,j]=sp.nodes[:,axis];assert not np.any(A@vec.ravel())
        # Global Bernstein coefficients of a quadratic x_a x_b times d(d-1).
        for a in range(3):
            for b in range(3):
                coeff=np.zeros(sp.nscalar,dtype=np.int64);done=set()
                for t,tet in enumerate(mesh.tets):
                    verts=mesh.vertices[tet]
                    for j,alpha in enumerate(sp.alpha):
                        g=int(sp.local_to_global[t,j]);val=int((alpha@verts[:,a])*(alpha@verts[:,b])-alpha@(verts[:,a]*verts[:,b]))
                        if g in done:assert coeff[g]==val
                        else:coeff[g]=val;done.add(g)
                for comp in range(3):
                    vec=np.zeros((sp.nscalar,3),dtype=np.int64);vec[:,comp]=coeff
                    assert not np.any(A@vec.ravel())
        checks.append(dict(test='constant_affine_quadratic_exact',N=N,d=d,passed=True))
    # Original face trace evaluation at independent exact rational points.
    root=__import__('pathlib').Path(__file__).resolve().parents[1]
    B=sparse.load_npz(root/'results/N2_d6/local_generators.npz');sp=BernsteinSpace.build(Mesh.build(2),6)
    rng=np.random.default_rng(633);internal=[(f,inc) for f,inc in sp.mesh.faces.items() if len(inc)==2];samples=0
    chosen=rng.choice(B.shape[0],size=45,replace=False)
    for idx in chosen:
        vec=B.getrow(int(idx)).toarray().ravel();active=set()
        for gn in np.flatnonzero(np.any(vec.reshape(-1,3)!=0,axis=1)):active.update(sp.node_tets[int(gn)])
        relevant=[(f,inc) for f,inc in internal if any(t in active for t,_ in inc)]
        for fi in rng.choice(len(relevant),size=min(3,len(relevant)),replace=False):
            face,inc=relevant[int(fi)];weights=rng.integers(1,5,size=3);den=int(sum(weights))
            point=[sum(Fraction(int(w)*int(sp.mesh.vertices[v,j]),den) for v,w in zip(face,weights)) for j in range(3)]
            vp,cp=eval_tet(sp,inc[0][0],vec,point);vm,cm=eval_tet(sp,inc[1][0],vec,point)
            assert vp==vm and cp==cm;samples+=1
    checks.append(dict(test='independent_rational_value_and_curl_face_evaluation',generators=45,face_samples=samples,passed=True))
    # First support audit uses the original tetrahedra, not just node indices.
    ck=json.load(open(root/'results/N2_d6/checkpoints.json'));offset=0
    for patch in ck['patches']:
        n=patch['dimension'];block=B[offset:offset+n];star=sp.mesh.stars[patch['vertex']]
        for t in set(range(len(sp.mesh.tets)))-star:
            cols=vector_columns(sp.local_to_global[t]);assert block[:,cols].nnz==0
        offset+=n
    checks.append(dict(test='all_local_generators_zero_on_every_outside_tetrahedron',generators=offset,passed=True))
    print(json.dumps(dict(passed=True,seconds=time.time()-start,checks=checks),indent=2))
if __name__=='__main__':main()
