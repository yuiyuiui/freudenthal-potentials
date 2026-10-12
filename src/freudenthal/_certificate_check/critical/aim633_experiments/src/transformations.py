"""Exact coefficient transformations; all returned arrays are integer matrices."""
from geometry import *

def degree_multipliers(low:BernsteinSpace,high:BernsteinSpace)->list[sparse.csr_matrix]:
    """Degree d -> d+1, multiplied by 1,x,y,z and by common scale d+1.
    Actual maps equal these matrices divided by d+1; scaling does not
    affect spans. Coordinates are integer mesh coordinates, not x/N.
    """
    if high.d!=low.d+1 or high.mesh.N!=low.mesh.N:raise ValueError('incompatible degrees/meshes')
    amap={tuple(a):i for i,a in enumerate(low.alpha)}
    entries=[([],[],[]) for _ in range(4)];done=set()
    for t,tet in enumerate(high.mesh.tets):
        for j,a in enumerate(high.alpha):
            g=int(high.local_to_global[t,j])
            if g in done:continue
            done.add(g)
            for i,ai in enumerate(a):
                if not ai:continue
                b=a.copy();b[i]-=1;lo=int(low.local_to_global[t,amap[tuple(b)]])
                factors=[1,*high.mesh.vertices[tet[i]].tolist()]
                for f,z in enumerate(factors):
                    if z:entries[f][0].append(g);entries[f][1].append(lo);entries[f][2].append(int(ai)*int(z))
    out=[]
    for r,c,v in entries:
        M=sparse.coo_matrix((np.array(v,dtype=np.int64),(r,c)),shape=(high.nscalar,low.nscalar)).tocsr()
        M.sum_duplicates();M.eliminate_zeros();out.append(M)
    return out


def gradient_map(scalar:BernsteinSpace,vector:BernsteinSpace)->sparse.csr_matrix:
    """Scalar C1 degree d+1 -> continuous vector degree d.
    Traces are chosen using one incident tetrahedron. C1 constraints
    guarantee agreement with the other tetrahedra.
    """
    if scalar.d!=vector.d+1 or scalar.mesh.N!=vector.mesh.N:raise ValueError('incompatible spaces')
    amap={tuple(a):i for i,a in enumerate(scalar.alpha)}
    rows=[];cols=[];vals=[];done=set()
    for t,tet in enumerate(vector.mesh.tets):
        for j,b in enumerate(vector.alpha):
            g=int(vector.local_to_global[t,j])
            if g in done:continue
            done.add(g)
            for i,grad in enumerate(scalar.mesh.gradients[t]):
                a=b.copy();a[i]+=1;sg=int(scalar.local_to_global[t,amap[tuple(a)]])
                for comp,z in enumerate(grad):
                    if z:rows.append(3*g+comp);cols.append(sg);vals.append(scalar.d*int(z))
    M=sparse.coo_matrix((np.array(vals,dtype=np.int64),(rows,cols)),shape=(3*vector.nscalar,scalar.nscalar)).tocsr()
    M.eliminate_zeros();return M


def embed(K,ncols):
    B=K.basis.tocoo()
    return sparse.coo_matrix((B.data,(B.row,K.columns[B.col])),shape=(B.shape[0],ncols)).tocsr()
