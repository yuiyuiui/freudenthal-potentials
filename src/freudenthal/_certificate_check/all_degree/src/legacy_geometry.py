"""Exact integer Bernstein assembly for AIM 633.

Coordinates are integer cube coordinates; x -> x/N scales all derivatives
by the same nonzero factor and hence leaves the homogeneous constraints
unchanged. Degree d is the POTENTIAL degree: d=k+1.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import permutations, product, combinations
from functools import lru_cache
import numpy as np
from scipy import sparse

@lru_cache(None)
def compositions(d: int, n: int) -> tuple[tuple[int, ...], ...]:
    if d < 0 or n < 1:
        return ()
    if n == 1:
        return ((d,),)
    return tuple((a,)+b for a in range(d+1) for b in compositions(d-a,n-1))

@dataclass
class Mesh:
    N: int
    vertices: np.ndarray
    tets: np.ndarray
    gradients: np.ndarray
    faces: dict
    stars: list[set]

    @classmethod
    def build(cls, N: int) -> 'Mesh':
        if not isinstance(N,int) or N<1:
            raise ValueError('N must be a positive integer')
        vertices=np.array(list(product(range(N+1),repeat=3)),dtype=np.int64)
        lookup={tuple(x):i for i,x in enumerate(vertices)}
        tets=[]
        for a in product(range(N),repeat=3):
            for perm in permutations(range(3)):
                v=np.array(a,dtype=np.int64); tet=[lookup[tuple(v)]]
                for j in perm:
                    v=v.copy(); v[j]+=1; tet.append(lookup[tuple(v)])
                tets.append(tet)
        tets=np.array(tets,dtype=np.int64)
        gradients=[]; faces={}; stars=[set() for _ in vertices]
        for t,tet in enumerate(tets):
            verts=vertices[tet]
            M=np.vstack([np.ones(4,dtype=np.int64),verts.T])
            inv=np.rint(np.linalg.inv(M)).astype(np.int64)
            assert np.array_equal(inv@M,np.eye(4,dtype=np.int64))
            gradients.append(inv[:,1:])
            for v in tet: stars[v].add(t)
            for opp in range(4):
                face=tuple(sorted(int(v) for i,v in enumerate(tet) if i!=opp))
                faces.setdefault(face,[]).append((t,opp))
        assert all(len(a) in (1,2) for a in faces.values())
        return cls(N,vertices,tets,np.array(gradients),faces,stars)

@dataclass
class BernsteinSpace:
    mesh: Mesh
    d: int
    alpha: np.ndarray
    nodes: np.ndarray
    local_to_global: np.ndarray
    node_tets: list[set]
    node_owner_vertices: list[set]
    node_entity: list[tuple]

    @classmethod
    def build(cls,mesh: Mesh,d:int)->'BernsteinSpace':
        if d<1: raise ValueError('d must be >= 1')
        alpha=np.array(compositions(d,4),dtype=np.int64)
        lookup={}; nodes=[]; node_tets=[]; node_owner_vertices=[]; node_entity=[]
        lg=np.empty((len(mesh.tets),len(alpha)),dtype=np.int64)
        for t,tet in enumerate(mesh.tets):
            points=alpha@mesh.vertices[tet]
            for j,p in enumerate(points):
                key=tuple(int(z) for z in p)
                entity=tuple(sorted(int(tet[i]) for i in range(4) if alpha[j,i]))
                if key not in lookup:
                    lookup[key]=len(nodes); nodes.append(key); node_tets.append(set())
                    node_owner_vertices.append(set(int(v) for v in tet)); node_entity.append(entity)
                g=lookup[key]; lg[t,j]=g; node_tets[g].add(t)
                node_owner_vertices[g].intersection_update(int(v) for v in tet)
                assert node_entity[g]==entity
        assert len(nodes)==(mesh.N*d+1)**3
        return cls(mesh,d,alpha,np.array(nodes,dtype=np.int64),lg,node_tets,node_owner_vertices,node_entity)

    @property
    def nscalar(self)->int:
        return len(self.nodes)

    def allowed_nodes(self, support: set[int])->np.ndarray:
        return np.array([i for i,inc in enumerate(self.node_tets) if inc<=support],dtype=np.int64)

    def vertex_nodes(self,v:int)->np.ndarray:
        return np.array([i for i,owners in enumerate(self.node_owner_vertices) if v in owners],dtype=np.int64)

    def curl_constraints(self,all_components:bool=False)->tuple[sparse.csr_matrix,list]:
        """Curl jump divided by d, in degree d-1 face Bernstein coefficients.

        C0 is encoded by shared Bernstein coefficients. With C0 imposed,
        the normal curl jump vanishes identically. By default retain two
        components complementary to a nonzero face-normal component.
        """
        amap={tuple(a):i for i,a in enumerate(self.alpha)}
        rows=[]; cols=[]; vals=[]; meta=[]; r=0
        for face,inc in self.mesh.faces.items():
            if len(inc)!=2: continue  # NO physical boundary condition.
            t0,opp0=inc[0]; normal=self.mesh.gradients[t0,opp0]
            omit=int(np.flatnonzero(normal)[0])
            components=range(3) if all_components else [i for i in range(3) if i!=omit]
            for beta3 in compositions(self.d-1,3):
                terms=[{} for _ in range(3)]
                for sign,(t,opp) in zip((1,-1),inc):
                    tet=self.mesh.tets[t]; beta=[0]*4
                    for v,b in zip(face,beta3): beta[list(tet).index(v)]=b
                    for i,g in enumerate(self.mesh.gradients[t]):
                        a=beta.copy(); a[i]+=1
                        gn=int(self.local_to_global[t,amap[tuple(a)]])
                        # g cross c, c has Cartesian components.
                        cross=((0,-g[2],g[1]),(g[2],0,-g[0]),(-g[1],g[0],0))
                        for comp in range(3):
                            for j,z in enumerate(cross[comp]):
                                if z:
                                    col=3*gn+j
                                    terms[comp][col]=terms[comp].get(col,0)+sign*int(z)
                for comp in components:
                    for c,z in terms[comp].items():
                        if z: rows.append(r); cols.append(c); vals.append(z)
                    meta.append((face,beta3,comp)); r+=1
        A=sparse.coo_matrix((np.array(vals,dtype=np.int64),(rows,cols)),shape=(r,3*self.nscalar)).tocsr()
        A.eliminate_zeros()
        return A,meta

    def c1_constraints(self)->tuple[sparse.csr_matrix,list]:
        """Scalar C1 constraints: one transverse derivative jump per face mode."""
        amap={tuple(a):i for i,a in enumerate(self.alpha)}
        rows=[];cols=[];vals=[];meta=[];r=0
        for face,inc in self.mesh.faces.items():
            if len(inc)!=2: continue
            t0,opp0=inc[0]; normal=self.mesh.gradients[t0,opp0]
            axis=int(np.flatnonzero(normal)[0])
            for beta3 in compositions(self.d-1,3):
                terms={}
                for sign,(t,opp) in zip((1,-1),inc):
                    tet=self.mesh.tets[t]; beta=[0]*4
                    for v,b in zip(face,beta3): beta[list(tet).index(v)]=b
                    for i,g in enumerate(self.mesh.gradients[t]):
                        if g[axis]:
                            a=beta.copy();a[i]+=1
                            col=int(self.local_to_global[t,amap[tuple(a)]])
                            terms[col]=terms.get(col,0)+sign*int(g[axis])
                for c,z in terms.items():
                    if z: rows.append(r);cols.append(c);vals.append(z)
                meta.append((face,beta3,axis));r+=1
        A=sparse.coo_matrix((np.array(vals,dtype=np.int64),(rows,cols)),shape=(r,self.nscalar)).tocsr()
        A.eliminate_zeros()
        return A,meta

    def curl_map(self)->sparse.csr_matrix:
        """Map C0 vector coefficients to BROKEN degree d-1 curl coefficients.
        Unlike curl_constraints, keeps factor d.
        """
        amap={tuple(a):i for i,a in enumerate(self.alpha)}
        beta_all=compositions(self.d-1,4); rows=[];cols=[];vals=[]
        for t,tet in enumerate(self.mesh.tets):
            for bidx,b in enumerate(beta_all):
                for i,g in enumerate(self.mesh.gradients[t]):
                    a=list(b);a[i]+=1; gn=int(self.local_to_global[t,amap[tuple(a)]])
                    cross=((0,-g[2],g[1]),(g[2],0,-g[0]),(-g[1],g[0],0))
                    for comp in range(3):
                        for j,z in enumerate(cross[comp]):
                            if z:
                                rows.append(3*(t*len(beta_all)+bidx)+comp)
                                cols.append(3*gn+j);vals.append(self.d*int(z))
        C=sparse.coo_matrix((np.array(vals,dtype=np.int64),(rows,cols)),shape=(3*len(self.mesh.tets)*len(beta_all),3*self.nscalar)).tocsr()
        C.eliminate_zeros();return C


def vector_columns(nodes:np.ndarray)->np.ndarray:
    return (3*nodes[:,None]+np.arange(3)).ravel()


def matrix_components(A:sparse.csr_matrix):
    """Connected components of the row/column incidence graph, excluding zero cols.
    Returns list (rows,cols) plus unconstrained columns.
    """
    from scipy.sparse.csgraph import connected_components
    m,n=A.shape
    pattern=A.copy();pattern.data=np.ones_like(pattern.data,dtype=np.int8)
    graph=sparse.bmat([[None,pattern],[pattern.T,None]],format='csr')
    _,labels=connected_components(graph,directed=False)
    rr={};cc={}
    for i,l in enumerate(labels[:m]): rr.setdefault(int(l),[]).append(i)
    free=[]
    for j,l in enumerate(labels[m:]):
        if int(l) in rr: cc.setdefault(int(l),[]).append(j)
        else: free.append(j)
    blocks=[(np.array(rr[l],dtype=np.int64),np.array(cs,dtype=np.int64)) for l,cs in cc.items()]
    blocks.sort(key=lambda rc:(-len(rc[1]),-len(rc[0])))
    return blocks,np.array(free,dtype=np.int64)
