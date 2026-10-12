"""Load and evaluate an exported vertex-star potential basis.

Example:
    basis = PotentialBasis('results/N2_d6')
    value, curl = basis.evaluate(0, [0.5, 0.5, 0.5])

Points are physical coordinates in [0,1]^3. Stored coefficients are
Bernstein coefficients, NOT nodal function values.
"""
from pathlib import Path
from itertools import permutations
import math
import numpy as np
from scipy import sparse
from geometry import Mesh,compositions

class PotentialBasis:
    def __init__(self,folder):
        folder=Path(folder);g=np.load(folder/'geometry.npz')
        self.N=int(g['N']);self.degree=int(g['degree']);self.mesh=Mesh.build(self.N)
        self.local_to_global=g['local_to_global'];self.coefficients=sparse.load_npz(folder/'selected_local_basis.npz')
        self.alpha=compositions(self.degree,4);self.amap={a:i for i,a in enumerate(self.alpha)}
        self.perms=list(permutations(range(3)))
    def __len__(self):return self.coefficients.shape[0]
    @staticmethod
    def bernstein(alpha,lambdas):
        val=float(math.factorial(sum(alpha)))
        for a,l in zip(alpha,lambdas):val*=float(l)**a/math.factorial(a)
        return val
    def evaluate(self,basis_id,point):
        if not (0<=basis_id<len(self)):raise IndexError('basis index out of range')
        x=np.asarray(point,dtype=float)
        if x.shape!=(3,) or not np.all(np.isfinite(x)) or np.any(x<0) or np.any(x>1):raise ValueError('point must lie in [0,1]^3')
        y=self.N*x;cell=np.minimum(np.floor(y).astype(int),self.N-1);r=y-cell
        perm=tuple(sorted(range(3),key=lambda j:-r[j]));tet=6*((cell[0]*self.N+cell[1])*self.N+cell[2])+self.perms.index(perm)
        rr=r[list(perm)];lam=np.array([1-rr[0],rr[0]-rr[1],rr[1]-rr[2],rr[2]])
        cols=(3*self.local_to_global[tet][:,None]+np.arange(3)).ravel();c=self.coefficients.getrow(basis_id)[:,cols].toarray().reshape(-1,3)
        val=sum((self.bernstein(a,lam)*z for a,z in zip(self.alpha,c)),start=np.zeros(3))
        curl=np.zeros(3)
        for beta in compositions(self.degree-1,4):
            deriv=np.zeros(3)
            for i,g in enumerate(self.mesh.gradients[tet]):
                a=list(beta);a[i]+=1;deriv+=np.cross(g,c[self.amap[tuple(a)]])
            curl+=self.degree*self.N*self.bernstein(beta,lam)*deriv
        return val,curl
