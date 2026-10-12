"""Sparse exact modular elimination and rational reconstruction utilities."""
from __future__ import annotations
from fractions import Fraction
import math,time
import numpy as np
from scipy import sparse

class SparseEchelon:
    def __init__(self,ncols:int,p:int=1000003,priority:np.ndarray|None=None):
        self.ncols=ncols;self.p=p;self.rows={};self.max_nnz=0;self.operations=0
        self.priority=np.arange(ncols) if priority is None else np.asarray(priority)
        self.pr=[int(i) for i in self.priority]
        self.input_count=0;self.independent_inputs=[]
    @property
    def rank(self):return len(self.rows)
    def add(self,row:dict[int,int],record=True)->bool:
        p=self.p; row={int(j):int(v)%p for j,v in row.items() if int(v)%p}
        idx=self.input_count;self.input_count+=1
        while row:
            col=min(row,key=self.pr.__getitem__)
            val=row[col]
            if col not in self.rows:
                inv=pow(val,-1,p)
                row={j:(v*inv)%p for j,v in row.items()}
                self.rows[col]=row;self.max_nnz=max(self.max_nnz,len(row))
                self.independent_inputs.append(idx)
                return True
            pivot=self.rows[col];del row[col]
            for j,z in pivot.items():
                if j==col:continue
                v=(row.get(j,0)-val*z)%p
                if v:row[j]=v
                elif j in row:del row[j]
            self.operations+=len(pivot)
        return False
    def add_matrix_rows(self,A:sparse.csr_matrix):
        for i in range(A.shape[0]):
            sl=slice(A.indptr[i],A.indptr[i+1])
            self.add(dict(zip(A.indices[sl],A.data[sl])))
    def nullspace_rows(self)->list[dict[int,int]]:
        """Sparse basis vectors in ambient coordinates (one per free variable)."""
        p=self.p
        free=[j for j in range(self.ncols) if j not in self.rows]
        expr={j:{j:1} for j in free}
        for col in sorted(self.rows,key=self.pr.__getitem__,reverse=True):
            out={}
            for j,z in self.rows[col].items():
                if j==col:continue
                for f,a in expr[j].items():
                    v=(out.get(f,0)-z*a)%p
                    if v:out[f]=v
                    elif f in out:del out[f]
            expr[col]=out
        basis={f:{} for f in free}
        for j,ex in expr.items():
            for f,a in ex.items():basis[f][j]=a
        return [basis[f] for f in free]


def coo_from_rows(rows:list[dict[int,int]],ncols:int)->sparse.csr_matrix:
    ri=[];ci=[];va=[]
    for i,row in enumerate(rows):
        for j,v in row.items():
            if v:ri.append(i);ci.append(j);va.append(v)
    return sparse.coo_matrix((np.array(va,dtype=np.int64),(ri,ci)),shape=(len(rows),ncols)).tocsr()


def sparse_mod_rank(A:sparse.csr_matrix,p=1000003,priority=None):
    e=SparseEchelon(A.shape[1],p,priority);e.add_matrix_rows(A);return e


def rational_reconstruct(a:int,p:int)->Fraction|None:
    """Symmetric rational reconstruction |num|,den <= sqrt((p-1)/2)."""
    a=int(a)%p
    if a==0:return Fraction(0)
    bound=math.isqrt((p-1)//2)
    r0,r1=p,a;t0,t1=0,1
    while abs(r1)>bound:
        q=r0//r1;r0,r1=r1,r0-q*r1;t0,t1=t1,t0-q*t1
    if t1==0 or abs(t1)>bound or math.gcd(r1,t1)!=1:return None
    if (r1-a*t1)%p:return None
    return Fraction(r1,t1)


def lift_rows(rows:list[dict[int,int]],p:int)->tuple[list[dict[int,int]],dict]:
    out=[];max_num=0;max_den=1
    for row in rows:
        rec={j:rational_reconstruct(a,p) for j,a in row.items()}
        if any(z is None for z in rec.values()):raise ValueError('rational reconstruction failed')
        den=math.lcm(*(z.denominator for z in rec.values())) if rec else 1
        vec={j:int(z*den) for j,z in rec.items()}
        if vec:
            gcd=math.gcd(*vec.values());vec={j:v//gcd for j,v in vec.items()}
        max_num=max(max_num,max((abs(v) for v in vec.values()),default=0))
        max_den=max(max_den,den);out.append(vec)
    return out,dict(max_abs_integer_coefficient=max_num,max_reconstruction_denominator=max_den)
