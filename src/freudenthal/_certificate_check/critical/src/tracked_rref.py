"""Exact-liftable, tracked modular row reduction for the proof attempt."""
import sys,time,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'aim633_experiments'/'src'))
from experiment_core import *

def axpy(a,b,c,p,omit=None):
 for j,z in b.items():
  if j==omit:continue
  v=(a.get(j,0)+c*z)%p
  if v:a[j]=v
  elif j in a:del a[j]

def tracked(A,pr,p=1000003):
 piv={};comb={};pri=[int(z) for z in pr]
 for i in range(A.shape[0]):
  sl=slice(A.indptr[i],A.indptr[i+1]);r={int(j):int(z)%p for j,z in zip(A.indices[sl],A.data[sl]) if z%p};t={i:1}
  while r:
   c=min(r,key=pri.__getitem__);v=r[c]
   if c not in piv:
    z=pow(v,-1,p);piv[c]={j:a*z%p for j,a in r.items()};comb[c]={j:a*z%p for j,a in t.items()};break
   r.pop(c);axpy(r,piv[c],-v,p,c);axpy(t,comb[c],-v,p)
 for c in sorted(piv,key=pri.__getitem__,reverse=True):
  r=piv[c];t=comb[c]
  for j in sorted([j for j in r if j!=c and j in piv],key=pri.__getitem__):
   if j not in r:continue
   v=r.pop(j);axpy(r,piv[j],-v,p,j);axpy(t,comb[j],-v,p)
 return piv,comb

def run(N,d):
 st=time.time();m=Mesh.build(N);s=BernsteinSpace.build(m,d);A,meta=s.curl_constraints();pr=ordering(s);piv,comb=tracked(A,pr)
 ps=list(piv)
 C,info=lift_rows([piv[c] for c in ps],1000003)
 # Need rows C and multipliers normalized with matching rational factors.
 from fractions import Fraction
 W=[]
 for i,c in enumerate(ps):
  scale=C[i][c]
  row={j:rational_reconstruct(z,1000003)*scale for j,z in comb[c].items()}
  if any(x.denominator!=1 for x in row.values()):raise ValueError('nonintegral row multiplier')
  W.append({j:int(x) for j,x in row.items()})
 C=coo_from_rows(C,A.shape[1]);W=coo_from_rows(W,A.shape[0]);res=W@A-C;res.eliminate_zeros();assert res.nnz==0
 out=Path(__file__).resolve().parents[1]/'results'/f'rref_N{N}_d{d}';out.mkdir(exist_ok=True)
 sparse.save_npz(out/'C.npz',C);sparse.save_npz(out/'W.npz',W);np.save(out/'pivots.npy',ps)
 # Bound source coefficient stencil relative to the pivot.
 cmax=0;wmax=0
 for i,c in enumerate(ps):
  q=s.nodes[c//3]
  cc=C.indices[C.indptr[i]:C.indptr[i+1]];cmax=max(cmax,int(np.max(np.abs(s.nodes[cc//3]-q))))
  rr=W.indices[W.indptr[i]:W.indptr[i+1]];ac=A[rr].indices;wmax=max(wmax,int(np.max(np.abs(s.nodes[ac//3]-q))))
 print(json.dumps(dict(N=N,d=d,rank=len(ps),C_nnz=C.nnz,W_nnz=W.nnz,C_radius=cmax,source_radius=wmax,max_Wrow=int(np.max(np.diff(W.indptr))),seconds=time.time()-st)),flush=True)
 return s,A,C,W,ps
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--N',type=int,default=3);ap.add_argument('--degree',type=int,default=6);a=ap.parse_args();run(a.N,a.degree)
