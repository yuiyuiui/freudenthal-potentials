"""Construct the certified local basis directly, without a nullspace search.

Scope of this new constructor: N>=2 and potential degree d>=7. The d=6
case is supplied by the accompanying critical-degree proof package.
"""
from __future__ import annotations
from pathlib import Path
from itertools import permutations
import argparse,json,time
import numpy as np
from scipy import sparse
from legacy_geometry import Mesh,BernsteinSpace
ROOT=Path(__file__).resolve().parents[1]
PERMS=list(permutations(range(3)))
def key(q,c,N,d):
 a=np.minimum(q//d,N-1);r=q-d*a
 pm=tuple(sorted(range(3),key=lambda i:(-int(r[i]),i)))
 rr=[int(r[j]) for j in pm]
 g=(d-rr[0],rr[0]-rr[1],rr[1]-rr[2],rr[2]);g=tuple(min(z,3) for z in g)
 bc=tuple(0 if x==0 else 2 if x==N-1 else 1 for x in a)
 b=bc[0]*9+bc[1]*3+bc[2];gg=g[0]*64+g[1]*16+g[2]*4+g[3]
 return ((b*6+PERMS.index(pm))*256+gg)*3+c,a

def construct(N:int,d:int,save:Path|None=None,verify:bool=True):
 if N<2 or d<7:raise ValueError('This constructor uses the new all-degree table: require N>=2,d>=7.')
 st=time.time();rules=json.loads((ROOT/'results/all_degree_candidate_rules.json').read_text());D={}
 for r in rules:
  bc,pm,g,c=r['key'];b=bc[0]*9+bc[1]*3+bc[2];gg=g[0]*64+g[1]*16+g[2]*4+g[3]
  D[((b*6+PERMS.index(tuple(pm)))*256+gg)*3+c]=r
 ls=(ROOT/'results/universal_owners.txt').read_text().splitlines();owners={int(a):int(b) for a,b in (s.split() for s in ls[1:])};assert len(owners)==int(ls[0])
 mesh=Mesh.build(N);space=BernsteinSpace.build(mesh,d);lookup={tuple(q):i for i,q in enumerate(space.nodes)};vlookup={tuple(v):i for i,v in enumerate(mesh.vertices)}
 row=[];col=[];val=[];ps=[];free=[];vertex=[]
 for i,q in enumerate(space.nodes):
  for c in range(3):
   kk,a=key(q,c,N,d);r=D[kk];index=3*i+c
   if r['kind']=='pivot':
    ri=len(ps);ps.append(index)
    for delta,co,z in r['row']:
     p=tuple(q+np.array(delta));assert p in lookup
     row.append(ri);col.append(3*lookup[p]+co);val.append(z)
   else:
    free.append(index);mask=owners[kk];bit=(mask&-mask).bit_length()-1
    v=a+np.array([bit//4,(bit//2)%2,bit%2]);vertex.append(vlookup[tuple(v)])
 ps=np.array(ps,dtype=np.int64);free=np.array(free,dtype=np.int64);vertex=np.array(vertex,dtype=np.int64);n=3*space.nscalar
 C=sparse.coo_matrix((np.array(val,dtype=np.int64),(row,col)),shape=(len(ps),n)).tocsr()
 CF=C[:,free].tocoo();G=sparse.coo_matrix((np.r_[np.ones(len(free),dtype=np.int64),-CF.data],(np.r_[np.arange(len(free)),CF.col],np.r_[free,ps[CF.row]])),shape=(len(free),n)).tocsr()
 dim=3*N**3*(d-1)**2*(d-2)+3*N**2*(d-1)*(5*d-4)+9*N*(2*d-1)+6;assert len(free)==dim
 if verify:
  A,_=space.curl_constraints(all_components=True)
  residual=C[:,ps]-sparse.eye(len(ps),dtype=np.int64,format='csr');residual.eliminate_zeros();assert residual.nnz==0
  # Products are integer; bound all possible accumulation before multiplication.
  bound=int(np.max(np.diff(A.indptr)))*int(np.max(np.abs(A.data),initial=0))*int(max(np.max(np.abs(C.data),initial=0),np.max(np.abs(G.data),initial=0)))
  assert bound<np.iinfo(np.int64).max
  for residual in [A-A[:,ps]@C,A@G.T]:residual.eliminate_zeros();assert residual.nnz==0
  for i,v in enumerate(vertex):
   for cc in G.indices[G.indptr[i]:G.indptr[i+1]]:assert int(v) in space.node_owner_vertices[int(cc)//3]
 else:bound=None
 out=dict(N=N,potential_degree=d,k=d-1,dimension=len(free),constraint_rank=len(ps),basis_nnz=G.nnz,max_abs_basis_coefficient=int(np.max(np.abs(G.data),initial=0)),all_basis_vectors_vertex_local=True,exact_reassembly_checks=verify,integer_product_bound=bound,seconds=time.time()-st)
 if save:
  save.mkdir(parents=True,exist_ok=True);sparse.save_npz(save/'basis.npz',G);sparse.save_npz(save/'normalform.npz',C)
  np.savez(save/'metadata.npz',N=N,d=d,free=free,pivots=ps,owner_vertices=vertex,coefficient_nodes=space.nodes,vertices=mesh.vertices,tets=mesh.tets,local_to_global=space.local_to_global)
  (save/'verified.json').write_text(json.dumps(out,indent=2))
 return out,G

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--N',type=int,default=2);ap.add_argument('--degree',type=int,default=7);ap.add_argument('--save',type=Path);a=ap.parse_args();out,_=construct(a.N,a.degree,a.save);print(json.dumps(out),flush=True)
if __name__=='__main__':main()
