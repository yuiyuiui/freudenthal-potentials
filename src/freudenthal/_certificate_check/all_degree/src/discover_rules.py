"""All-degree normal-form templates: exact affine checks on gap chambers."""
from __future__ import annotations
from pathlib import Path
import sys,itertools,collections,json,time
import numpy as np
from scipy import sparse
BASE=Path(__file__).resolve().parents[1]
OLD=BASE/'dependencies'/'aim633_k5_uniform_proof'
sys.path.insert(0,str(OLD/'src'))
from tracked_rref import run as tracked_run
from geometry import Mesh, BernsteinSpace
from template_signatures import signature
from certify_repairs import kernel_from_C


def position(q,N,d):
 a=np.minimum(np.asarray(q)//d,N-1);r=np.asarray(q)-d*a
 pm=tuple(sorted(range(3),key=lambda j:(-int(r[j]),j)))
 rs=[int(r[j]) for j in pm]
 alpha=(d-rs[0],rs[0]-rs[1],rs[1]-rs[2],rs[2])
 bc=tuple(0 if j==0 else 2 if j==N-1 else 1 for j in a)
 return a,pm,alpha,bc

def key(q,c,N,d):
 a,pm,al,bc=position(q,N,d)
 return (bc,pm,tuple(min(x,3) for x in al),int(c))

def R_matrix(pm):
 R=np.zeros((3,4),dtype=np.int64)
 for t,j in enumerate(pm):R[j,t+1:]=1
 return R

def minimum(coeff,const,gap,lower_degree=7):
 # min f over alpha_i=gap_i<3, alpha_i>=3 otherwise, sum(alpha)>=lower.
 coeff=[int(x) for x in coeff];const=int(const)
 unbound=[i for i,x in enumerate(gap) if x==3]
 if any(coeff[i]<0 for i in unbound):return None
 ans=const+sum(c*x for c,x in zip(coeff,gap))
 deficit=max(0,lower_degree-sum(gap))
 if deficit:
  if not unbound:return float('inf')
  ans += deficit*min(coeff[i] for i in unbound)
 return ans

def identically_zero(coeff,const,gap,lower_degree=7):
 return minimum(coeff,const,gap,lower_degree)==0 and minimum(-np.asarray(coeff),-const,gap,lower_degree)==0

def source_term(s,meta_idx,anchor,R,gap,bc,coefficient,meta):
 face,beta,comp=meta[meta_idx];a,_,_,_=position(anchor,s.mesh.N,s.d)
 face_u=s.mesh.vertices[list(face)]-a
 delta=np.asarray(beta)@face_u-(np.asarray(anchor)-s.d*a)
 tets=[]
 for t,opp in s.mesh.faces[face]:
  u=s.mesh.vertices[s.mesh.tets[t]]-a;tets.append(u)
  for ax in range(3):
   lo=int(u[:,ax].min());hi=int(u[:,ax].max())
   if bc[ax]==0 and (lo<0 or hi>2):return None,('geometry low',ax,lo,hi)
   if bc[ax]==2 and (lo < -1 or hi>1):return None,('geometry high',ax,lo,hi)
   if bc[ax]==1 and (lo < -1 or hi>2):return None,('geometry internal',ax,lo,hi)
 t,opp=s.mesh.faces[face][0];u=tets[0];M=np.vstack([np.ones(4,dtype=int),u.T]);inv=np.rint(np.linalg.inv(M)).astype(np.int64)
 assert np.array_equal(inv@M,np.eye(4,dtype=np.int64))
 ell=inv[:,0];grad=inv[:,1:]
 co=ell[:,None]*np.ones((1,4),dtype=np.int64)+grad@R
 cn=-ell+grad@delta
 if not identically_zero(co[opp],cn[opp],gap):return None,('face not identical',co[opp].tolist(),int(cn[opp]))
 for j in range(4):
  if j!=opp:
   mn=minimum(co[j],cn[j],gap)
   if mn is None or mn<0:return None,('negative beta',co[j].tolist(),int(cn[j]),mn)
 term=dict(tets=[u.tolist() for u in tets],delta=delta.tolist(),component=int(comp),weight=int(coefficient),face=face_u.tolist())
 return term,None

def entity(pm,al):
 u=np.zeros(3,dtype=int);out=[]
 for i in range(4):
  if al[i]>0:out.append(tuple(int(x) for x in u))
  if i<3:u=u.copy();u[pm[i]]+=1
 return set(out)

def build():
 st=time.time();D={};cert={};owner={};fail=collections.Counter();fail_ex={};counts=[]
 for N,d in [(3,7),(3,8),(3,12)]:
  pp=OLD/'results'/f'rref_N{N}_d{d}'
  if not pp.exists():tracked_run(N,d)
  s=BernsteinSpace.build(Mesh.build(N),d);A,meta=s.curl_constraints();C=sparse.load_npz(pp/'C.npz');W=sparse.load_npz(pp/'W.npz');ps=np.load(pp/'pivots.npy');pm={int(p):i for i,p in enumerate(ps)}
  # Recheck the finite identities used only to discover candidate rules.
  assert (W@A-C).nnz==0
  assert (A-A[:,ps]@C).nnz==0
  assert (C[:,ps]-sparse.eye(len(ps),format='csr',dtype=np.int64)).nnz==0
  for i,q in enumerate(s.nodes):
   a,per,alpha,bc=position(q,N,d);gap=tuple(min(x,3) for x in alpha);R=R_matrix(per)
   for c in range(3):
    col=3*i+c;ky=(bc,per,gap,c)
    sg=signature(C.getrow(pm[col]),q,s.nodes) if col in pm else ('free',)
    if ky in D:assert D[ky]==sg,('conflict',ky)
    else:D[ky]=sg
    if col in pm and ky not in cert:
     terms=[];j=pm[col];reason=None
     for mi,coef in zip(W.indices[W.indptr[j]:W.indptr[j+1]],W.data[W.indptr[j]:W.indptr[j+1]]):
      term,reason=source_term(s,int(mi),q,R,gap,bc,int(coef),meta)
      if reason:break
      terms.append(term)
     if reason:
      fail[reason[0]]+=1
      if ky not in fail_ex:fail_ex[ky]=reason
     else:cert[ky]=terms
  G,F=kernel_from_C(s,C,ps)
  non=0;empty=0
  for i,f in enumerate(F):
   q=s.nodes[f//3];a,per,alpha,bc=position(q,N,d);ky=key(q,f%3,N,d)
   own={tuple(a+u) for u in entity(per,alpha)}
   for col in G.indices[G.indptr[i]:G.indptr[i+1]]:
    a1,per1,al1,bc1=position(s.nodes[col//3],N,d)
    own &= {tuple(a1+u) for u in entity(per1,al1)}
   own={tuple(int(x) for x in np.asarray(v)-a) for v in own}
   if not own:non+=1
   if ky not in owner:owner[ky]=own
   else:owner[ky] &= own
   if not owner[ky]:empty+=1
  rec=dict(N=N,d=d,rules=len(D),pivot_rules=sum(v!=('free',) for v in D.values()),universal_source_certificates=len(cert),failed_trials=dict(fail),nonlocal_count=non,empty_owner_keys=sum(not x for x in owner.values()),seconds=time.time()-st)
  counts.append(rec);print(json.dumps(rec),flush=True)
 rules=[]
 for ky,val in D.items():
  r=dict(key=ky,kind='free' if val==('free',) else 'pivot')
  if r['kind']=='pivot':
   r['row']=val
   if ky in cert:r['sources']=cert[ky]
   else:r['failed_reason']=fail_ex.get(ky)
  else:r['owner_candidates']=sorted(owner.get(ky,[]))
  rules.append(r)
 (BASE/'results'/'all_degree_candidate_rules.json').write_text(json.dumps(rules,separators=(',',':')))
 (BASE/'results'/'candidate_build.json').write_text(json.dumps(counts,indent=2))
 return D,cert,owner
if __name__=='__main__':build()
