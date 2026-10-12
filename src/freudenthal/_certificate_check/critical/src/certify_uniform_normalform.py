"""Certify the finite identities used in the all-N, degree-six transfer theorem.

The mathematical transfer lemma (not a rank extrapolation) is in the report.
This checker reassembles the source constraints; verifies W*A=C and the
reverse inclusion A=A[:,pivots]*C; and audits locality and template coverage.
"""
from tracked_rref import *
from template_signatures import key,signature
from collections import Counter

def run():
 st=time.time();N=d=6;base=Path(__file__).resolve().parents[1]/'results';src=base/'rref_N6_d6'
 s=BernsteinSpace.build(Mesh.build(N),d);A,meta=s.curl_constraints();C=sparse.load_npz(src/'C.npz');W=sparse.load_npz(src/'W.npz');ps=np.load(src/'pivots.npy');pmap={int(c):i for i,c in enumerate(ps)}
 # Explicit bound before int64 products.
 bound=int(np.max(np.abs(W.data)))*int(np.max(np.diff(W.indptr)))*int(np.max(np.abs(A.data)))
 assert bound<np.iinfo(np.int64).max
 R=W@A-C;R.eliminate_zeros();assert R.nnz==0
 R=A-A[:,ps]@C;R.eliminate_zeros();assert R.nnz==0
 R=C[:,ps]-sparse.eye(len(ps),format='csr',dtype=np.int64);R.eliminate_zeros();assert R.nnz==0
 templates={};rC=0
 for qid,q in enumerate(s.nodes):
  for c in range(3):
   col=3*qid+c;k=key(q,c,N,d,5)
   if col in pmap:
    rr=C.getrow(pmap[col]);sig=signature(rr,q,s.nodes);rC=max(rC,int(np.max(np.abs(s.nodes[rr.indices//3]-q))))
   else:sig=('free',)
   if k in templates:assert templates[k]==sig,('nonlocal normal form dependence',k)
   else:templates[k]=sig
 assert rC<=7
 # Coefficient rows have radius two; their two incident tetrahedra radius seven.
 geomlow=[];geomhi=[];rA=0;rAgeom=0
 for i,(face,beta,comp) in enumerate(meta):
  vv=np.concatenate([s.mesh.vertices[s.mesh.tets[t]] for t,_ in s.mesh.faces[face]])*d
  geomlow.append(vv.min(axis=0));geomhi.append(vv.max(axis=0))
  cc=A.indices[A.indptr[i]:A.indptr[i+1]]
  if len(cc):
   qq=s.nodes[cc//3];q=qq[np.lexsort((qq[:,2],qq[:,1],qq[:,0]))[0]]
   rA=max(rA,int(np.max(np.abs(qq-q))));rAgeom=max(rAgeom,int(np.max(np.abs(vv-q))))
 assert rA<=2 and rAgeom<=7
 geomlow=np.array(geomlow);geomhi=np.array(geomhi);rWgeom=0
 for i,p in enumerate(ps):
  q=s.nodes[p//3];ids=W.indices[W.indptr[i]:W.indptr[i+1]]
  rWgeom=max(rWgeom,int(max(np.max(np.abs(geomlow[ids]-q)),np.max(np.abs(geomhi[ids]-q)))))
 assert rWgeom<=11
 # For any N>=5, each axis is: left distance <=L; right distance <=L;
 # or interior with one of d residues. The two boundary bands are disjoint.
 coverage={}
 for L in [5,7,11,12]:
  actual=set(tuple((int(x)%d,min(int(x),L+1),min(N*d-int(x),L+1))) for x in range(N*d+1))
  expected=set((x%d,x,L+1) for x in range(L+1)) | set(((-x)%d,L+1,x) for x in range(L+1)) | set((r,L+1,L+1) for r in range(d))
  assert actual==expected
  coverage[L]=len(actual)
 # Store the rules as plain JSON in addition to the exact source matrices.
 rules=[dict(key=k,kind='free') if v==('free',) else dict(key=k,kind='pivot',row=v) for k,v in templates.items()]
 (base/'degree6_normalform_templates.json').write_text(json.dumps(rules,separators=(',',':')))
 result=dict(reference_N=6,potential_degree=6,variables=A.shape[1],constraint_rows=A.shape[0],rank=len(ps),dimension=A.shape[1]-len(ps),templates=len(templates),template_boundary_cutoff=5,C_coefficient_radius=rC,source_full_geometry_radius=rWgeom,A_coefficient_radius=rA,A_full_geometry_radius=rAgeom,axis_state_coverage=coverage,exact_WA_minus_C_nnz=0,exact_A_minus_ApC_nnz=0,pivot_identity_nnz=0,integer_product_bound=bound,passed=True,seconds=time.time()-st)
 (base/'uniform_normalform_verified.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':run()
