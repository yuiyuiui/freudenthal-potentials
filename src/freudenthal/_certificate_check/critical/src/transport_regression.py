"""Out-of-reference exact tests of the universal construction.
Not used as a substitute for the analytic finite-window transfer lemma.
"""
from tracked_rref import *
from template_signatures import key,signature
from certify_repairs import kernel_from_C

def freeze(x):return tuple(freeze(a) for a in x) if isinstance(x,list) else x

def run(N):
 start=time.time();base=Path(__file__).resolve().parents[1]/'results';d=6
 rules={freeze(r['key']):r for r in json.load(open(base/'degree6_normalform_templates.json'))}
 s=BernsteinSpace.build(Mesh.build(N),d);A,_=s.curl_constraints();lookup={tuple(int(x) for x in q):i for i,q in enumerate(s.nodes)}
 cr=[];cc=[];cv=[];ps=[]
 for qid,q in enumerate(s.nodes):
  for comp in range(3):
   rule=rules[key(q,comp,N,d,5)]
   if rule['kind']=='free':continue
   ri=len(ps);ps.append(3*qid+comp)
   for offset,c,z in rule['row']:
    coord=tuple(int(x) for x in q+np.array(offset));assert coord in lookup
    cr.append(ri);cc.append(3*lookup[coord]+c);cv.append(z)
 C=sparse.coo_matrix((np.array(cv,dtype=np.int64),(cr,cc)),shape=(len(ps),A.shape[1])).tocsr();ps=np.array(ps,dtype=np.int64)
 R=C[:,ps]-sparse.eye(len(ps),dtype=np.int64,format='csr');R.eliminate_zeros();assert R.nnz==0
 R=A-A[:,ps]@C;R.eliminate_zeros();assert R.nnz==0
 G,free=kernel_from_C(s,C,ps)
 ref=base/'repairs_N6_d6';s0=BernsteinSpace.build(Mesh.build(6),6);G0=sparse.load_npz(ref/'G.npz');f0=np.load(ref/'free.npy');owner=np.load(ref/'local_owner.npy');P0=sparse.load_npz(ref/'repair_parts.npz');rmeta={r['free_index']:r for r in json.load(open(ref/'repairs.json'))};pmeta=json.load(open(ref/'part_metadata.json'))
 representatives={}
 for i,f in enumerate(f0):representatives.setdefault(key(s0.nodes[f//3],f%3,6,6,12),i)
 repairparts=[];nonlocal_count=0
 for i,f in enumerate(free):
  q=s.nodes[f//3];j=representatives[key(q,f%3,N,6,12)];q0=s0.nodes[f0[j]//3];delta=q-q0;assert np.all(delta%6==0)
  assert signature(G.getrow(i),q,s.nodes)==signature(G0.getrow(j),q0,s0.nodes)
  if owner[j]>=0:
   vcoord=s0.mesh.vertices[owner[j]]+delta//6;v=int(np.ravel_multi_index(tuple(vcoord),(N+1,N+1,N+1)))
   assert all(v in s.node_owner_vertices[int(c)//3] for c in G.indices[G.indptr[i]:G.indptr[i+1]])
  else:
   nonlocal_count+=1;r=rmeta[j];parts=[]
   for pi in r['part_indices']:
    row=P0.getrow(pi);vcoord=s0.mesh.vertices[pmeta[pi]['vertex']]+delta//6;assert np.all(vcoord>=0) and np.all(vcoord<=N);v=int(np.ravel_multi_index(tuple(vcoord),(N+1,N+1,N+1)))
    vals={3*lookup[tuple(int(x) for x in s0.nodes[c//3]+delta)]+int(c%3):int(z) for c,z in zip(row.indices,row.data)}
    assert all(v in s.node_owner_vertices[int(c)//3] for c in vals)
    part=coo_from_rows([vals],A.shape[1]);parts.append(part);repairparts.append(part)
   R=sum(parts,sparse.csr_matrix((1,A.shape[1]),dtype=np.int64))-r['denominator']*G.getrow(i);R.eliminate_zeros();assert R.nnz==0
 if repairparts:
  B=sparse.vstack(repairparts,format='csr');R=A@B.T;R.eliminate_zeros();assert R.nnz==0
 result=dict(N=N,d=d,variables=A.shape[1],dimension=G.shape[0],predicted_dimension=300*N**3+390*N**2+99*N+6,transported_nonlocal_repairs=nonlocal_count,exact_constraint_normalform=True,exact_all_kernel_template_matches=True,exact_local_support_and_repairs=True,passed=True,seconds=time.time()-start)
 (base/f'transport_N{N}_verified.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':
 for N in [5,7]:run(N)
