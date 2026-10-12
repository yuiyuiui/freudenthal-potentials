from tracked_rref import *

def run_repair(N,d):
 st=time.time();m=Mesh.build(N);s=BernsteinSpace.build(m,d);A,_=s.curl_constraints();pr=ordering(s);e=sparse_mod_rank(A,priority=pr);free=[j for j in range(A.shape[1]) if j not in e.rows];G=e.nullspace_rows();localcache={};groups={};out=[]
 for f,g in zip(free,G):
  owners=set(range(len(m.vertices)))
  for c in g:owners.intersection_update(s.node_owner_vertices[c//3])
  if owners:continue
  vs=s.node_entity[f//3]
  if vs not in groups:
   E=SparseEchelon(A.shape[1],priority=pr)
   for v in vs:
    if v not in localcache:
     K=restricted_kernel(A,vector_columns(s.vertex_nodes(v)),pr);B=K.basis.tocoo();localcache[v]=sparse.coo_matrix((B.data,(B.row,K.columns[B.col])),shape=(B.shape[0],A.shape[1])).tocsr()
    E.add_matrix_rows(localcache[v])
   groups[vs]=E
  E=groups[vs];r=dict(g)
  while r:
   c=min(r,key=E.pr.__getitem__);z=r[c]
   if c not in E.rows:break
   r.pop(c);axpy(r,E.rows[c],-z,1000003,c)
  out.append(dict(f=f,q=s.nodes[f//3].tolist(),vs=list(vs),passed=not r,residual_nnz=len(r)))
 print(json.dumps(dict(N=N,d=d,nonlocal_count=len(out),failed=sum(not r['passed'] for r in out),examples=[r for r in out if not r['passed']][:8],seconds=time.time()-st)),flush=True)
 return out
if __name__=='__main__':
 for N in [2,3,4]:run_repair(N,6)
