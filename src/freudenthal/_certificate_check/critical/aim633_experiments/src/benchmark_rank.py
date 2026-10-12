from geometry import *
from algebra import *
import json,time
for N,d in [(1,2),(2,3),(2,5),(2,6),(3,6)]:
 mesh=Mesh.build(N);sp=BernsteinSpace.build(mesh,d);A,_=sp.curl_constraints()
 # interior to boundary entities, in physical coordinate order
 order=sorted(range(A.shape[1]),key=lambda c:(-len(sp.node_entity[c//3]),tuple(sp.nodes[c//3]),c%3))
 priority=np.argsort(order)
 t=time.time();e=sparse_mod_rank(A,priority=priority)
 print(json.dumps(dict(N=N,d=d,rank=e.rank,nullity=A.shape[1]-e.rank,max_nnz=e.max_nnz,ops=e.operations,seconds=time.time()-t)),flush=True)
