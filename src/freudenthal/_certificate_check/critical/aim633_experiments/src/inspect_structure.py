from geometry import *
import time,json
from collections import Counter
for N,d in [(1,2),(2,3),(2,5),(2,6),(3,6),(4,6),(2,7),(2,8),(2,10)]:
 t=time.time();mesh=Mesh.build(N);sp=BernsteinSpace.build(mesh,d);A,meta=sp.curl_constraints();blocks,free=matrix_components(A)
 print(json.dumps(dict(N=N,d=d,nscalar=sp.nscalar,shape=A.shape,nnz=A.nnz,blocks=Counter((len(r),len(c)) for r,c in blocks).__str__(),free=len(free),seconds=round(time.time()-t,3))),flush=True)
