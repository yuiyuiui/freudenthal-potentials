from tracked_rref import *
from fractions import Fraction

def key(q,c,N,d,L):return (tuple((int(x)%d,min(int(x),L+1),min(N*d-int(x),L+1)) for x in q),int(c))
def signature(row,q,nodes):
 row=row.tocsr();pcoeff=None
 return tuple(sorted((tuple(int(x) for x in nodes[j//3]-q),int(j%3),int(z)) for j,z in zip(row.indices,row.data)))

def load(N,d):
 root=Path(__file__).resolve().parents[1]/'results'/f'rref_N{N}_d{d}'
 s=BernsteinSpace.build(Mesh.build(N),d);C=sparse.load_npz(root/'C.npz');piv=np.load(root/'pivots.npy');pset=set(piv)
 sig={}
 for i,p in enumerate(piv):sig[int(p)]=signature(C.getrow(i),s.nodes[p//3],s.nodes)
 return s,sig
if __name__=='__main__':
 data=[(N,*load(N,6)) for N in [3,4,6]]
 for L in range(0,13):
  D={};conflicts=0;example=None
  for N,s,sigs in data:
   for qid,q in enumerate(s.nodes):
    for c in range(3):
     k=key(q,c,N,6,L);v=sigs.get(3*qid+c,('free',))
     if k in D and D[k]!=v:
      conflicts+=1
      if example is None:example=(N,q.tolist(),c)
     else:D[k]=v
  print(json.dumps(dict(L=L,templates=len(D),conflicts=conflicts,example=example)),flush=True)
