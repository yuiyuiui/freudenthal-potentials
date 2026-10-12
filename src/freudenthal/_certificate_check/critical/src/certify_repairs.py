"""Exact repairs of every nonlocal canonical kernel vector in a reference grid."""
from tracked_rref import *
from collections import defaultdict
from fractions import Fraction
import math

def forward_tracked(B,pr,p=1000003):
 piv={};comb={};pri=[int(x) for x in pr]
 for i in range(B.shape[0]):
  sl=slice(B.indptr[i],B.indptr[i+1]);r={int(j):int(z)%p for j,z in zip(B.indices[sl],B.data[sl]) if z%p};t={i:1}
  while r:
   c=min(r,key=pri.__getitem__);v=r[c]
   if c not in piv:
    z=pow(v,-1,p);piv[c]={j:a*z%p for j,a in r.items()};comb[c]={j:a*z%p for j,a in t.items()};break
   r.pop(c);axpy(r,piv[c],-v,p,c);axpy(t,comb[c],-v,p)
 return piv,comb,pri

def kernel_from_C(s,C,ps):
 n=C.shape[1];free=np.setdiff1d(np.arange(n),ps);cf=C[:,free].tocoo()
 G=sparse.coo_matrix((np.r_[np.ones(len(free),dtype=np.int64),-cf.data],(np.r_[np.arange(len(free)),cf.col],np.r_[free,ps[cf.row]])),shape=(len(free),n)).tocsr()
 return G,free

def run(N=6,d=6):
 start=time.time();base=Path(__file__).resolve().parents[1]/'results';src=base/f'rref_N{N}_d{d}';out=base/f'repairs_N{N}_d{d}';out.mkdir(exist_ok=True)
 m=Mesh.build(N);s=BernsteinSpace.build(m,d);lookup={tuple(int(x) for x in q):j for j,q in enumerate(s.nodes)};A,_=s.curl_constraints();pr=ordering(s);C=sparse.load_npz(src/'C.npz');ps=np.load(src/'pivots.npy');G,free=kernel_from_C(s,C,ps)
 R=A@G.T;R.eliminate_zeros();assert R.nnz==0
 groups=defaultdict(list);local_owner=np.full(len(free),-1,dtype=np.int64)
 for i,f in enumerate(free):
  cols=G.indices[G.indptr[i]:G.indptr[i+1]];owners=set(s.node_owner_vertices[int(f)//3])
  for c in cols:owners.intersection_update(s.node_owner_vertices[int(c)//3])
  if owners:local_owner[i]=min(owners);continue
  vs=s.node_entity[int(f)//3];anchor=m.vertices[vs[0]]
  offsets=tuple(tuple(int(x) for x in m.vertices[v]-anchor) for v in vs)
  context=(offsets,tuple((min(int(x),3),min(N-int(x),3)) for x in anchor))
  groups[context].append((i,int(f),vs,anchor))
 print(json.dumps(dict(event='start',N=N,d=d,dimension=len(free),nonlocal_count=int(sum(local_owner<0)),group_types=len(groups))),flush=True)
 np.save(out/'local_owner.npy',local_owner);np.save(out/'free.npy',free);sparse.save_npz(out/'G.npz',G)
 localcache={};repairs=[];parts=[];metaparts=[]
 for gi,(context,targets) in enumerate(groups.items()):
  i,f,vs,a0=targets[0];Bs=[];owners=[]
  for v in vs:
   if v not in localcache:
    K=restricted_kernel(A,vector_columns(s.vertex_nodes(v)),pr);bb=K.basis.tocoo();localcache[v]=sparse.coo_matrix((bb.data,(bb.row,K.columns[bb.col])),shape=(bb.shape[0],A.shape[1])).tocsr()
   Bs.append(localcache[v]);owners.extend([v]*Bs[-1].shape[0])
  B=sparse.vstack(Bs,format='csr');cc=np.unique(B.indices);labels=[tuple(int(x) for x in s.nodes[c//3]-d*a0)+(int(c%3),) for c in cc];labmap={lab:j for j,lab in enumerate(labels)};Bc=B[:,cc].tocsr();pv,co,pri=forward_tracked(Bc,pr[cc]);owners=np.array(owners)
  for i,f,actual_vs,a in targets:
   gg=G.getrow(i);target={labmap[tuple(int(x) for x in s.nodes[c//3]-d*a)+(int(c%3),)]:int(z) for c,z in zip(gg.indices,gg.data)}
   r={j:z%1000003 for j,z in target.items()};h={}
   while r:
    c=min(r,key=pri.__getitem__);z=r[c]
    if c not in pv:raise AssertionError(('unrepairable',i,f,context,len(r)))
    r.pop(c);axpy(r,pv[c],-z,1000003,c);axpy(h,co[c],z,1000003)
   hf={j:rational_reconstruct(z,1000003) for j,z in h.items()};assert all(z is not None for z in hf.values())
   den=math.lcm(*(x.denominator for x in hf.values()));hh={j:int(z*den) for j,z in hf.items()};hr=coo_from_rows([hh],Bc.shape[0]);check=hr@Bc-den*coo_from_rows([target],Bc.shape[1]);check.eliminate_zeros();assert check.nnz==0
   partids=[]
   for v,av in zip(vs,actual_vs):
    hv={j:z for j,z in hh.items() if owners[j]==v}
    if not hv:continue
    vv=(coo_from_rows([hv],Bc.shape[0])@Bc).tocsr();vv.eliminate_zeros()
    # Transfer the combined local vector from representative to the target.
    row={}
    for j,z in zip(vv.indices,vv.data):
     x=tuple(int(z0) for z0 in np.array(labels[j][:3])+d*a)
     # node lookup below is built lazily.
     c=3*lookup[x]+labels[j][3];row[c]=int(z)
    part=coo_from_rows([row],A.shape[1]);chk=A@part.T;chk.eliminate_zeros();assert chk.nnz==0
    assert all(av in s.node_owner_vertices[c//3] for c in row)
    # All supporting tetrahedra lie in the geometric box of radius 2d about q_f.
    q=s.nodes[f//3]
    geom=np.concatenate([m.vertices[m.tets[t]]*d for t in m.stars[av]])
    assert np.max(np.abs(geom-q))<=2*d
    partids.append(len(parts));parts.append(part);metaparts.append(dict(free_index=i,vertex=int(av),denominator=den))
   total=sum((parts[j] for j in partids),sparse.csr_matrix((1,A.shape[1]),dtype=np.int64))-den*gg;total.eliminate_zeros();assert total.nnz==0
   repairs.append(dict(free_index=i,free_column=f,part_indices=partids,denominator=den))
  print(json.dumps(dict(event='group',index=gi,total=len(groups),targets=len(targets),cumulative=len(repairs),seconds=time.time()-start)),flush=True)
 sparse.save_npz(out/'repair_parts.npz',sparse.vstack(parts,format='csr'))
 (out/'repairs.json').write_text(json.dumps(repairs));(out/'part_metadata.json').write_text(json.dumps(metaparts))
 result=dict(N=N,d=d,dimension=len(free),nonlocal_count=int(sum(local_owner<0)),repaired=len(repairs),repair_parts=len(parts),group_types=len(groups),passed=True,seconds=time.time()-start)
 (out/'verified.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--N',type=int,default=6);ap.add_argument('--degree',type=int,default=6);args=ap.parse_args()
 # Shared coordinate lookup for target transfers.
 ms=BernsteinSpace.build(Mesh.build(args.N),args.degree);lookup={tuple(int(x) for x in q):j for j,q in enumerate(ms.nodes)}
 run(args.N,args.degree)
