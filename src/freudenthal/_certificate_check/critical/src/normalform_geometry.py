from tracked_rref import *
from template_signatures import key,signature
from collections import Counter
N=6;d=6;root=Path(__file__).resolve().parents[1]/'results'/f'rref_N{N}_d{d}'
s=BernsteinSpace.build(Mesh.build(N),d);A,meta=s.curl_constraints();C=sparse.load_npz(root/'C.npz');W=sparse.load_npz(root/'W.npz');ps=np.load(root/'pivots.npy');diag=C[np.arange(len(ps)),ps].A.ravel();print('diag',Counter(diag),flush=True)
# full geometric envelope for each source equation
alo=[];ahi=[];rownoderad=0;rowgeomrad=0
for i,(face,beta,comp) in enumerate(meta):
 inc=s.mesh.faces[face];vv=np.concatenate([s.mesh.vertices[s.mesh.tets[t]] for t,_ in inc])*d
 alo.append(vv.min(axis=0));ahi.append(vv.max(axis=0))
 cc=A.indices[A.indptr[i]:A.indptr[i+1]]
 if len(cc):
  qq=s.nodes[cc//3];q=qq[0]
  rownoderad=max(rownoderad,int(np.abs(qq-q).max()));rowgeomrad=max(rowgeomrad,int(np.abs(vv-q).max()))
alo=np.array(alo);ahi=np.array(ahi);rgeom=0;hist=Counter()
for i,c in enumerate(ps):
 q=s.nodes[c//3];rr=W.indices[W.indptr[i]:W.indptr[i+1]]
 r=int(max(np.abs(alo[rr]-q).max(),np.abs(ahi[rr]-q).max()));rgeom=max(rgeom,r);hist[r]+=1
print(json.dumps(dict(N=N,d=d,source_full_geometry_radius=rgeom,source_radius_hist=dict(hist),original_row_node_radius=rownoderad,original_row_geom_radius=rowgeomrad)),flush=True)
