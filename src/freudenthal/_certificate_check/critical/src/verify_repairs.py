"""Independent exact verification of saved local repairs; no nullspace search."""
from tracked_rref import *
from certify_repairs import kernel_from_C

def run():
 st=time.time();base=Path(__file__).resolve().parents[1]/'results';src=base/'rref_N6_d6';rep=base/'repairs_N6_d6';N=d=6
 s=BernsteinSpace.build(Mesh.build(N),d);A,_=s.curl_constraints();C=sparse.load_npz(src/'C.npz');ps=np.load(src/'pivots.npy');G=sparse.load_npz(rep/'G.npz');free=np.load(rep/'free.npy');owner=np.load(rep/'local_owner.npy');parts=sparse.load_npz(rep/'repair_parts.npz');rmeta=json.load(open(rep/'repairs.json'));pmeta=json.load(open(rep/'part_metadata.json'))
 G2,free2=kernel_from_C(s,C,ps);assert np.array_equal(free,free2);R=G-G2;R.eliminate_zeros();assert R.nnz==0
 boundA=int(np.max(np.abs(A.data)))*int(np.max(np.diff(A.indptr)))
 assert boundA*max(int(np.max(np.abs(G.data))),int(np.max(np.abs(parts.data))))<np.iinfo(np.int64).max
 for B in (G,parts):
  R=A@B.T;R.eliminate_zeros();assert R.nnz==0
 for i,v in enumerate(owner):
  if v<0:continue
  cc=G.indices[G.indptr[i]:G.indptr[i+1]]
  assert all(int(v) in s.node_owner_vertices[int(c)//3] for c in cc)
  q=s.nodes[free[i]//3];assert np.max(np.abs(d*s.mesh.vertices[int(v)]-q))<=6;geo=np.concatenate([s.mesh.vertices[s.mesh.tets[t]]*d for t in s.mesh.stars[int(v)]])
  assert np.max(np.abs(geo-q))<=12
 for j,info in enumerate(pmeta):
  v=info['vertex'];i=info['free_index'];cc=parts.indices[parts.indptr[j]:parts.indptr[j+1]]
  assert all(v in s.node_owner_vertices[int(c)//3] for c in cc)
  q=s.nodes[free[i]//3];assert np.max(np.abs(d*s.mesh.vertices[int(v)]-q))<=6;geo=np.concatenate([s.mesh.vertices[s.mesh.tets[t]]*d for t in s.mesh.stars[v]])
  assert np.max(np.abs(geo-q))<=12
 seen=set()
 for r in rmeta:
  i=r['free_index'];assert owner[i]<0 and i not in seen;seen.add(i);ids=r['part_indices'];assert all(pmeta[j]['free_index']==i and pmeta[j]['denominator']==r['denominator'] for j in ids)
  row=sparse.csr_matrix(parts[ids].sum(axis=0))-r['denominator']*G.getrow(i);row.eliminate_zeros();assert row.nnz==0
 assert seen==set(np.flatnonzero(owner<0))
 out=dict(dimension=G.shape[0],already_vertex_local=int(sum(owner>=0)),nonlocal_vectors=len(seen),repair_parts=parts.shape[0],max_parts_per_vector=max(len(r['part_indices']) for r in rmeta),max_denominator=max(r['denominator'] for r in rmeta),max_abs_part_coefficient=int(np.max(np.abs(parts.data))),vertex_distance_from_free_coefficient=6,full_star_geometry_radius=12,exact_membership_and_sums=True,passed=True,seconds=time.time()-st)
 (base/'independent_repairs_verified.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
if __name__=='__main__':run()
