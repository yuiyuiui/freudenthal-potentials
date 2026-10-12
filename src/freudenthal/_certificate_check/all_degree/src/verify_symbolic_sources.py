"""Independent integer verifier of universal forward row-space certificates.

Reads the finite data, reconstructs every source equation geometrically, and
proves its admissibility throughout an unbounded degree-parameter chamber.
No reference nullspace, modular rank, or floating point inverse is used.
"""
from pathlib import Path
from itertools import permutations,product,combinations
from functools import lru_cache
from collections import defaultdict,Counter
import json,time,hashlib
BASE=Path(__file__).resolve().parents[1]
PERMS=list(permutations(range(3)))
def freeze(x):return tuple(freeze(y) for y in x) if isinstance(x,list) else x

def aff_min(coeff,const,gap):
    ids=[i for i,g in enumerate(gap) if g==3]
    if any(coeff[i]<0 for i in ids):return None
    v=const+sum(c*g for c,g in zip(coeff,gap))
    extra=max(0,7-sum(gap))
    if extra:
        assert ids
        v+=extra*min(coeff[i] for i in ids)
    return v

def aff_zero(coeff,const,gap):
    return all(coeff[i]==0 for i,g in enumerate(gap) if g==3) and const+sum(c*g for c,g in zip(coeff,gap))==0

@lru_cache(None)
def tet_data(t):
    assert len(set(t))==4
    pm=[]
    for v,w in zip(t,t[1:]):
        delta=tuple(y-x for x,y in zip(v,w))
        assert sorted(delta)==[0,0,1]
        pm.append(delta.index(1))
    assert len(set(pm))==3
    grad=[[0]*3 for _ in range(4)]
    grad[0][pm[0]]=-1
    grad[1][pm[0]]=1;grad[1][pm[1]]=-1
    grad[2][pm[1]]=1;grad[2][pm[2]]=-1
    grad[3][pm[2]]=1
    ell=[1+t[0][pm[0]],t[0][pm[1]]-t[0][pm[0]],t[0][pm[2]]-t[0][pm[1]],-t[0][pm[2]]]
    for i in range(4):
        for j,v in enumerate(t):assert ell[i]+sum(g*x for g,x in zip(grad[i],v))==int(i==j)
    return ell,grad

def entity(pm,g):
    u=[0,0,0];ans=[]
    for i in range(4):
        if g[i]>0:ans.append(tuple(u))
        if i<3:u[pm[i]]=1
    return ans

def expected_keys():
    E=set()
    for bc in product(range(3),repeat=3):
      for pm in PERMS:
       for g in product(range(4),repeat=4):
        if 3 not in g and sum(g)<7:continue
        if any(g[i+1]==0 and pm[i]>pm[i+1] for i in range(2)):continue
        if g[0]==0:
            top=[]
            for i in range(3):
                top.append(pm[i])
                if g[i+1]>0:break
            if any(bc[j]!=2 for j in top):continue
        for co in range(3):E.add((bc,pm,g,co))
    return E

def encode(k):
    bc,pm,g,c=k;b=bc[0]*9+bc[1]*3+bc[2];p=PERMS.index(pm);gg=g[0]*64+g[1]*16+g[2]*4+g[3]
    return ((b*6+p)*256+gg)*3+c

def cross(g):return ((0,-g[2],g[1]),(g[2],0,-g[0]),(-g[1],g[0],0))

def verify_sources(rules):
    nterms=0;nineq=0;neq=0;maxw=0;maxc=0
    for r in rules:
      if r['kind']!='pivot':continue
      bc,pm,gap,component=freeze(r['key'])
      R=[[0]*4 for _ in range(3)]
      for i,ax in enumerate(pm):
        for j in range(i+1,4):R[ax][j]=1
      result=defaultdict(int)
      for s in r['sources']:
        tets=freeze(s['tets']);face=set(freeze(s['face']));delta=tuple(s['delta']);comp=int(s['component']);weight=int(s['weight'])
        assert len(tets)==2 and set(tets[0])&set(tets[1])==face and len(face)==3
        assert comp in range(3) and weight!=0
        nterms+=1;maxw=max(maxw,abs(weight))
        for ti,t in enumerate(tets):
          ell,grad=tet_data(t)
          for ax in range(3):
            L=0 if bc[ax]==0 else -1;U=1 if bc[ax]==2 else 2
            assert all(L<=v[ax]<=U for v in t),(r['key'],'geometry',t,ax)
          opp=[i for i,v in enumerate(t) if v not in face];assert len(opp)==1;opp=opp[0]
          for i in range(4):
            co=[ell[i]+sum(grad[i][ax]*R[ax][j] for ax in range(3)) for j in range(4)]
            const=-ell[i]+sum(grad[i][ax]*delta[ax] for ax in range(3))
            if i==opp:
                assert aff_zero(co,const,gap),(r['key'],'face plane',co,const);neq+=1
            else:
                m=aff_min(co,const,gap);assert m is not None and m>=0,(r['key'],'negative face barycentric',co,const,m);nineq+=1
            sign=1 if ti==0 else -1
            for c,z in enumerate(cross(grad[i])[comp]):
                if z:result[(tuple(delta[ax]+t[i][ax] for ax in range(3)),c)]+=weight*sign*z
      result={k:v for k,v in result.items() if v}
      row={(tuple(q),int(c)):int(z) for q,c,z in r['row']}
      assert len(row)==len(r['row']) and all(row.values()) and result==row,(r['key'],'coefficient identity')
      maxc=max(maxc,max(abs(v) for v in row.values()))
    return dict(source_terms=nterms,universal_nonnegative_barycentric_inequalities=nineq,universal_face_plane_identities=neq,max_source_multiplier=maxw,max_normalform_coefficient=maxc)

def verify_integer_export(rules):
    lines=iter((BASE/'results/rules_integer.txt').read_text().splitlines());assert int(next(lines))==len(rules)
    for r in rules:
        k=freeze(r['key']);bc,pm,g,c=k;row=r.get('row',[])
        mask=sum(1<<(v[0]*4+v[1]*2+v[2]) for v in entity(pm,g))
        expect=[encode(k),int(r['kind']=='pivot'),*bc,*pm,*g,c,mask,len(row)]
        assert list(map(int,next(lines).split()))==expect
        for q,co,z in row:assert list(map(int,next(lines).split()))==[*q,co,z]
    assert next(lines,None) is None

def profile_key(N,origin,face,tets):
    width=tuple(max(v[j] for t in tets for v in t) for j in range(3))
    assert all(w in (1,2) for w in width)
    assert all(min(v[j] for t in tets for v in t)==0 for j in range(3))
    pro=tuple((int(origin[j]==0),min(N-origin[j]-width[j],2)) for j in range(3))
    assert all(z>=0 for _,z in pro)
    return (tuple(sorted(face)),tuple(sorted(tets)),pro)

def verify_face_coverage():
    expected=set()
    for N in range(2,7):
        faces=defaultdict(list)
        for a in product(range(N),repeat=3):
            for p in PERMS:
                cur=list(a);tet=[tuple(cur)]
                for j in p:cur=cur.copy();cur[j]+=1;tet.append(tuple(cur))
                tet=tuple(tet)
                for f in combinations(tet,3):faces[tuple(sorted(f))].append(tet)
        for f,tets in faces.items():
            assert len(tets) in (1,2)
            if len(tets)!=2:continue
            origin=tuple(min(v[j] for t in tets for v in t) for j in range(3))
            fu=tuple(tuple(v[j]-origin[j] for j in range(3)) for v in f)
            tu=tuple(tuple(tuple(v[j]-origin[j] for j in range(3)) for v in t) for t in tets)
            expected.add(profile_key(N,origin,fu,tu))
    lines=iter((BASE/'results/face_profiles.txt').read_text().splitlines());n=int(next(lines));actual=set()
    for _ in range(n):
        h=list(map(int,next(lines).split()));N=h[0];origin=tuple(h[1:]);assert N in range(2,7)
        f=tuple(tuple(map(int,next(lines).split())) for _ in range(3))
        t=tuple(tuple(tuple(map(int,next(lines).split())) for _ in range(4)) for _ in range(2))
        for tet in t:tet_data(tet)
        assert set(t[0])&set(t[1])==set(f) and len(set(f))==3
        actual.add(profile_key(N,origin,f,t))
    assert actual==expected and len(actual)==n and next(lines,None) is None
    return n

def main():
    if not __debug__:raise RuntimeError("Verification requires assertions enabled; do not run Python with -O.")
    st=time.time();p=BASE/'results/all_degree_candidate_rules.json';rules=json.loads(p.read_text());keys=[freeze(r['key']) for r in rules]
    assert len(keys)==len(set(keys)) and set(keys)==expected_keys()
    verify_integer_export(rules);out=verify_sources(rules);nf=verify_face_coverage()
    out.update(passed=True,rules=len(rules),pivot_rules=sum(r['kind']=='pivot' for r in rules),face_profiles=nf,scope='all N>=2, all potential degrees d>=7',source_data_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),arithmetic='arbitrary-precision Python integers and exact affine lower bounds',seconds=time.time()-st)
    (BASE/'results/universal_sources_verified.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
if __name__=='__main__':main()
