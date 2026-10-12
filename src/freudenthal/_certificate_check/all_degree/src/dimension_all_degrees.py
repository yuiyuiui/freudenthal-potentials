"""Exact dimension count from the certified free-rule types, N>=2,d>=7."""
from pathlib import Path
from collections import Counter
import json,math
import sympy as sp
ROOT=Path(__file__).resolve().parents[1]
def main():
 rules=json.loads((ROOT/'results/all_degree_candidate_rules.json').read_text())
 counts=Counter()
 for r in rules:
  if r['kind']=='free':
   bc,pm,g,c=r['key'];counts[(bc.count(1),g.count(3),sum(g))]+=1
 N,d=sp.symbols('N d',integer=True)
 poly=sp.Integer(0)
 for (b,m,s),n in counts.items():
  if m:poly += n*(N-2)**b*sp.Rational(1,math.factorial(m-1))*sp.prod(d-s+j for j in range(1,m))
 expected=3*N**3*(d-1)**2*(d-2)+3*N**2*(d-1)*(5*d-4)+9*N*(2*d-1)+6
 assert sp.expand(poly-expected)==0
 small=[]
 for degree in range(7,13):
  exact=sp.Integer(0)
  for (b,m,s),n in counts.items():
   ways=(math.comb(degree-s+m-1,m-1) if degree>=s else 0) if m else int(degree==s)
   exact+=n*(N-2)**b*ways
  assert sp.expand(exact-expected.subs(d,degree))==0
  small.append(degree)
 # At d>=12 all lower bounds are inactive; m=0 types have total degree<=8.
 assert max(s for b,m,s in counts if m)<=12
 assert max(s for b,m,s in counts if not m)<=8
 out=dict(passed=True,scope='N>=2 and d>=7',formula=str(expected),expanded=str(sp.expand(expected)),finite_transition_degrees_checked=small,eventual_range='all d>=12, by the polynomial identity',arithmetic='exact rational symbolic algebra',count_table=[dict(interior_axes=b,saturated_gaps=m,minimum_degree=s,multiplicity=n) for (b,m,s),n in sorted(counts.items())])
 (ROOT/'results/all_degree_dimension.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='count_table'}),flush=True)
if __name__=='__main__':main()
