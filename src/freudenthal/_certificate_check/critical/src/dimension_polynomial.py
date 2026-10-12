import json,sympy as sy
from pathlib import Path
N=sy.symbols('N',integer=True,positive=True);base=Path(__file__).resolve().parents[1]/'results';rules=json.load(open(base/'degree6_normalform_templates.json'));counts={}
for r in rules:
 if r['kind']!='free':continue
 a=r['key'][0];w=1
 for residue,left,right in a:
  if left<=5 or right<=5:f=1
  else:f=N-1 if residue==0 else N-2
  w*=f
 ex=sy.Poly(w,N)
 for (power,),c in ex.terms():counts[power]=counts.get(power,0)+c
p=sy.expand(sum(c*N**j for j,c in counts.items()));assert p==300*N**3+390*N**2+99*N+6
result=dict(exact_count_polynomial=str(p),reference_value=int(p.subs(N,6)),derived_from_template_count=True)
(base/'dimension_polynomial.json').write_text(json.dumps(result,indent=2));print(result)
