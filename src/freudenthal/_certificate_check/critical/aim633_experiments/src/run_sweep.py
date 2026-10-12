from experiment_core import *
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import traceback

def one(case):
    N,d,p=case
    try:
        r=run_vertex(N,d,p)
        fn=Path(__file__).resolve().parents[1]/'results'/'sweep'/f'N{N}_d{d}_p{p}.json'
        fn.parent.mkdir(exist_ok=True,parents=True);fn.write_text(json.dumps(r,indent=2))
        return {k:v for k,v in r.items() if k!='patches'}
    except Exception as exc:
        return dict(N=N,potential_degree=d,prime=p,error=str(exc),traceback=traceback.format_exc())
if __name__=='__main__':
    cases=[(1,d,1000003) for d in range(1,11)]
    cases += [(2,d,1000003) for d in range(1,11)]
    cases += [(3,d,1000003) for d in range(2,10)]
    cases += [(4,d,1000003) for d in range(3,9)]
    cases += [(5,d,1000003) for d in range(5,8)]
    cases += [(6,6,1000003)]
    cases += [(N,d,1000033) for N,d in [(2,4),(2,5),(2,6),(3,5),(3,6),(3,7),(4,5),(4,6),(5,6),(6,6)]]
    logfile=Path(__file__).resolve().parents[1]/'results'/'sweep_summary.jsonl'
    with logfile.open('w') as f,ProcessPoolExecutor(max_workers=2) as pool:
        fs={pool.submit(one,c):c for c in cases}
        for fu in as_completed(fs):
            s=json.dumps(fu.result());print(s,flush=True);f.write(s+'\n');f.flush()
