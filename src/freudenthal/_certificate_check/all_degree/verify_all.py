"""Verify the complete all-N, all-k proof package, including its k=5 dependency."""
from pathlib import Path
import argparse,json,os,shlex,shutil,subprocess,sys,time,zipfile
ROOT=Path(__file__).resolve().parent

def main():
    if not __debug__:raise RuntimeError('Do not run the verifier with Python -O.')
    ap=argparse.ArgumentParser();ap.add_argument('--skip-critical',action='store_true',help='Verify only the NEW d>=7 theorem, not the full proof dependency.');a=ap.parse_args()
    start=time.time();done=[];env=os.environ.copy();env.pop('PYTHONOPTIMIZE',None)
    def run(args,cwd=ROOT):
        print('RUN:', ' '.join(map(str,args)),flush=True)
        subprocess.run(list(map(str,args)),cwd=cwd,env=env,check=True)
    run([sys.executable,'src/verify_symbolic_sources.py']);done.append('all-degree source certificates, coverage, face-profile coverage, and integer export')
    cc=shlex.split(os.environ.get('CXX','c++'))
    if not shutil.which(cc[0]):raise RuntimeError('A C++17 compiler is required; set CXX to its command.')
    build=ROOT/'build';build.mkdir(exist_ok=True);exe=build/'exact_parameter_check'
    run([*cc,'-std=c++17','-O2','-Wall','-Wextra','-Werror','src/exact_parameter_check.cpp','-o',exe])
    run([exe,ROOT/'results','all','5']);done.append('unbounded parameter chambers: pivot identity, vertex support, reverse inclusion')
    run([sys.executable,'src/dimension_all_degrees.py']);done.append('exact all-degree dimension count')
    for N,d in [(2,20),(4,7),(3,13)]:
        run([sys.executable,'src/construct_basis.py','--N',str(N),'--degree',str(d)]);
    done.append('independent original-constraint reassembly at (N,d)=(2,20),(4,7),(3,13)')
    if not a.skip_critical:
        archive=ROOT/'dependencies/aim633_k5_uniform_proof_2026-10-10.zip'
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                target=(ROOT/'dependencies'/name).resolve()
                if not target.is_relative_to((ROOT/'dependencies').resolve()):raise ValueError('Unsafe dependency archive path')
            z.extractall(ROOT/'dependencies')
        dep=ROOT/'dependencies/aim633_k5_uniform_proof'
        run([sys.executable,'verify_all.py'],cwd=dep);done.append('complete critical d=6 dependency, including all-N transfer certificates and small N cases')
    out=dict(passed=True,full_original_target_verified=not a.skip_critical,scope='all N>=1,k>=5' if not a.skip_critical else 'new theorem N>=2,d>=7 only; N=1 is elementary',completed=done,external_peer_review=False,formal_proof_assistant=False,seconds=time.time()-start)
    (ROOT/'results/final_verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
if __name__=='__main__':main()
