"""Fixed-checkpoint, predeclared randomized live-controller evaluation.

Preserve every outcome. This produces compact physical state recordings, not new
training data. Once evaluated, these seeds must not be called unseen again.
"""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from simulation_lab.storage import require_space

ROOT=Path(__file__).resolve().parents[1]

def hashes(checkpoint):
    files=list(Path(checkpoint).resolve().glob('*'))
    files += [ROOT/p for p in ['simulation_lab/learned_task.py','simulation_lab/primitive_policy.py','simulation_lab/bottle_vision.py','simulation_lab/scene.py','simulation_lab/dinner.py','simulation_lab/policy_control.py','simulation_lab/dinner_autonomy.py','scripts/run_command_demo.py']]
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}

def run(a):
    out=Path(a.output);seeds=list(range(a.first_seed,a.first_seed+a.count))
    if out.exists():raise FileExistsError(out)
    require_space(out,a.count*32*1024**2);out.mkdir(parents=True)
    frozen=hashes(a.checkpoint);protocol={'seeds':seeds,'instruction':'place the bottle','mode':'learned_bottle','checkpoint':Path(a.checkpoint).as_posix(),
        'scope':'Fresh scene seeds at declaration; experimental bottle task only, not a full-table success claim',
        'source_hashes':frozen,'physics_timeout_s':65,'preserve_all_outcomes':True}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));rows=[]
    for seed in seeds:
        require_space(out,32*1024**2)
        if hashes(a.checkpoint)!=frozen:raise RuntimeError('Checkpoint/source changed during evaluation.')
        folder=out/str(seed);started=time.perf_counter()
        cmd=[sys.executable,str(ROOT/'scripts/run_command_demo.py'),'--text','place the bottle','--mode','learned_bottle',
             '--seed',str(seed),'--seconds','65','--output',str(folder),'--checkpoint',str(a.checkpoint)]
        with (out/f'{seed}.log').open('w') as log:
            p=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        if (folder/'result.json').is_file():
            r=json.loads((folder/'result.json').read_text());task=r['task']
            row={'seed':seed,'status':r['status'],'message':task['message'],'simulated_seconds':r['simulated_seconds'],'metrics':task['metrics']}
        else:row={'seed':seed,'status':'execution_error','exit_code':p.returncode}
        row['wall_seconds']=time.perf_counter()-started;rows.append(row)
        report={'protocol':protocol,'trials':rows,'successes':sum(r['status']=='succeeded' for r in rows),'completed':len(rows),
                'planned':len(seeds),'complete':len(rows)==len(seeds),'all_frozen_hashes_match':hashes(a.checkpoint)==frozen}
        (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(row),flush=True)
    if hashes(a.checkpoint)!=frozen:raise RuntimeError('Checkpoint/source changed during evaluation.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--first-seed',type=int,default=2026091201);p.add_argument('--count',type=int,default=10)
    p.add_argument('--checkpoint',default='models/bottle_primitive')
    a=p.parse_args()
    if not 1<=a.count<=100:p.error('count must be in [1,100]')
    run(a)
