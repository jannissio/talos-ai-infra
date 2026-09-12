"""Bounded development evaluation, preserving every result and model hash."""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from simulation_lab.storage import require_space

def run(a):
    out=Path(a.output);checkpoint=Path(a.checkpoint)
    if out.exists():raise FileExistsError(out)
    require_space(out,32*1024**2);out.mkdir(parents=True)
    frozen={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in checkpoint.iterdir() if f.is_file()}
    cases=[('seed-'+str(s),['--seed',str(s)]) for s in a.seeds]
    cases += [(Path(e).name,['--episode',e]) for e in a.episodes]
    report={'scope':'Development evaluation. All listed seeds become exposed, never fresh holdout again.','checkpoint':checkpoint.as_posix(),'checkpoint_hashes':frozen,'cases':[n for n,_ in cases],'results':[]}
    (out/'protocol.json').write_text(json.dumps(report,indent=2))
    for name,args in cases:
        require_space(out,8*1024**2)
        if any(hashlib.sha256((checkpoint/n).read_bytes()).hexdigest()!=h for n,h in frozen.items()):raise RuntimeError('Checkpoint changed')
        began=time.perf_counter()
        with (out/(name+'.log')).open('w') as log:
            p=subprocess.run([sys.executable,'scripts/verify_live_policy.py','--checkpoint',str(checkpoint),'--output',str(out/(name+'.json')),*args],stdout=log,stderr=subprocess.STDOUT,timeout=300)
        if (out/(name+'.json')).is_file():
            r=json.loads((out/(name+'.json')).read_text());row={'case':name,'status':r['status'],'message':r['message'],'elapsed_s':r['elapsed_s'],'metrics':r['metrics']}
        else:row={'case':name,'status':'execution_error','exit_code':p.returncode}
        report['results'].append(row);report['successes']=sum(r['status']=='succeeded' for r in report['results']);report['complete']=len(report['results'])==len(cases)
        (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps({'case':name,'status':row['status'],'wall_seconds':time.perf_counter()-began,'successes':report['successes']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--seeds',nargs='*',type=int,default=[]);p.add_argument('--episodes',nargs='*',default=[]);run(p.parse_args())
