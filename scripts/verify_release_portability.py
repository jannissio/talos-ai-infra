"""Reproduce an exposed candidate workflow from an isolated minimal checkout."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.release_experiment import read,repository_path,sha,space,write_json


def run(args):
    p = read(args.protocol)
    base,evidence = repository_path(p['raw_root']),repository_path(p['evidence_package'])
    target = base/'portable-checkout'
    restore_log,trial_log,result_path = (base/name for name in ('portability-restore.log','portability-trial.log','portability.json'))
    if any(path.exists() for path in (target,restore_log,trial_log,result_path)):
        raise FileExistsError('All portable-checkout, result and console-log paths must be unused.')
    summary = read(evidence/'development/summary.json')
    match = next((r for r in summary['rows'] if r['seed']==args.seed and r['preset']==args.preset and r['controller']=='candidate'),None)
    if not match or not match['passed']:
        raise ValueError('Choose an already-exposed passing candidate case for portability, never a new selection case.')
    frozen = read(evidence/'development/frozen-inputs.json')
    manifest = read(evidence/'package-manifest.json')
    files = {r['file'] for r in manifest['files']}
    files.update(frozen['inputs']['source_sha256'])
    files.update(frozen['inputs']['asset_sha256'])
    for selection in frozen['inputs']['suites'].values():
        files.update(name for name in selection['files_sha256'] if not name.startswith('.run/'))
    protocol_name = args.protocol.resolve().relative_to(ROOT).as_posix()
    files.update([protocol_name,'scripts/package_release_workflows.py','scripts/release_experiment.py',
                  (evidence/'package-manifest.json').relative_to(ROOT).as_posix(),p['training_package']+'/input-replay.json'])
    copy_bytes = sum((ROOT/name).stat().st_size for name in files)
    preflight = space(p,target,copy_bytes+16*1024**2)
    target.mkdir(parents=True)
    for name in sorted(files):
        payload = (ROOT/name).read_bytes()
        out = target/name
        space(p,out,len(payload)+1024**2)
        out.parent.mkdir(parents=True,exist_ok=True)
        with out.open('xb') as stream:stream.write(payload)
    with restore_log.open('x',encoding='utf-8') as stream:
        restored = subprocess.run([sys.executable,str(target/'scripts/package_release_workflows.py'),
                                  '--protocol',protocol_name,'--restore-evaluation-inputs'],
                                  cwd=target,stdout=stream,stderr=subprocess.STDOUT)
    if restored.returncode:
        result = {'passed':False,'stage':'restore','exit_code':restored.returncode,
                  'scope':'Portability failed; original evidence and all local diagnostic copies/logs are retained.'}
        write_json(result_path,result)
        return result
    out = target/p['raw_root']/'portable-trial'
    space(p,out,12*1024**2)
    try:
        with trial_log.open('x',encoding='utf-8') as stream:
            trial = subprocess.run([sys.executable,str(target/'scripts/evaluate_release_workflows.py'),
                '--protocol',protocol_name,'trial','--freeze',p['evidence_package']+'/development/frozen-inputs.json',
                '--seed',str(args.seed),'--preset',args.preset,'--controller','candidate','--output',str(out)],
                cwd=target,stdout=stream,stderr=subprocess.STDOUT,timeout=900)
    except subprocess.TimeoutExpired:
        result = {'passed':False,'stage':'physical reproduction','status':'execution_timeout','seed':args.seed,'preset':args.preset,
                  'scope':'Portability timeout, not a new selection trial; partial evidence and logs are retained.'}
        write_json(result_path,result)
        return result
    original = evidence/'development'/f'{args.seed}-{args.preset}-candidate'
    if not (out/'report.json').exists() or not (out/'states.npz').exists():
        result = {'passed':False,'stage':'physical reproduction','status':'missing_physical_output','exit_code':trial.returncode,
                  'seed':args.seed,'preset':args.preset,'scope':'Reproduction failed; partial output and console logs remain preserved.'}
        write_json(result_path,result)
        return result
    before,after = read(original/'report.json'),read(out/'report.json')
    original_metrics = {r['skill']:r['metrics'] for r in before['task']['results']}
    reproduced_metrics = {r['skill']:r['metrics'] for r in after.get('task',{}).get('results',[])}
    with np.load(original/'states.npz',allow_pickle=False) as x,np.load(out/'states.npz',allow_pickle=False) as y:
        identical = set(x.files)==set(y.files) and all(np.array_equal(x[name],y[name]) for name in x.files)
    result = {'scope':'Exposed development reproduction from an isolated minimal checkout; not a new selection or final trial.',
              'files_copied':len(files),'copy_bytes':copy_bytes,'disk_preflight':preflight,'restore_exit':restored.returncode,
              'trial_exit':trial.returncode,'seed':args.seed,'preset':args.preset,'controller':'candidate','status':after['status'],
              'all_physical_metrics_identical':original_metrics==reproduced_metrics,'all_synchronized_arrays_identical':identical,
              'initial_state_identical':before['initial_state_sha256']==after['initial_state_sha256'],
              'original_states_sha256':sha(original/'states.npz'),'reproduced_states_sha256':sha(out/'states.npz'),
              'original_report_sha256':sha(original/'report.json'),'simulation_seconds':after['simulation_seconds'],
              'wall_seconds':after['wall_seconds'],'script_sha256':sha(Path(__file__))}
    result['passed'] = (trial.returncode==0 and result['all_physical_metrics_identical']
                        and result['initial_state_identical'] and identical)
    write_json(result_path,result)
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--preset',choices=('upright','wide_left'),required=True)
    result = run(parser.parse_args())
    print(json.dumps(result),flush=True)
    raise SystemExit(0 if result['passed'] else 1)
