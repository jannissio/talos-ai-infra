"""Run all four V2 conditions without excluding invalid starts or refusals."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from simulation_lab.storage import require_space


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    if args.output.exists():raise FileExistsError(args.output)
    protocol=json.loads(args.protocol.read_text());seeds=protocol[args.split+'_seeds']
    if not 1<=args.workers<=3:raise ValueError('Use one to three workers.')
    modes=('nominal','nominal-frozen','push-live','push-frozen')
    source_names=list(json.loads(Path('docs/robotics/experiments/rgb-servo-final-evaluation-v1.json').read_text())['source_sha256'])
    source_names+=['simulation_lab/rgb_servo_routing.py','scripts/evaluate_rgb_servo_routing.py','scripts/evaluate_rgb_servo_routing_suite.py']
    frozen={'protocol':args.protocol.as_posix(),'protocol_sha256':sha(args.protocol),'routing':args.routing.as_posix(),
            'routing_sha256':sha(args.routing),'source_sha256':{name:sha(Path(name)) for name in source_names},
            'model_manifest_sha256':sha(args.model/'experiment.json'),'split':args.split,'seeds':seeds,'conditions':list(modes),
            'ablation':'Initial image frozen separately per leg. Every later leg acquires fresh RGB after release and parking.'}
    if args.split=='evaluation':
        if not args.freeze:raise ValueError('Freeze selection in a new protocol before exposing final scenes.')
        selection=json.loads(args.freeze.read_text())
        for key in ('protocol_sha256','routing_sha256','source_sha256','model_manifest_sha256'):
            if selection[key]!=frozen[key]:raise ValueError('Candidate differs from its final freeze: '+key)
        if selection['evaluation_seeds']!=seeds:raise ValueError('Final seeds differ from the freeze.')
    def used_bytes():
        return sum(p.stat().st_size for d in Path('.run').glob('rgb-servo-v2*') if d.is_dir() for p in d.rglob('*') if p.is_file())
    expected=len(seeds)*len(modes)*3*1024**2
    if used_bytes()+expected>protocol['budget']['max_new_data_mib']*1024**2:
        raise ValueError('Expected writes exceed the V2 data budget; preserve existing data.')
    require_space(args.output,expected);args.output.mkdir(parents=True)
    (args.output/'frozen-inputs.json').write_text(json.dumps(frozen,indent=2)+'\n')
    began=time.perf_counter();rows=[]
    def one(seed,mode):
        name=f'{seed}-{mode}';folder=args.output/name
        require_space(folder,16*1024**2)
        command=[sys.executable,'scripts/evaluate_rgb_servo_routing.py','--protocol',str(args.protocol),
                 '--routing',str(args.routing),'--model',str(args.model),'--seed',str(seed),'--split',args.split,'--output',str(folder)]
        if mode in ('nominal-frozen','push-frozen'):command+=['--mode','frozen']
        if mode.startswith('push-'):command+=['--push',protocol['push_protocol']]
        with (args.output/(name+'.log')).open('w') as log:
            process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        path=folder/'report.json'
        if not path.exists():return {'seed':seed,'case':mode,'passed':False,'status':'execution_error','exit_code':process.returncode}
        report=json.loads(path.read_text());physical=report.get('physical') or {};metrics=physical.get('metrics') or {}
        return {'seed':seed,'case':mode,'passed':report['status']=='succeeded' and bool(physical.get('passed')),
                'status':report['status'],'message':report.get('message'),'route':(report.get('route') or {}).get('kind'),
                'completed_legs':len(report.get('completed_legs',[])),'simulation_seconds':report.get('simulation_seconds'),
                'wall_seconds':report.get('wall_seconds'),'placement_error_mm':metrics.get('placement_error_mm'),
                'push_translation_mm':report.get('push_translation_before_grasp_mm'),
                'report':name+'/report.json','report_sha256':sha(path)}
    def summary():
        return {'frozen_inputs':frozen,'attempted':len(rows),'planned':len(seeds)*len(modes),
                'passed_by_case':{mode:sum(row['passed'] for row in rows if row['case']==mode) for mode in modes},
                'rows':sorted(rows,key=lambda row:(row['seed'],row['case'])),'wall_seconds':time.perf_counter()-began,
                'scope':'Every seed retained, including invalid starts, pre-motion refusals and relay-leg failures.'}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs=[pool.submit(one,seed,mode) for seed in seeds for mode in modes]
        for job in as_completed(jobs):
            row=job.result();rows.append(row);require_space(args.output,1024**2)
            (args.output/'summary.json').write_text(json.dumps(summary(),indent=2)+'\n')
            print(json.dumps(row),flush=True)
    print(json.dumps({key:value for key,value in summary().items() if key not in ('rows','frozen_inputs')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,default=Path('docs/robotics/experiments/rgb-servo-bottle-v2.json'))
    p.add_argument('--routing',type=Path,required=True);p.add_argument('--model',type=Path,default=Path('models/bottle_servo_v1'))
    p.add_argument('--split',choices=['development','evaluation'],required=True);p.add_argument('--freeze',type=Path)
    p.add_argument('--workers',type=int,default=3);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
