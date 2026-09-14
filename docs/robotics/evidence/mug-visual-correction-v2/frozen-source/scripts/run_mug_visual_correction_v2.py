"""Run every declared physical comparison once, with at most two workers."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.evaluate_mug_visual_correction_v2 import read,sha,space,write,fingerprint


def run(args):
    p=read(args.protocol);root=ROOT/p['raw_root']
    frozen=read(root/'development-freeze.json')
    if fingerprint(p,args.protocol)!=frozen['inputs']:raise ValueError('Frozen experiment inputs changed.')
    summary=root/(args.split+'-batch.json')
    if summary.exists():raise FileExistsError('Preserve earlier batches.')
    cases=[(seed,preset,variant) for seed in p[args.split+'_seeds'] for preset in p['starting_presets'] for variant in p['modes']]
    jobs=[]
    for seed,preset,variant in cases:
        name=f'{seed}-{preset}-{variant}'
        output=root/args.split/name;log=ROOT/'.run/final-goal'/f'mug-visual-correction-v2-{args.split}-{name}.log'
        if output.exists() or log.exists():raise FileExistsError('Preserve earlier output/log for '+name)
        jobs.append((seed,preset,variant,output,log))
    space(p,summary,len(jobs)*16*1024**2)
    def one(job):
        seed,preset,variant,output,log=job
        if output.exists() or log.exists():raise FileExistsError('A case was created before launch.')
        preflight=space(p,output,16*1024**2)
        with log.open('x',encoding='utf-8') as stream:
            result=subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_mug_visual_correction_v2.py'),
                '--mode','trial','--protocol',str(args.protocol),'--split',args.split,'--seed',str(seed),'--preset',preset,'--variant',variant],
                cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        report=read(output/'report.json') if (output/'report.json').exists() else {'status':'missing_report'}
        mug=next((r for r in report.get('task',{}).get('results',[]) if r.get('skill')=='mug'),None)
        return {'seed':seed,'preset':preset,'variant':variant,'exit_code':result.returncode,'status':report['status'],
            'mug_status':mug.get('status') if mug else None,'wall_seconds':report.get('wall_seconds'),'preflight':preflight,
            'report':(output/'report.json').relative_to(ROOT).as_posix(),'console':log.relative_to(ROOT).as_posix()}
    results=[]
    with ThreadPoolExecutor(max_workers=p['budget']['maximum_physical_workers']) as pool:
        futures={pool.submit(one,job):job for job in jobs}
        for future in as_completed(futures):
            result=future.result();results.append(result)
            print({'completed':len(results),'total':len(jobs),**{key:result[key] for key in ('seed','preset','variant','status','mug_status','wall_seconds')}},flush=True)
    results.sort(key=lambda r:(r['seed'],r['preset'],r['variant']))
    write(p,summary,{'schema':p['schema'],'protocol_sha256':sha(args.protocol),'freeze_sha256':sha(root/'development-freeze.json'),
        'split':args.split,'planned':len(jobs),'completed':len(results),'rows':results,
        'scope':'All declared physical workflows attempted. Acceptance requires the separate complete paired-result audit.'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'docs/robotics/experiments/mug-visual-correction-v2.json')
    parser.add_argument('--split',choices=['development','evaluation'],required=True)
    args=parser.parse_args();args.protocol=args.protocol.resolve();run(args)
