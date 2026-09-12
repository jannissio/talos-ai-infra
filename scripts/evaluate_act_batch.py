"""Run familiar-start diagnostics sequentially so GPU latency is not cross-contaminated."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def run(a):
    output=Path(a.output).resolve();output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name in ['upright-01','upright-04','sideways-01','sideways-03','sideways-11']:
        target=output/(name+'.json')
        command=[sys.executable,str(ROOT/'scripts/evaluate_act.py'),'--checkpoint',str(Path(a.checkpoint).resolve()),
                 '--episode',str(Path(a.episode_root).resolve()/name),'--output',str(target),'--seconds','60','--gripper-cap','.25','--device',a.device]
        print('Evaluating',name,flush=True)
        if a.record_video:command+=['--video',str(output/(name+'.mp4'))]
        with (output/(name+'.log')).open('w') as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        if result.returncode:raise RuntimeError('Evaluation failed to execute; inspect '+str(output/(name+'.log')))
        row=json.loads(target.read_text());rows.append(row)
        print(name,row['success'],row['reason'],flush=True)
    report={'evaluation':'five familiar training starts; not held-out generalization',
            'successes':sum(x['success'] for x in rows),'trials':len(rows),'results':rows}
    (output/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--episode-root',default=str(ROOT/'.run/bottle-pilot'));p.add_argument('--record-video',action='store_true')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');run(p.parse_args())
