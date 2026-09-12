"""Evaluate a frozen split; held-out trials require a recorded model/source freeze."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.bottle_robustness import budget

def main(a):
    protocol=json.loads((ROOT/'docs/robotics/bottle-robustness-protocol.json').read_text())
    checkpoint=Path(a.checkpoint).resolve();out=Path(a.output).resolve()
    if out.exists():raise FileExistsError(out)
    if a.split=='held_out':
        freeze=json.loads((ROOT/'docs/robotics/bottle-robustness-freeze.json').read_text())
        expected=freeze['checkpoints'][str(checkpoint.relative_to(ROOT)).replace('\\','/')]
        for name,digest in expected.items():
            assert hashlib.sha256((checkpoint/name).read_bytes()).hexdigest()==digest,'Checkpoint changed after freeze'
        for name,digest in freeze['sources'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,'Policy/evaluator source changed after freeze'
    budget();out.mkdir(parents=True);rows=[]
    for spec in protocol[a.split]:
        name=spec['id'];folder=ROOT/'.run/bottle-robustness-data'/name;result=out/(name+'.json')
        command=[sys.executable,str(ROOT/'scripts/evaluate_act.py'),'--checkpoint',str(checkpoint),'--episode',str(folder),
                 '--output',str(result),'--device','cpu']
        if a.record_video:command+=['--video',str(out/(name+'.mp4'))]
        with (out/(name+'.log')).open('w') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
        row=json.loads(result.read_text());row['policy_details'].pop('trace',None);rows.append(row)
        print(name,row['success'],row['reason'],flush=True)
    summary={'split':a.split,'checkpoint':str(checkpoint),'successes':sum(r['success'] for r in rows),
             'trials':len(rows),'results':rows}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print('Passed',summary['successes'],'of',summary['trials'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--split',choices=['training','development','held_out'],required=True)
    p.add_argument('--record-video',action='store_true');main(p.parse_args())
