"""Sequence-policy suites with a fresh, gated held-out test set."""
import argparse,hashlib,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from simulation_lab.storage import require_space

def prepare():
    import mujoco,numpy as np
    from scripts.prepare_bottle_data import STATE,state
    protocol=json.loads((ROOT/'docs/robotics/bottle-sequence-protocol.json').read_text())
    out=ROOT/'.run/bottle-sequence-test-states'
    if out.exists():raise FileExistsError(out)
    require_space(out,len(protocol['held_out'])*8*1024**2)
    out.mkdir()
    for spec in protocol['held_out']:
        require_space(out,8*1024**2)
        source=ROOT/'.run/bottle-matched-nvidia'/spec['anchor'];folder=out/spec['id'];folder.mkdir()
        shutil.copy2(source/'scene.xml',folder/'scene.xml')
        model=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'));data=mujoco.MjData(model)
        mujoco.mj_setState(model,data,np.load(source/'initial-integration-state.npy',allow_pickle=False),STATE)
        adr=model.joint('bottle_free').qposadr[0];data.qpos[adr:adr+2]+=[spec['dx'],spec['dy']]
        mujoco.mj_forward(model,data);np.save(folder/'initial-integration-state.npy',state(model,data),allow_pickle=False)
        manifest=json.loads((source/'manifest.json').read_text());manifest.update(id=spec['id'],training_eligible=False,
            images={'status':'not_captured'},sequence_test_spec=spec)
        (folder/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Created six held-out initial states, without rendering or training on them.')

def main(a):
    if a.prepare:return prepare()
    checkpoint=Path(a.checkpoint).resolve();out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    if a.suite=='test':
        frozen=json.loads((ROOT/'docs/robotics/bottle-sequence-freeze.json').read_text())
        key=checkpoint.relative_to(ROOT).as_posix()
        for name,digest in frozen['checkpoints'][key].items():assert hashlib.sha256((checkpoint/name).read_bytes()).hexdigest()==digest
        for name,digest in frozen['sources'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
        folders=[ROOT/'.run/bottle-sequence-test-states'/s['id'] for s in json.loads((ROOT/'docs/robotics/bottle-sequence-protocol.json').read_text())['held_out']]
    elif a.suite=='training':
        folders=[Path(r['folder']) for r in json.loads((ROOT/'.run/bottle-robustness-data/fit-lineage.json').read_text())['episodes']]
    elif a.suite=='original':
        folders=[ROOT/'.run/bottle-matched-nvidia'/n for n in ['upright-01','upright-04','sideways-01','sideways-03','sideways-11']]
    else:
        protocol=json.loads((ROOT/'docs/robotics/bottle-robustness-protocol.json').read_text())
        folders=[ROOT/'.run/bottle-robustness-data'/s['id'] for s in protocol['development']+protocol['held_out']]
    require_space(out,30*1024**2);out.mkdir(parents=True);rows=[]
    for folder in folders:
        result=out/(folder.name+'.json')
        cmd=[sys.executable,str(ROOT/'scripts/evaluate_act.py'),'--checkpoint',str(checkpoint),'--episode',str(folder),
             '--output',str(result),'--device','cpu']
        if a.video:cmd+=['--video',str(out/(folder.name+'.mp4'))]
        with (out/(folder.name+'.log')).open('w') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
        row=json.loads(result.read_text());row.get('policy_details',{}).pop('trace',None);rows.append(row)
        print(folder.name,row['success'],row['reason'],flush=True)
    (out/'summary.json').write_text(json.dumps({'suite':a.suite,'successes':sum(r['success'] for r in rows),'trials':len(rows),'results':rows},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--checkpoint');p.add_argument('--output')
    p.add_argument('--suite',choices=['original','training','development','test'],default='original');p.add_argument('--video',action='store_true');main(p.parse_args())
