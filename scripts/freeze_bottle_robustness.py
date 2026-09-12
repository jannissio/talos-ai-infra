"""Record the selected candidate before looking at held-out trial outcomes."""
import argparse,hashlib,json,shutil
from importlib.metadata import version
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(a):
    out=ROOT/'docs/robotics/bottle-robustness-freeze.json'
    if out.exists():raise FileExistsError('The held-out selection is already frozen.')
    checkpoints=[ROOT/'.run/retrieval-fit5-v4',Path(a.candidate).resolve()]
    source_paths=['simulation_lab/retrieval_policy.py','simulation_lab/act_learning.py',
        'simulation_lab/policy_control.py','scripts/evaluate_act.py','scripts/evaluate_bottle_protocol.py',
        'simulation_lab/dinner_autonomy.py','simulation_lab/autonomy.py','simulation_lab/scene.py',
        'simulation_lab/dataset.py','simulation_lab/storage.py','scripts/prepare_bottle_data.py',
        'docs/robotics/bottle-robustness-protocol.json']
    protocol=json.loads((ROOT/source_paths[-1]).read_text())
    for row in protocol['held_out']:
        for name in ['scene.xml','initial-integration-state.npy','manifest.json']:
            source_paths.append('.run/bottle-robustness-data/'+row['id']+'/'+name)
    snapshot=ROOT/'.run/bottle-robustness-frozen-sources'
    if snapshot.exists():raise FileExistsError(snapshot)
    for name in source_paths:
        target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    report={'frozen_at':datetime.now(timezone.utc).isoformat(),'selection_reason':a.reason,
            'source_snapshot':str(snapshot.relative_to(ROOT)).replace('\\','/'),
            'runtime_versions':{name:version(name) for name in ['mujoco','torch','numpy','Pillow']},
            'candidate':str(checkpoints[1].relative_to(ROOT)).replace('\\','/'),
            'checkpoints':{str(p.relative_to(ROOT)).replace('\\','/'):{f.name:digest(f) for f in p.iterdir() if f.is_file()} for p in checkpoints},
            'sources':{p:digest(ROOT/p) for p in source_paths},
            'rule':'No model, controller, success-criterion or test-state tuning from held-out outcomes in this milestone.'}
    out.write_text(json.dumps(report,indent=2));print('Candidate and test protocol frozen.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--reason',required=True);main(p.parse_args())
