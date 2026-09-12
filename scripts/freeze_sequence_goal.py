"""Freeze model, execution sources and six untouched starts before final comparison."""
import argparse,hashlib,json,shutil
from pathlib import Path
from datetime import datetime,timezone
from importlib.metadata import version
ROOT=Path(__file__).resolve().parents[1]

def main(a):
    out=ROOT/'docs/robotics/bottle-sequence-freeze.json'
    if out.exists():raise FileExistsError('Selection already frozen; do not retune from held-out outcomes.')
    folders=[ROOT/'.run/retrieval-fit5-v4',Path(a.candidate).resolve()]
    paths=['simulation_lab/retrieval_policy.py','simulation_lab/sequence_policy.py','simulation_lab/primitive_policy.py',
        'simulation_lab/act_learning.py','simulation_lab/policy_control.py','simulation_lab/dinner_autonomy.py',
        'simulation_lab/autonomy.py','simulation_lab/scene.py','simulation_lab/dataset.py','simulation_lab/storage.py',
        'scripts/evaluate_act.py','scripts/evaluate_sequence_suite.py','scripts/prepare_bottle_data.py',
        'docs/robotics/bottle-sequence-protocol.json']
    protocol=json.loads((ROOT/paths[-1]).read_text())
    for row in protocol['held_out']:
        paths += ['.run/bottle-sequence-test-states/'+row['id']+'/'+name for name in ['scene.xml','manifest.json','initial-integration-state.npy']]
    snapshot=ROOT/'.run/bottle-sequence-frozen-sources'
    if snapshot.exists():raise FileExistsError(snapshot)
    for name in paths:
        target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    report={'frozen_at':datetime.now(timezone.utc).isoformat(),'selection_reason':a.reason,
            'candidate':folders[1].relative_to(ROOT).as_posix(),'source_snapshot':snapshot.relative_to(ROOT).as_posix(),
            'runtime_versions':{name:version(name) for name in ['mujoco','torch','numpy','Pillow']},
            'checkpoints':{p.relative_to(ROOT).as_posix():{f.name:digest(f) for f in p.iterdir() if f.is_file()} for p in folders},
            'sources':{p:digest(ROOT/p) for p in paths},
            'rule':'No model, controller, criterion or test-state tuning from held-out outcomes during this goal.'}
    out.write_text(json.dumps(report,indent=2));print('Frozen neural selection and six new test states.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--reason',required=True);main(p.parse_args())
