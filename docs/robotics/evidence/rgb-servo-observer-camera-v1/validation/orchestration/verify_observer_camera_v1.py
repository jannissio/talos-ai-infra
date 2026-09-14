import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from simulation_lab.storage import require_space

def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def relative(path):return path.relative_to(ROOT).as_posix()
base=ROOT/'.run/rgb-servo-observer-camera-v1'
protocol_path=ROOT/'docs/robotics/experiments/rgb-servo-observer-camera-training-v1.json'
protocol=read(protocol_path)
training=read(base/'fit/training.json')
if not training['training_completed'] or not training['selection']:
    print(json.dumps({'stopped_before_export':True,'reason':'No completed eligible training selection.'}),flush=True)
    raise SystemExit(0)
selected=read(base/'fit/selected.json')
assert training['selection']==selected and selected['protocol_sha256']==sha(protocol_path)
checkpoint=ROOT/selected['checkpoint']
assert selected['checkpoint_sha256']==sha(checkpoint)
for entry in training['dataset_manifests'].values():
    manifest_path=ROOT/entry['path']
    assert sha(manifest_path)==entry['sha256']
    for shard in entry['shards']:
        assert sha(manifest_path.parent/shard['file'])==shard['sha256']

def run(name,arguments):
    output,log=base/name,base/(name+'.log')
    if output.exists() or log.exists():raise FileExistsError('Preserve previous verification data and logs.')
    used=sum(p.stat().st_size for p in base.rglob('*') if p.is_file())
    if used+64*1024**2>protocol['budget']['maximum_new_data_gib']*1024**3:raise ValueError('Experiment data budget would be exceeded.')
    require_space(output,64*1024**2)
    with log.open('x',encoding='utf-8') as stream:
        subprocess.run([sys.executable,*arguments,'--output',relative(output)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=300)
    return output

export=run('openvino-export',['scripts/export_rgb_servo_openvino.py','--observer',relative(checkpoint),
                            '--motor',protocol['motor']+'/motor.safetensors'])
parity=read(export/'parity.json')
assert parity['observer_sha256']==sha(checkpoint) and all(row['parity_passed'] for row in parity['networks'].values())

def summarize(result):
    gate=protocol['perception_gate']
    errors=result['error_mm']
    passed=bool(result['present'] and result['frames']>result['present'] and errors
                and result['accepted_present']/result['present']>=gate['minimum_present_acceptance_fraction']
                and result['false_accepted_absent']<=gate['maximum_false_accepted_absent']
                and errors['p95']<=gate['maximum_accepted_p95_error_mm'] and errors['max']<=gate['maximum_accepted_error_mm'])
    return {'states':result['frames'],'present':result['present'],'accepted_present':result['accepted_present'],
            'absent':result['frames']-result['present'],'false_accepted_absent':result['false_accepted_absent'],
            'accepted_error_p95_mm':errors['p95'] if errors else None,'accepted_error_max_mm':errors['max'] if errors else None,'gate_passed':passed}

def evaluate(split,configuration):
    output=run(split+'-cpu-'+configuration,['scripts/evaluate_rgb_servo_observer.py',
        '--checkpoint',relative(checkpoint),'--dataset',relative(base/(split+'-'+configuration)),
        '--minimum-views','2','--rigid-geometry','--openvino',relative(export)])
    return read(output/'results.json')

development={}
chosen=next(row for row in training['candidates'] if row['step']==selected['step'])
for configuration in protocol['camera_configurations']:
    result=evaluate('development',configuration)
    gpu=chosen['development'][configuration]
    assert len(result['rows'])==len(gpu['rows'])
    decisions=sum(a['status']!=b['observation']['status'] for a,b in zip(result['rows'],gpu['rows']))
    delta=max((abs(a['scoring_only_error_mm']-b['scoring_only_error_mm']) for a,b in zip(result['rows'],gpu['rows'])
               if 'scoring_only_error_mm' in a and b['scoring_only_error_mm'] is not None),default=0.)
    development[configuration]={**summarize(result),'changed_acceptance_decisions_from_cuda':decisions,
                                'maximum_score_delta_from_cuda_mm':delta}
    print(json.dumps({'cpu_development':configuration,**development[configuration]}),flush=True)
require_space(base/'cpu-development-audit.json',1024**2)
(base/'cpu-development-audit.json').write_text(json.dumps(development,indent=2)+'\n')
if not development[protocol['deployment_configuration']]['gate_passed']:
    print(json.dumps({'stopped_before_fresh_perception':True,'reason':'Selected OpenVINO model failed the development gate.'}),flush=True)
    raise SystemExit(0)
freeze=dict(selected)
freeze.update(openvino_artifact_sha256={name:sha(export/name) for name in ('observer.xml','observer.bin','motor.xml','motor.bin')},
              export_parity_sha256=sha(export/'parity.json'),cpu_development_audit_sha256=sha(base/'cpu-development-audit.json'),
              fresh_evaluation_rng_seed=protocol['evaluation_rng_seed'],fresh_evaluation_states=protocol['evaluation_states'],
              evaluation_source_sha256=sha(ROOT/'scripts/evaluate_rgb_servo_observer.py'))
freeze_path=base/'fresh-perception-freeze.json'
assert not freeze_path.exists()
require_space(freeze_path,1024**2)
freeze_path.write_text(json.dumps(freeze,indent=2)+'\n')
fresh={}
for configuration in protocol['camera_configurations']:
    run('evaluation-'+configuration,['scripts/collect_rgb_servo_camera_observer.py','--protocol',relative(protocol_path),
        '--split','evaluation','--camera-configuration',configuration,'--selection',relative(freeze_path)])
    fresh[configuration]=summarize(evaluate('evaluation',configuration))
    print(json.dumps({'fresh_perception':configuration,**fresh[configuration]}),flush=True)
record={'schema':protocol['schema'],'protocol_sha256':sha(protocol_path),'freeze_sha256':sha(freeze_path),
        'development':development,'fresh_perception':fresh,'selected_configuration':protocol['deployment_configuration'],
        'selected_fresh_gate_passed':fresh[protocol['deployment_configuration']]['gate_passed'],
        'physical_trials':0,'scope':'OpenVINO CPU perception validation on AMD; no physical promotion or Intel evidence.'}
require_space(base/'perception-verification.json',1024**2)
(base/'perception-verification.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
