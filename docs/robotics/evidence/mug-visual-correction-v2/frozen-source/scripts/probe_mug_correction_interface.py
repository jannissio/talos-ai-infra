"""Compose frozen RGB observations and actual joints on all saved mug states."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from simulation_lab.mug_visual_geometry import correction_request
from simulation_lab.mug_correction_runtime import LocalMugMotor
from simulation_lab.storage import require_space


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def run(args):
    p=read(args.protocol)
    output=ROOT/p['raw_root']/'interface.json'
    if output.exists():raise FileExistsError('Preserve earlier interface evaluations.')
    inputs={}
    for field,hash_field in (('observer','observer_manifest_sha256'),('motor','motor_manifest_sha256')):
        root=ROOT/p[field]
        if sha(root/'manifest.json')!=p[hash_field]:raise ValueError('A declared model package changed.')
        for name,digest in read(root/'manifest.json')['files'].items():
            if sha(root/name)!=digest:raise ValueError('Model artifact changed: '+name)
            inputs[(root/name).relative_to(ROOT).as_posix()]=digest
    if sha(ROOT/p['motor']/'runtime.json')!=p['motor_runtime_sha256']:raise ValueError('Motor contract changed.')
    for field in ('observer_physical_report','visibility_report'):
        path=ROOT/p['interface'][field]
        if sha(path)!=p['interface'][field+'_sha256']:raise ValueError('Declared exposed input changed.')
        inputs[path.relative_to(ROOT).as_posix()]=sha(path)
    observation_report=read(ROOT/p['interface']['observer_physical_report'])
    visibility=read(ROOT/p['interface']['visibility_report'])
    if len(observation_report['rows'])!=p['interface']['all_exposed_states'] or len(visibility['states'])!=p['interface']['all_exposed_states']:
        raise ValueError('The complete exposed set is required.')
    for name,digest in observation_report['input_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Recorded physics/image input changed: '+name)
        inputs[name]=digest
    preflight=require_space(output,8*1024**2,p['budget']['reserve_gib']*1024**3)
    started=time.perf_counter();rows=[];current=None;runtime=None
    for index,(saved,state) in enumerate(zip(observation_report['rows'],visibility['states'])):
        if saved['index']!=index or saved['frame_index']!=state['frame_index'] or saved['trace']!=state['trace']:
            raise ValueError('Frozen observation/state indexing disagrees.')
        trace=ROOT/state['trace_root']/state['trace']
        if current!=trace:
            current=trace;model=mujoco.MjModel.from_xml_path(str(trace/'scene.xml'))
            runtime=LocalMugMotor(ROOT/p['motor'],model)
            with np.load(trace/'states.npz',allow_pickle=False) as archive:qpos=archive['qpos'].copy()
        origin,requested=correction_request(saved['observation'],p['control'])
        motor=runtime.predict(qpos[state['frame_index'],6:11],requested)
        # Full object state is restored only after action-input composition, for
        # independent scoring of the semantic origin. This is offline replay.
        scoring=mujoco.MjData(model);scoring.qpos[:]=qpos[state['frame_index']];mujoco.mj_forward(model,scoring)
        origin_error=float(np.linalg.norm(origin-scoring.body('mug').xpos)*1000)
        rows.append({'index':index,'trace':state['trace'],'frame_index':state['frame_index'],'phase':saved['phase'],
            'control_eligible':saved['phase'] in p['interface']['control_phases'],'estimated_origin_m':origin.tolist(),
            'scoring_only_origin_error_mm':origin_error,'motor':motor})
    eligible=[r for r in rows if r['control_eligible']]
    if len(eligible)!=p['interface']['control_states']:raise ValueError('Wrong preregistered late-placement denominator.')
    errors=[r['motor']['position_error_mm'] for r in eligible]
    summary={'all_states':len(rows),'all_motor_accepted':sum(r['motor']['accepted'] for r in rows),
        'control_states':len(eligible),'control_motor_accepted':sum(r['motor']['accepted'] for r in eligible),
        'control_p95_effective_error_mm':float(np.quantile(errors,.95)),'control_maximum_effective_error_mm':max(errors),
        'maximum_origin_error_mm':max(r['scoring_only_origin_error_mm'] for r in rows)}
    gate=p['interface']
    summary['gate_passed']=bool(summary['control_motor_accepted']/len(eligible)>=gate['minimum_control_motor_acceptance_fraction']
        and summary['control_p95_effective_error_mm']<=gate['maximum_control_p95_effective_error_mm']
        and max(errors)<=gate['maximum_control_effective_error_mm'])
    sources=[Path(__file__),ROOT/'simulation_lab/mug_visual_geometry.py',ROOT/'simulation_lab/mug_correction_runtime.py',
        ROOT/'simulation_lab/mug_correction_limits.py',ROOT/'simulation_lab/mug_correction_motor.py']
    result={'schema':p['schema'],'protocol_sha256':sha(args.protocol),'source_sha256':{path.relative_to(ROOT).as_posix():sha(path) for path in sources},
        'input_sha256':inputs,'summary':summary,'rows':rows,'preflight':preflight,'wall_seconds':time.perf_counter()-started,
        'new_physical_trials':0,'scope':'Every preserved neural observation composed with its recorded joints. All 144 exposed states retained; the predefined 72 late-placement states determine the interface gate. No new physics, fitting or corrective success claim.'}
    payload=(json.dumps(result,indent=2)+'\n').encode()
    require_space(output,len(payload)+1024,p['budget']['reserve_gib']*1024**3);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as stream:stream.write(payload)
    print({'summary':summary,'wall_seconds':result['wall_seconds']},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'docs/robotics/experiments/mug-visual-correction-v1.json')
    args=parser.parse_args();args.protocol=args.protocol.resolve();run(args)
