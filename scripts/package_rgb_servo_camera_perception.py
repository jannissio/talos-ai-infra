"""Audit and package a frozen paired-camera perception test, including failures."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from simulation_lab.rgb_servo_cameras import KEYPOINTS
from simulation_lab.scene import build_scene
from simulation_lab.storage import require_space


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(folder):
    report,protocol,parent=read(folder/'report.json'),read(folder/'protocol.json'),read(folder/'parent-protocol.json')
    for name,key in [('protocol.json','protocol_sha256'),('parent-protocol.json','parent_protocol_sha256'),
                     ('states.npz','states_sha256'),('evaluate_rgb_servo_camera_perception.py','source_sha256')]:
        if sha(folder/name)!=report[key]:
            raise ValueError('Evidence hash mismatch: '+name)
    for name,digest in {**report['runtime_source_sha256'],**report['arm_pose_source_sha256']}.items():
        if sha(ROOT/name)!=digest:
            raise ValueError('Reproduction input changed: '+name)
    model_folder=ROOT/parent['observer_model']
    if sha(model_folder/'observer.safetensors')!=parent['observer_sha256']:
        raise ValueError('Observer checkpoint changed.')
    for name,digest in parent['observer_ir_sha256'].items():
        if sha(model_folder/'openvino'/name)!=digest:
            raise ValueError('Observer export changed.')
    with np.load(folder/'states.npz',allow_pickle=False) as archive:
        states={name:archive[name] for name in archive.files}
    count=protocol['states']
    if any(len(value)!=count for value in states.values()) or any(not np.isfinite(states[name]).all() for name in ('qpos','light_diffuse','headlight_ambient')):
        raise ValueError('Sampled-state arrays are incomplete or nonfinite.')
    rows=report['rows']
    expected={(index,name) for index in range(count) for name in protocol['configurations']}
    if len(rows)!=len(expected) or {(r['index'],r['configuration']) for r in rows}!=expected:
        raise ValueError('Missing, repeated or undeclared perception observation.')
    xml,_=build_scene(seed=42,scenario='dinner',dinner_preset='task')
    model=mujoco.MjModel.from_xml_string(xml)
    data=mujoco.MjData(model)
    truth=[]
    for pose in states['qpos']:
        data.qpos[:]=pose
        data.qvel[:]=0
        mujoco.mj_forward(model,data)
        truth.append(KEYPOINTS @ data.body('bottle').xmat.reshape(3,3).T+data.body('bottle').xpos)
    delta=0.
    for row in rows:
        present=bool(states['present'][row['index']])
        if row['scoring_only_present']!=present:
            raise ValueError('Recorded presence differs from sampled state.')
        if present and row['observation']['status']=='observed':
            error=float(np.linalg.norm(np.asarray(row['observation']['keypoints_m'])-truth[row['index']],axis=1).max()*1000)
            delta=max(delta,abs(error-row['scoring_only_error_mm']))
        elif row['scoring_only_error_mm'] is not None:
            raise ValueError('Invalid accepted-error entry.')
    if delta>1e-8:
        raise ValueError('Independent state geometry disagrees with the report.')
    gate,summaries=protocol['gate'],{}
    for name in protocol['configurations']:
        subset=[r for r in rows if r['configuration']==name]
        present=[r for r in subset if r['scoring_only_present']]
        accepted=[r for r in present if r['observation']['status']=='observed']
        absent=[r for r in subset if not r['scoring_only_present']]
        false=sum(r['observation']['status']=='observed' for r in absent)
        errors=[r['scoring_only_error_mm'] for r in accepted]
        p95,maximum=(float(np.quantile(errors,.95)),max(errors)) if errors else (None,None)
        passed=bool(present and absent and errors and len(accepted)/len(present)>=gate['minimum_present_acceptance_fraction']
                    and false<=gate['maximum_false_accepted_absent'] and p95<=gate['maximum_accepted_p95_error_mm']
                    and maximum<=gate['maximum_accepted_error_mm'])
        summaries[name]={'states':len(subset),'present':len(present),'accepted_present':len(accepted),'absent':len(absent),
                         'false_accepted_absent':false,'accepted_error_p95_mm':p95,'accepted_error_max_mm':maximum,'gate_passed':passed}
    selected=protocol['selected_configuration']
    if report['selected_configuration']!=selected or summaries!=report['summaries'] or report['selected_gate_passed']!=summaries[selected]['gate_passed']:
        raise ValueError('Reported selection or gate differs from independent aggregation.')
    return {'all_observations_verified':len(rows),'sampled_states_verified':count,'maximum_scoring_discrepancy_mm':delta,
            'summaries':summaries,'selected_configuration':selected,'selected_gate_passed':summaries[selected]['gate_passed'],
            'physical_trials':0,'promoted':False,'scope':'Fresh static perception test on exposed arm-pose families; no physical, Intel or general manipulation claim.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    if args.verify_only:
        for row in read(args.output/'manifest.json')['files']:
            if sha(args.output/row['file'])!=row['sha256']:
                raise ValueError('Packaged file changed: '+row['file'])
        result=audit(args.output)
        if result!=read(args.output/'audit.json'):
            raise ValueError('Saved audit changed.')
    else:
        if not args.source:
            parser.error('--source is required for a new package')
        if args.output.exists():
            raise FileExistsError('Preserve the existing package.')
        result=audit(args.source)
        names=['report.json','states.npz','protocol.json','parent-protocol.json','evaluate_rgb_servo_camera_perception.py']
        names += [f'frame-{index:03d}-{name}.png' for index in range(0,256,64) for name in read(args.source/'protocol.json')['configurations']]
        files=[args.source/name for name in names]
        expected=sum(p.stat().st_size for p in args.source.rglob('*') if p.is_file())+sum(p.stat().st_size for p in files)+2*1024**2
        protocol=read(args.source/'protocol.json')
        if expected>protocol['budget']['maximum_new_data_mib']*1024**2:
            raise ValueError('Raw plus packaged evidence exceeds the declared budget.')
        require_space(args.output,expected)
        args.output.mkdir(parents=True)
        entries=[]
        for source in files:
            target=args.output/source.name
            require_space(target,source.stat().st_size+1024**2)
            target.write_bytes(source.read_bytes())
            entries.append({'file':target.name,'sha256':sha(target),'bytes':target.stat().st_size,'transformation':'byte-identical copy'})
        if audit(args.output)!=result:
            raise ValueError('Packaged audit changed.')
        for name,value in [('audit.json',result),('manifest.json',{'files':entries,'source':args.source.as_posix(),'originals_modified':False})]:
            require_space(args.output/name,1024**2)
            (args.output/name).write_bytes((json.dumps(value,indent=2)+'\n').encode())
    print(json.dumps(result))
