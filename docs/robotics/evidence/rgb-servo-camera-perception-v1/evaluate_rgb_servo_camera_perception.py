"""Fresh paired perception evaluation of a preselected fixed camera layout."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from PIL import Image
import torch
from simulation_lab.rgb_servo_cameras import VIEWS, KEYPOINTS, camera_argument, calibration
from simulation_lab.rgb_servo_openvino import OpenVinoBottleObserver
from simulation_lab.scene import build_scene, HOME
from simulation_lab.storage import require_space


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(rows, gate):
    present = [r for r in rows if r['scoring_only_present']]
    accepted = [r for r in present if r['observation']['status']=='observed']
    absent = [r for r in rows if not r['scoring_only_present']]
    false_accepts = sum(r['observation']['status']=='observed' for r in absent)
    errors = [r['scoring_only_error_mm'] for r in accepted]
    p95, maximum = (float(np.quantile(errors,.95)),max(errors)) if errors else (None,None)
    passed = bool(present and absent and errors and len(accepted)/len(present)>=gate['minimum_present_acceptance_fraction']
                  and false_accepts<=gate['maximum_false_accepted_absent'] and p95<=gate['maximum_accepted_p95_error_mm']
                  and maximum<=gate['maximum_accepted_error_mm'])
    return {'states':len(rows),'present':len(present),'accepted_present':len(accepted),'absent':len(absent),
            'false_accepted_absent':false_accepts,'accepted_error_p95_mm':p95,'accepted_error_max_mm':maximum,
            'gate_passed':passed}


def run(args):
    if args.output.exists():
        raise FileExistsError('Preserve earlier evidence; choose an unused output folder.')
    protocol = json.loads(args.protocol.read_text())
    parent_path = ROOT / protocol['parent']
    parent = json.loads(parent_path.read_text())
    diagnostic = json.loads((ROOT / 'docs/robotics/evidence/rgb-servo-camera-visibility-v1/audit.json').read_text())
    if protocol['selected_configuration']!=diagnostic['selected_configuration'] or not diagnostic['summaries'][protocol['selected_configuration']]['diagnostic_gate_passed']:
        raise ValueError('Camera selection differs from the completed diagnostic.')
    require_space(args.output,protocol['budget']['maximum_new_data_mib']*1024**2)
    model_folder = ROOT / parent['observer_model']
    if sha(model_folder / 'observer.safetensors')!=parent['observer_sha256']:
        raise ValueError('Observer checkpoint changed.')
    for name,digest in parent['observer_ir_sha256'].items():
        if sha(model_folder / 'openvino' / name)!=digest:
            raise ValueError('Observer IR changed.')
    pose_arrays=[]
    for name in protocol['arm_pose_sources']:
        with np.load(ROOT / name,allow_pickle=False) as states:
            pose_arrays.append(states['qpos'][:,:12])
    arm_poses=np.concatenate(pose_arrays)
    xml,layout=build_scene(seed=42,scenario='dinner',dinner_preset='task')
    model=mujoco.MjModel.from_xml_string(xml)
    model.vis.quality.offsamples=0
    data=mujoco.MjData(model)
    data.qpos[:12]=HOME*2
    data.ctrl[:]=HOME*2
    mujoco.mj_forward(model,data)
    initial=data.qpos.copy()
    address=int(model.joint('bottle_free').qposadr[0])
    diffuse=model.light_diffuse.copy()
    ambient=model.vis.headlight.ambient.copy()
    renderer=mujoco.Renderer(model,width=320,height=240)
    option=mujoco.MjvOption()
    option.geomgroup[3:]=0
    torch.set_num_threads(2)
    observer=OpenVinoBottleObserver(model_folder / 'openvino/observer.xml',minimum_views=2,rigid_geometry=True)
    rng=np.random.default_rng(protocol['rng_seed'])
    args.output.mkdir(parents=True)
    began=time.perf_counter()
    rows,cameras,qpos_rows,diffuse_rows,ambient_rows,present_rows=[],{},[],[],[],[]
    try:
        for index in range(protocol['states']):
            if time.perf_counter()-began>protocol['budget']['maximum_compute_minutes']*60:
                raise TimeoutError('Declared compute budget reached; retain partial output.')
            require_space(args.output,8*1024**2)
            data.qpos[:]=initial
            data.qvel[:]=0
            data.qpos[:12]=arm_poses[int(rng.integers(len(arm_poses)))] if rng.random()<.8 else np.array(HOME*2)
            data.qpos[:12]+=rng.normal(0,.025,12)
            data.qpos[:12]=np.clip(data.qpos[:12],model.actuator_ctrlrange[:,0],model.actuator_ctrlrange[:,1])
            present=bool(rng.random()>=.1)
            x,y,lift=rng.uniform(-.16,.23),rng.uniform(-.20,.09),rng.uniform(0.,.08)
            if rng.random()<.4:
                lift=0.
            angle,tilt=rng.uniform(-.6,.6),rng.uniform(-.07,.07)
            yaw=np.array([np.cos(angle/2),0.,0.,np.sin(angle/2)])
            lean=np.array([np.cos(tilt/2),np.sin(tilt/2),0.,0.])
            quat=np.zeros(4)
            mujoco.mju_mulQuat(quat,yaw,lean)
            data.qpos[address:address+7]=[x,y,layout['table_z']+lift+.001,*quat]
            if not present:
                data.qpos[address:address+3]=[2.,2.,2.]
            model.light_diffuse[:]=diffuse*rng.uniform(.65,1.25,(model.nlight,1))*rng.uniform(.92,1.08,(1,3))
            model.vis.headlight.ambient[:]=ambient*rng.uniform(.7,1.3)
            mujoco.mj_forward(model,data)
            qpos_rows.append(data.qpos.copy()); diffuse_rows.append(model.light_diffuse.copy())
            ambient_rows.append(model.vis.headlight.ambient.copy()); present_rows.append(present)
            for configuration in protocol['configurations']:
                images,calibrations={},{}
                for slot in VIEWS:
                    camera=camera_argument(slot)
                    camera.azimuth,camera.elevation,camera.distance=parent['camera_configurations'][configuration][slot]
                    renderer.update_scene(data,camera=camera,scene_option=option)
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
                    images[slot]=renderer.render().copy()
                    calibrations[slot]=calibration(renderer)
                if configuration in cameras and cameras[configuration]!=calibrations:
                    raise ValueError('Fixed camera calibration changed between sampled states.')
                cameras[configuration]=calibrations
                observation=observer.observe(images,calibrations)
                truth=KEYPOINTS @ data.body('bottle').xmat.reshape(3,3).T + data.body('bottle').xpos
                error=float(np.linalg.norm(np.asarray(observation['keypoints_m'])-truth,axis=1).max()*1000) if present and observation['status']=='observed' else None
                rows.append({'index':index,'configuration':configuration,'observation':observation,
                             'scoring_only_present':present,'scoring_only_error_mm':error})
                if index%64==0:
                    require_space(args.output,1024**2)
                    Image.fromarray(np.concatenate([images[slot] for slot in VIEWS],axis=1)).save(args.output/f'frame-{index:03d}-{configuration}.png')
            if (index+1)%64==0:
                require_space(args.output,16*1024**2)
                np.savez_compressed(args.output/'states.npz',qpos=np.asarray(qpos_rows),light_diffuse=np.asarray(diffuse_rows),
                                    headlight_ambient=np.asarray(ambient_rows),present=np.asarray(present_rows,dtype=bool))
                (args.output/'partial.json').write_text(json.dumps(rows)+'\n')
                print(json.dumps({'completed_states':index+1,'observations':len(rows)}),flush=True)
    finally:
        renderer.close()
    summaries={name:score([r for r in rows if r['configuration']==name],protocol['gate']) for name in protocol['configurations']}
    dependencies=json.loads((ROOT/'docs/robotics/evidence/rgb-servo-v5/physical/development/frozen-inputs.json').read_text())['source_sha256']
    if any(sha(ROOT/name)!=digest for name,digest in dependencies.items()):
        raise ValueError('A runtime source changed during evaluation.')
    report={'schema':protocol['schema'],'protocol_sha256':sha(args.protocol),'parent_protocol_sha256':sha(parent_path),
            'source_sha256':sha(Path(__file__)),'parent_git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'runtime_source_sha256':dependencies,'arm_pose_source_sha256':{name:sha(ROOT/name) for name in protocol['arm_pose_sources']},
            'states_sha256':sha(args.output/'states.npz'),'calibrations':cameras,'summaries':summaries,'rows':rows,
            'selected_configuration':protocol['selected_configuration'],'selected_gate_passed':summaries[protocol['selected_configuration']]['gate_passed'],
            'environment':{'python':sys.version.split()[0],'numpy':np.__version__,'mujoco':mujoco.__version__,
                           'torch':str(torch.__version__),'openvino':version('openvino')},'inference_runtime':'OpenVINO CPU',
            'wall_seconds':time.perf_counter()-began,'scope':'Fresh static perception samples with known exposed arm-pose families; not physical trials, new-shape generalization or Intel evidence.'}
    require_space(args.output,16*1024**2)
    (args.output/'report.json').write_bytes((json.dumps(report,indent=2)+'\n').encode())
    for source,name in [(args.protocol,'protocol.json'),(parent_path,'parent-protocol.json'),(Path(__file__),'evaluate_rgb_servo_camera_perception.py')]:
        require_space(args.output,source.stat().st_size+1024**2)
        (args.output/name).write_bytes(source.read_bytes())
    generated=sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())
    if generated>protocol['budget']['maximum_new_data_mib']*1024**2:
        raise ValueError('Data budget exceeded; retain output without proceeding.')
    print(json.dumps({'summaries':summaries,'selected_gate_passed':report['selected_gate_passed'],'wall_seconds':report['wall_seconds'],'generated_bytes':generated}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=Path('docs/robotics/experiments/rgb-servo-camera-perception-v1.json'))
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
