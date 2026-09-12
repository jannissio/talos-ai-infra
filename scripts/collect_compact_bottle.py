"""Broader physical demonstrations with three initial images and compact actions.

This dataset is for initial-image-conditioned policies. It is NOT a continuous
visual observation dataset. Failed teacher/replay cases are never training eligible.
"""
import argparse,hashlib,json,math,os,sys,xml.etree.ElementTree as ET
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco,numpy as np
from PIL import Image
from simulation_lab.scene import HOME,build_scene
from simulation_lab.dinner_autonomy import DinnerTask
from simulation_lab.policy_control import apply_targets
from simulation_lab.storage import require_space,GIB
from scripts.prepare_bottle_data import state,replay


def trial(spec,out):
    require_space(out,32*1024**2);folder=out/spec['id'];folder.mkdir()
    xml,layout=build_scene(seed=spec['seed'],scenario='dinner',dinner_preset='task')
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);d.qpos[:12]=HOME*2;d.ctrl[:]=HOME*2
    if spec.get('pose'):
        p=spec['pose'];c,s=math.cos(p['yaw']/2),math.sin(p['yaw']/2)
        q=np.array([c,-s,c,s])*math.sqrt(.5) if p['sideways'] else [c,0,0,s]
        adr=m.joint('bottle_free').qposadr[0]
        d.qpos[adr:adr+7]=[p['x'],p['y'],layout['table_z']+(.026 if p['sideways'] else .001),*q]
    mujoco.mj_forward(m,d)
    for _ in range(300):mujoco.mj_step(m,d)
    d.time=0.;initial=state(m,d);initial_pose=d.body('bottle').xpos.copy()
    scene=ET.fromstring(xml);compiler=scene.find('compiler');compiler.set('meshdir',os.path.relpath(compiler.get('meshdir'),folder))
    (folder/'scene.xml').write_text(ET.tostring(scene,encoding='unicode'));np.save(folder/'initial-integration-state.npy',initial,allow_pickle=False)
    task=DinnerTask(m,d,layout);task.start(side='left',object_id='bottle');targets=np.array(HOME*2);rows=[]
    for tick in range(20000):
        before=d.qpos.copy();velocity=d.qvel.copy();task.update(targets)
        assert np.array_equal(before,d.qpos) and np.array_equal(velocity,d.qvel)
        task.grip_torque=.25;task.metrics['gripper_torque_limit_nm']=.25
        if not task.active:break
        rows.append({'index':tick,'time_s':tick*.005,'stage':task.stage,'target':targets.copy()})
        d.ctrl[:]=apply_targets(m,d,targets,.25);mujoco.mj_step(m,d)
        assert m.neq==0 and not np.any(d.xfrc_applied) and not np.any(d.qfrc_applied)
    if task.active:task.cancel(targets)
    manifest={'spec':spec,'layout':layout,'outcome':task.snapshot(),'training_eligible':False,'gripper_cap_nm':.25,
              'input_contract':'Three initial RGB views plus motor feedback; no continuous camera sequence stored',
              'privileged_training_labels':{'bottle_initial_position_m':initial_pose.tolist()},'replays':[]}
    if task.status=='succeeded':
        raw=np.array([r['target'] for r in rows]);indices=np.unique(np.r_[np.arange(0,len(raw),10),len(raw)-1])
        endpoints=np.clip(raw[indices],m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])
        matched=np.column_stack([np.interp(np.arange(len(raw)),indices,endpoints[:,j]) for j in range(12)])
        rows=[dict(r,target=t) for r,t in zip(rows,matched)]
        manifest['replays']=[replay(m,initial,rows,task,hz) for hz in (200,20)]
        if all(r['passed'] for r in manifest['replays']):
            require_space(folder,32*1024**2)
            np.savez_compressed(folder/'trajectory.npz',actions20=endpoints.astype('float32'),action_indices=indices,
                                stages=np.array([r['stage'] for r in rows])[indices],physics_steps=len(rows))
            mujoco.mj_setState(m,d,initial,mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
            m.vis.quality.offsamples=0;renderer=mujoco.Renderer(m,height=240,width=320);opt=mujoco.MjvOption();opt.geomgroup[3:]=0
            try:
                for camera in ['overhead','left_wrist_cam','right_wrist_cam']:
                    renderer.update_scene(d,camera=camera,scene_option=opt);renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
                    Image.fromarray(renderer.render()).save(folder/(camera+'.png'))
            finally:renderer.close()
            manifest['training_eligible']=True
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return {'id':spec['id'],'seed':spec['seed'],'training_eligible':manifest['training_eligible'],
            'status':task.status,'message':task.message,'replays':manifest['replays']}


def main(a):
    out=Path(a.output)
    if out.exists():raise FileExistsError('Use a new batch folder; previous outcomes are preserved.')
    require_space(out,a.count*8*1024**2);out.mkdir(parents=True)
    rng=np.random.default_rng(617131);specs=[]
    for i in range(a.count):
        spec={'id':f'train-{i:03d}','seed':2026090001+i,'split':'training'}
        if i>=a.count//2:
            sideways=i%4==0
            spec['pose']={'x':float(rng.uniform(-.13,-.05) if sideways else rng.uniform(-.1,.05)),
                          'y':float(rng.uniform(-.15,-.07) if sideways else rng.uniform(-.15,-.04)),
                          'yaw':float(rng.uniform(.4,1.2) if sideways else rng.uniform(-.4,.4)),'sideways':sideways}
        specs.append(spec)
    (out/'protocol.json').write_text(json.dumps({'specs':specs,'storage_reserve_gib':10,'expected_max_mib':a.count*8,
        'not_evaluation_seeds':True,'purpose':'Broader training only; new frozen evaluation seeds must be chosen afterward'},indent=2))
    rows=[]
    # Initialize the already-installed CUDA runtime before rendering to match
    # the NVIDIA image source used in the previous training collection.
    import torch
    torch.cuda.init()
    for spec in specs:
        rows.append(trial(spec,out));print(json.dumps(rows[-1]),flush=True)
        (out/'summary.json').write_text(json.dumps({'episodes':rows,'eligible':sum(r['training_eligible'] for r in rows),'planned':len(specs)},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--count',type=int,default=32)
    a=p.parse_args()
    if not 2<=a.count<=100:p.error('Choose 2–100 demonstrations per bounded batch.')
    main(a)
