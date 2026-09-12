"""Re-record successful nominal trajectories through the actual policy controller.

Only the initial state is restored. Cameras and proprioception are captured from
the resulting physics, never copied from the old teacher observations.
"""
import argparse,gzip,hashlib,json,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco,numpy as np
from PIL import Image
from scripts.prepare_bottle_data import STATE,replay,observation,save
from simulation_lab.dinner_autonomy import DinnerTask
from simulation_lab.policy_control import apply_targets,GRIPPER_CAP_NM
from simulation_lab.storage import require_space,GIB

ROOT=Path(__file__).resolve().parents[1]
FIT=['sideways-01','sideways-03','sideways-11','upright-01','upright-04']
CAMERAS=['overhead','left_wrist_cam','right_wrist_cam']

def collect(name,out):
    require_space(out,int(.4*GIB))
    source=ROOT/'.run/bottle-pilot'/name;folder=out/name
    if folder.exists():raise FileExistsError(folder)
    manifest=json.loads((source/'manifest.json').read_text())
    assert manifest['training_eligible'] and manifest['arm']=='left'
    model=mujoco.MjModel.from_xml_path(str(source/'scene.xml'));data=mujoco.MjData(model)
    initial=np.load(source/'initial-integration-state.npy',allow_pickle=False)
    mujoco.mj_setState(model,data,initial,STATE);mujoco.mj_forward(model,data)
    assert model.neq==0
    with gzip.open(source/'actions.jsonl.gz','rt') as f:old=[json.loads(s) for s in f]
    nominal=np.asarray([a['target'] for a in old]);indices=np.unique(np.r_[np.arange(0,len(old),10),len(old)-1])
    # Clamp endpoint commands exactly as the policy adapter does, then interpolate.
    endpoints=np.clip(nominal[indices],model.actuator_ctrlrange[:,0],model.actuator_ctrlrange[:,1])
    targets=np.column_stack([np.interp(np.arange(len(old)),indices,endpoints[:,i]) for i in range(12)])
    task=DinnerTask(model,data,manifest['layout'])
    task._select_item('left',next(x for x in manifest['layout']['objects'] if x['id']=='bottle'))
    task.grip_torque=GRIPPER_CAP_NM
    rows=[dict(a,target=t.tolist()) for a,t in zip(old,targets)]
    checks=[replay(model,initial,rows,task,hz) for hz in (200,20)]
    if not all(c['passed'] for c in checks):raise RuntimeError(json.dumps({'episode':name,'checks':checks}))
    folder.mkdir(parents=True);(folder/'images').mkdir()
    for file in ['scene.xml','initial-integration-state.npy']:shutil.copy2(source/file,folder/file)
    # Both source and destination episode folders have the same depth below ROOT.
    model.vis.quality.offsamples=0
    renderer=mujoco.Renderer(model,height=240,width=320);option=mujoco.MjvOption();option.geomgroup[3:]=0
    observations=[];count=0
    try:
        with (folder/'images.jsonl').open('w') as index:
            for i,row in enumerate(rows):
                if i%10==0:
                    if i%1000==0:require_space(out,32*1024**2)
                    obs=observation(data,i,len(observations));observations.append(obs)
                    for camera in CAMERAS:
                        renderer.update_scene(data,camera=camera,scene_option=option)
                        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
                        path=folder/'images'/f'{obs["index"]:06d}_{camera}.png'
                        Image.fromarray(renderer.render()).save(path,compress_level=1)
                        index.write(json.dumps({'observation_index':obs['index'],'action_index':i,'time_s':obs['time_s'],
                            'camera':camera,'path':path.relative_to(folder).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})+'\n')
                        count+=1
                row['ctrl']=apply_targets(model,data,row['target']).tolist()
                data.ctrl[:]=row['ctrl'];mujoco.mj_step(model,data)
                assert not np.any(data.xfrc_applied) and not np.any(data.qfrc_applied)
        observations.append(observation(data,len(rows),len(observations),True))
    finally:renderer.close()
    for filename,values in [('actions',rows),('observations',observations)]:
        with gzip.open(folder/(filename+'.jsonl.gz'),'wt',compresslevel=3) as f:
            for row in values:f.write(json.dumps(row,default=lambda x:x.tolist())+'\n')
    manifest.update(gripper_torque_cap_nm=GRIPPER_CAP_NM,replays=checks,training_eligible=True,
        action_count=len(rows),observation_count=len(observations),
        matched_controller={'implementation':'simulation_lab.policy_control.apply_targets','cap_nm':GRIPPER_CAP_NM,
                            'teacher_updates':0,'state_resets':1,'source_manifest_sha256':hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest()},
        images={'status':'completed','count':count,'index':'images.jsonl','hz':20,'resolution':[320,240],
                'cameras':CAMERAS,'observation_stride':1,'multisampling':0,'encoding':'lossless PNG; direct physics observation'})
    save(folder/'manifest.json',manifest)
    print(name,'matched capture complete',len(observations),checks,flush=True)
    return {'episode':name,'frames':len(observations)-1,'images':count,'replays':checks}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='.run/bottle-matched');p.add_argument('--cuda-renderer',action='store_true');a=p.parse_args()
    if a.cuda_renderer:
        import torch
        torch.cuda.init()
    out=Path(a.output).resolve()
    if out.parent!=ROOT/'.run':raise ValueError('Use a dataset directly under .run to preserve mesh references.')
    results=[collect(n,out) for n in FIT]
    save(ROOT/'docs/robotics/act-matched-capture.json',{'cap_nm':GRIPPER_CAP_NM,'episodes':results})
