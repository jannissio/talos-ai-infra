"""Budgeted new demonstrations and frozen evaluation starts. Never rewrites old data."""
import argparse,gzip,hashlib,json,shutil,sys,xml.etree.ElementTree as ET
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco,numpy as np
from PIL import Image
from scripts.prepare_bottle_data import STATE,state,observation,replay,save
from simulation_lab.dinner_autonomy import DinnerTask
from simulation_lab.autonomy import Motion,PlanningError
from simulation_lab.scene import HOME
from simulation_lab.policy_control import apply_targets,GRIPPER_CAP_NM
from simulation_lab.storage import require_space,GIB

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'.run/bottle-robustness-data'
PROTOCOL=ROOT/'docs/robotics/bottle-robustness-protocol.json'
CAMERAS=['overhead','left_wrist_cam','right_wrist_cam']

def budget(additional=32*1024**2):
    protocol=json.loads(PROTOCOL.read_text())
    total=0
    for root in (ROOT/'.run').glob('bottle-robustness*'):
        total+=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) if root.is_dir() else root.stat().st_size
    if total+additional>protocol['budget_bytes']:raise OSError('Robustness milestone artifact budget exhausted.')
    require_space(OUT,additional,protocol['minimum_free_bytes'])
    return total

def prepare():
    protocol=json.loads(PROTOCOL.read_text());anchor=ROOT/protocol['anchor']
    if OUT.exists():raise FileExistsError(OUT)
    OUT.mkdir();budget()
    for split in ['training','development','held_out']:
        for spec in protocol[split]:
            folder=OUT/spec['id'];folder.mkdir()
            tree=ET.parse(anchor/'scene.xml')
            for node in list(tree.iter('light'))+list(tree.iter('headlight')):
                if node.get('diffuse') and spec.get('light',1)!=1:
                    node.set('diffuse',' '.join(str(float(v)*spec['light']) for v in node.get('diffuse').split()))
            tree.write(folder/'scene.xml',encoding='unicode')
            model=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'));data=mujoco.MjData(model)
            initial=ROOT/'.run/bottle-robustness-recovery-source.npy' if 'recovery_from' in spec else anchor/'initial-integration-state.npy'
            mujoco.mj_setState(model,data,np.load(initial,allow_pickle=False),STATE)
            adr=model.joint('bottle_free').qposadr[0]
            if 'recovery_from' not in spec:data.qpos[adr:adr+2]+=[spec['dx'],spec['dy']]
            mujoco.mj_forward(model,data)
            np.save(folder/'initial-integration-state.npy',state(model,data),allow_pickle=False)
            manifest=json.loads((anchor/'manifest.json').read_text())
            manifest.update(id=spec['id'],split=split,training_eligible=False,trajectory_complete=False,
                            images={'status':'pending' if split=='training' else 'not_captured'},
                            robustness_spec=spec,protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest())
            manifest['layout']['bottle_reset']={'settled_qpos':data.qpos[adr:adr+7].tolist(),'spec':spec}
            for key in ['outcome','replays','matched_controller']:manifest.pop(key,None)
            save(folder/'manifest.json',manifest)
    print('Prepared frozen training, development and held-out initial states.',flush=True)

def write_rows(folder,name,rows):
    with gzip.open(folder/(name+'.jsonl.gz'),'wt',encoding='utf-8',compresslevel=3) as f:
        for row in rows:f.write(json.dumps(row,default=lambda a:a.tolist(),separators=(',',':'))+'\n')

def collect(name,recover=False):
    folder=OUT/name;manifest=json.loads((folder/'manifest.json').read_text())
    if manifest['split']!='training':raise ValueError('Only frozen training starts may produce demonstrations.')
    if (folder/'teacher-result.json').exists():
        previous=json.loads((folder/'teacher-result.json').read_text())
        if not recover or previous['outcome']['status']!='failed':raise FileExistsError('Preserve prior attempt: '+name)
        preserved=folder/'teacher-result-attempt0.json'
        if preserved.exists():raise FileExistsError(preserved)
        (folder/'teacher-result.json').rename(preserved)
    budget(160*1024**2)
    model=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'));data=mujoco.MjData(model)
    initial=np.load(folder/'initial-integration-state.npy',allow_pickle=False)
    mujoco.mj_setState(model,data,initial,STATE);mujoco.mj_forward(model,data)
    assert model.neq==0
    task=DinnerTask(model,data,manifest['layout'])
    target=data.qpos[:12].copy();actions=[]
    if recover:
        task._select_item('left',next(o for o in manifest['layout']['objects'] if o['id']=='bottle'))
        forces,_,up,_=task._observe()
        if forces['base']<.02 or up<np.cos(np.deg2rad(5)):raise PlanningError('Recovery requires the observed upright, table-supported failure.')
        def move(stage,points,duration):
            gradient=float(np.abs(np.diff(points,axis=0)).max())*(len(points)-1)
            duration=max(duration,1.875*gradient/.8)
            motion=Motion(np.asarray(points),duration,float(data.time),float(data.qpos[5]),HOME[5])
            for _ in range(int(np.ceil(duration/.005))+1):
                i=len(actions);target[:6]=motion.sample(float(data.time))
                ctrl=apply_targets(model,data,target)
                actions.append({'index':i,'time_s':round(i*.005,9),'stage':stage,'target':target.copy(),'ctrl':ctrl.copy()})
                data.ctrl[:]=ctrl;mujoco.mj_step(model,data)
                collision=task._collision(data,True,.0008)
                if collision:raise PlanningError('Recovery collision: '+collision)
                assert not np.any(data.xfrc_applied) and not np.any(data.qfrc_applied)
        q=data.qpos[:5].copy();move('recovery_open',np.vstack([q,q]),2.)
        point=task.ik.point(data).copy()
        path=task._cartesian(point,point+[0,0,.055],data.qpos[:5].copy(),HOME[5])
        move('recovery_retreat',path,3.)
        path=task._joint_path(data.qpos[:5].copy(),np.array(HOME[:5]),HOME[5])
        move('recovery_home',path,4.)
        move('recovery_settle',np.vstack([HOME[:5],HOME[:5]]),1.)
    task.start(side='left',object_id='bottle')
    for i in range(20000):
        before=data.qpos.copy();velocity=data.qvel.copy()
        task.update(target)
        assert np.array_equal(before,data.qpos) and np.array_equal(velocity,data.qvel)
        task.grip_torque=GRIPPER_CAP_NM
        task.metrics['gripper_torque_limit_nm']=GRIPPER_CAP_NM
        if not task.active:break
        ctrl=apply_targets(model,data,target)
        index=len(actions)
        actions.append({'index':index,'time_s':round(index*.005,9),'stage':task.stage,'target':target.copy(),'ctrl':ctrl.copy()})
        data.ctrl[:]=ctrl;mujoco.mj_step(model,data)
        assert not np.any(data.xfrc_applied) and not np.any(data.qfrc_applied)
    result={'id':name,'outcome':task.snapshot(),'actions':len(actions),'cap_nm':GRIPPER_CAP_NM}
    if task.status!='succeeded':
        save(folder/'teacher-result.json',result);print(name,'teacher failed:',task.message,flush=True);return result
    # Exactly the same sampled nominal-target interpolation contract as existing training data.
    raw=np.asarray([r['target'] for r in actions]);indices=np.unique(np.r_[np.arange(0,len(raw),10),len(raw)-1])
    endpoints=np.clip(raw[indices],model.actuator_ctrlrange[:,0],model.actuator_ctrlrange[:,1])
    targets=np.column_stack([np.interp(np.arange(len(raw)),indices,endpoints[:,j]) for j in range(12)])
    actions=[dict(r,target=t) for r,t in zip(actions,targets)]
    checks=[replay(model,initial,actions,task,hz) for hz in (200,20)]
    result['replays']=checks
    if not all(c['passed'] for c in checks):
        save(folder/'teacher-result.json',result);print(name,'replay failed',flush=True);return result
    # Capture only after both physical replays pass; direct post-physics RGB, no state reconstruction.
    mujoco.mj_setState(model,data,initial,STATE);mujoco.mj_forward(model,data)
    model.vis.quality.offsamples=0
    renderer=mujoco.Renderer(model,height=240,width=320);option=mujoco.MjvOption();option.geomgroup[3:]=0
    (folder/'images').mkdir();observations=[];count=0
    try:
        from OpenGL.GL import glGetString,GL_RENDERER
        result['renderer']=(glGetString(GL_RENDERER) or b'unknown').decode()
        with (folder/'images.jsonl').open('w') as index:
            for i,row in enumerate(actions):
                if i%10==0:
                    if i%1000==0:budget()
                    obs=observation(data,i,len(observations));observations.append(obs)
                    for camera in CAMERAS:
                        renderer.update_scene(data,camera=camera,scene_option=option)
                        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
                        path=folder/'images'/f'{obs["index"]:06d}_{camera}.png'
                        Image.fromarray(renderer.render()).save(path,compress_level=3)
                        index.write(json.dumps({'observation_index':obs['index'],'action_index':i,'time_s':obs['time_s'],
                            'camera':camera,'path':path.relative_to(folder).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})+'\n');count+=1
                row['ctrl']=apply_targets(model,data,row['target']);data.ctrl[:]=row['ctrl'];mujoco.mj_step(model,data)
                assert not np.any(data.xfrc_applied) and not np.any(data.qfrc_applied)
        observations.append(observation(data,len(actions),len(observations),True))
    finally:renderer.close()
    write_rows(folder,'actions',actions);write_rows(folder,'observations',observations)
    manifest.update(trajectory_complete=True,training_eligible=True,action_count=len(actions),observation_count=len(observations),
        arm='left',gripper_torque_cap_nm=GRIPPER_CAP_NM,replays=checks,outcome=task.snapshot(),
        matched_controller={'implementation':'simulation_lab.policy_control.apply_targets','cap_nm':GRIPPER_CAP_NM,
            'teacher_updates_during_capture':0,'state_resets_during_capture':1,'teacher_used_for_labels':True},
        images={'status':'completed','index':'images.jsonl','count':count,'hz':20,'resolution':[320,240],
            'cameras':CAMERAS,'observation_stride':1,'multisampling':0,'encoding':'lossless PNG; direct physics observation'})
    save(folder/'manifest.json',manifest);save(folder/'teacher-result.json',result)
    print(name,'verified and captured',len(observations)-1,'frames',flush=True)
    return result

def lineage():
    old=json.loads((ROOT/'.run/act-matched-nvidia-fit5/talos_lineage.json').read_text())
    for row in old['episodes']:row['folder']=str(ROOT/'.run/bottle-matched-nvidia'/row['source'])
    for spec in json.loads(PROTOCOL.read_text())['training']:
        folder=OUT/spec['id'];manifest=json.loads((folder/'manifest.json').read_text())
        if manifest['training_eligible']:
            old['episodes'].append({'episode_index':len(old['episodes']),'source':spec['id'],'folder':str(folder),
                                   'frames':manifest['observation_count']-1,'gripper_cap_nm':GRIPPER_CAP_NM,
                                   'source_manifest_sha256':hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest()})
    old['split']='robustness_training_only'
    old['robustness_protocol_sha256']=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    save(OUT/'fit-lineage.json',old);print('Fit episodes',len(old['episodes']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','collect','recover','lineage']);p.add_argument('--episode')
    a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='lineage':lineage()
    else:
        import torch
        torch.cuda.init()  # Match the existing NVIDIA camera-rendering contract.
        names=[a.episode] if a.episode else [r['id'] for r in json.loads(PROTOCOL.read_text())['training']]
        for name in names:collect(name,recover=a.command=='recover')
