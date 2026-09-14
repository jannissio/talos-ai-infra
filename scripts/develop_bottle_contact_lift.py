"""Bounded exposed sideways-bottle contact diagnosis and motor-only lift tests."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,hashlib,json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from simulation_lab.scene import build_scene
from simulation_lab.storage import require_space
from simulation_lab.table_teacher import TableTeacher,grasp_candidates
from simulation_lab.bottle_lift_probe import BottleLiftProbe


def contacts(model,data,task):
    rows=[];force_by={'fixed':np.zeros(3),'moving':np.zeros(3),'table':np.zeros(3),'other':np.zeros(3)}
    rotation=data.body('bottle').xmat.reshape(3,3);gripper=data.body(task.side+'_gripper')
    for index,c in enumerate(data.contact):
        a,b=int(c.geom1),int(c.geom2)
        if task.body not in (model.geom_bodyid[a],model.geom_bodyid[b]):continue
        other=b if model.geom_bodyid[a]==task.body else a
        group='fixed' if other in task.fixed_geoms else 'moving' if other in task.moving_geoms else 'table' if other==task.base_geom else 'other'
        wrench=np.zeros(6);mujoco.mj_contactForce(model,data,index,wrench)
        world_force=(1 if model.geom_bodyid[b]==task.body else -1)*c.frame.reshape(3,3).T@wrench[:3]
        force_by[group]+=world_force
        if wrench[0]>.001:
            rows.append({'group':group,'object_geom':model.geom(a if model.geom_bodyid[a]==task.body else b).name,
                'other_geom':model.geom(other).name or model.body(model.geom_bodyid[other]).name,
                'penetration_mm':-float(c.dist)*1000,'normal_force_n':float(wrench[0]),'force_world_n':world_force.tolist(),
                'point_bottle_local_m':(rotation.T@(c.pos-data.body('bottle').xpos)).tolist(),
                'point_gripper_local_m':(gripper.xmat.reshape(3,3).T@(c.pos-gripper.xpos)).tolist()})
    return {'time_s':float(data.time),'bottle_origin':data.body('bottle').xpos.tolist(),
        'bottle_axis':rotation[:,2].tolist(),'tool_axes':gripper.xmat.reshape(3,3).tolist(),
        'bottle_axis_in_gripper':(gripper.xmat.reshape(3,3).T@rotation[:,2]).tolist(),
        'force_by_group_n':{k:v.tolist() for k,v in force_by.items()},'contacts':rows}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--prefix',type=Path,default=ROOT/'.run/whole-table-development-v23-bottle4001/lookahead-00-010-00-bottle')
    parser.add_argument('--continuation',type=Path,default=ROOT/'.run/table-lift-physical-v1/continuation')
    parser.add_argument('--probe-pitch',type=float)
    parser.add_argument('--probe-band',type=float,default=.070)
    parser.add_argument('--geometry-only',action='store_true')
    parser.add_argument('--body-axis',action='store_true')
    parser.add_argument('--side',default='left')
    parser.add_argument('--tool-x',type=float,default=.017)
    parser.add_argument('--opening',type=float,default=.85)
    args=parser.parse_args();out=args.out.resolve();log=out.with_suffix('.log')
    if out.exists() or log.exists():raise FileExistsError('Preserve output and log.')
    preflight=require_space(out,128*1024**2);out.mkdir(parents=True)
    for p in (ROOT/'simulation_lab').glob('*.py'):
        dest=out/'source/simulation_lab'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    dest=out/'source/scripts';dest.mkdir();(dest/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    xml,layout=build_scene(seed=2026114001,scenario='dinner',dinner_preset='task');(out/'scene.xml').write_text(xml)
    model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model);spec=mujoco.mjtState.mjSTATE_INTEGRATION
    action=json.loads((args.prefix/'action.json').read_text())
    with np.load(args.prefix/'states.npz') as archive:prefix={k:archive[k] for k in ('initial_integration','ctrl','qpos','qvel','stage')}
    mujoco.mj_setState(model,data,prefix['initial_integration'],spec);mujoco.mj_forward(model,data)
    (out/'provenance.json').write_text(json.dumps({'source':str(args.prefix),
        'source_states_sha256':hashlib.sha256((args.prefix/'states.npz').read_bytes()).hexdigest(),
        'initial_qpos_matches_source':bool(np.array_equal(data.qpos,prefix['qpos'][0])),
        'initial_qvel_matches_source':bool(np.array_equal(data.qvel,prefix['qvel'][0])),
        'source_prefix_motor_steps_before_new_probe':0,'preflight':preflight},indent=2)+'\n')
    if args.probe_pitch is not None:
        task=BottleLiftProbe(model,data,layout);task.start_probe(args.probe_band,math.radians(args.probe_pitch),args.side,args.body_axis,args.tool_x,args.opening)
        targets=data.qpos[:12].copy();task.update(targets)
        (out/'search.json').write_text(json.dumps(task.search_log,indent=2)+'\n')
        if args.geometry_only or not task.active:
            result={'scope':'Geometry only; no new physical steps.','status':task.status,'message':task.message,'geometry_planned':task.active,'search':task.search_log}
            (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');log.write_text(json.dumps(result)+'\n');print(log.read_text());return
        require_space(out,256*1024**2)
        initial=np.empty(mujoco.mj_stateSize(model,spec));mujoco.mj_getState(model,data,initial,spec)
        qs=[data.qpos.copy()];vs=[data.qvel.copy()];controls=[];stages=[task.stage];samples=[];previous=None
        for tick in range(18000):
            if not task.active:break
            if previous!=task.stage:print({'stage':task.stage,'time_s':float(data.time)},flush=True);previous=task.stage
            task.update(targets)
            if not task.active:break
            data.ctrl[:]=task.apply_gripper_limit(targets)
            assert model.neq==0 and not np.any(data.xfrc_applied) and not np.any(data.qfrc_applied)
            controls.append(data.ctrl.copy());mujoco.mj_step(model,data)
            qs.append(data.qpos.copy());vs.append(data.qvel.copy());stages.append(task.stage)
            if tick%5==0:samples.append(dict(frame=tick+1,stage=task.stage,**contacts(model,data,task)))
        if task.active:task._finish('failed','Bounded lift-probe tick budget exhausted.',True)
        with (out/'states.npz').open('xb') as stream:np.savez_compressed(stream,initial_integration=initial,ctrl=np.array(controls),qpos=np.array(qs),qvel=np.array(vs),stage=np.array(stages))
        replay=mujoco.MjData(model);mujoco.mj_setState(model,replay,initial,spec);mujoco.mj_forward(model,replay)
        exact=True
        for i,ctrl in enumerate(controls):
            replay.ctrl[:]=ctrl;mujoco.mj_step(model,replay)
            exact &= np.array_equal(replay.qpos,qs[i+1]) and np.array_equal(replay.qvel,vs[i+1])
        result={'scope':'Exposed bottle lift component only; no complete bottle setting or whole-table claim.',
            'status':task.status,'message':task.message,'metrics':task.metrics,'history':task.history,
            'frames':len(qs),'replay_exact':bool(exact),'lift_component_success':task.status=='succeeded' and bool(exact),
            'complete_bottle_setting':False,'whole_table_complete':False,'demonstration_eligible':False}
        (out/'contacts.json').write_text(json.dumps(samples,indent=2)+'\n');(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        log.write_text(json.dumps(result)+'\n');print(log.read_text());return
    candidate=next(x for x in grasp_candidates(data,'bottle') if x.name==action['grasp_candidate'])
    task=TableTeacher(model,data,layout);task.start(object_id='bottle',side=action['arm'])
    task._select_item(action['arm'],next(x for x in layout['objects'] if x['id']=='bottle'));task._configure(candidate)
    samples=[]
    for i,ctrl in enumerate(prefix['ctrl']):
        data.ctrl[:]=ctrl;mujoco.mj_step(model,data)
        assert np.array_equal(data.qpos,prefix['qpos'][i+1]) and np.array_equal(data.qvel,prefix['qvel'][i+1])
        if prefix['stage'][i+1]=='close' and i%30==0:samples.append(dict(segment='source_close',frame=i+1,**contacts(model,data,task)))
    initial=contacts(model,data,task)
    with np.load(args.continuation/'states.npz') as archive:continuation={k:archive[k] for k in ('ctrl','qpos','qvel')}
    assert np.array_equal(data.qpos,continuation['qpos'][0])
    for i,ctrl in enumerate(continuation['ctrl']):
        data.ctrl[:]=ctrl;mujoco.mj_step(model,data)
        assert np.array_equal(data.qpos,continuation['qpos'][i+1]) and np.array_equal(data.qvel,continuation['qvel'][i+1])
        if i%5==0 or i==len(continuation['ctrl'])-1:samples.append(dict(segment='failed_lift',frame=i+1,**contacts(model,data,task)))
    report={'scope':'Exact replay diagnosis only; no new manipulation trial.','preflight':preflight,'prefix_replay_exact':True,
        'continuation_replay_exact':True,'prefix_frames':len(prefix['ctrl']),'continuation_frames':len(continuation['ctrl']),
        'bottle_mass_kg':float(model.body('bottle').mass[0]),'bottle_center_of_mass_local':model.body('bottle').ipos.tolist(),
        'source_grasp':action,'initial_contact':initial,'samples':samples,
        'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.prefix/'states.npz',args.continuation/'states.npz')}}
    (out/'diagnosis.json').write_text(json.dumps(report,indent=2)+'\n');log.write_text(json.dumps({'prefix_replay_exact':True,'continuation_replay_exact':True,'samples':len(samples)})+'\n');print(log.read_text())


if __name__=='__main__':main()
