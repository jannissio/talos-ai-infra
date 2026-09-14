"""Retained airborne regrasp experiments from an exactly replayed physical prefix."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
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
from simulation_lab.scene import build_scene
from simulation_lab.storage import require_space
from simulation_lab.table_teacher import grasp_candidates
from simulation_lab.table_regrasp import TableRegrasp, coupled_pose_search


def put(path,value):
    content=value if isinstance(value,bytes) else (json.dumps(value,indent=2)+'\n').encode()
    require_space(path,len(content)+1024)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:stream.write(content)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--source',type=Path,default=ROOT/'.run/whole-table-development-v15-seed4002/lookahead-00-015-02-spoon')
    parser.add_argument('--seed',type=int,default=2026114002)
    parser.add_argument('--geometry-only',action='store_true')
    parser.add_argument('--endpoint-only',action='store_true')
    args=parser.parse_args(); out=args.out.resolve(); log=out.with_suffix('.log')
    if out.exists() or log.exists():raise FileExistsError('Both output and log must be unused.')
    preflight=require_space(out,512*1024**2); out.mkdir(parents=True)
    with log.open('x',encoding='utf-8') as console:
        def report(value):
            print(value,flush=True); console.write(json.dumps(value)+'\n'); console.flush()
        for path in (ROOT/'simulation_lab').glob('*.py'):put(out/'source/simulation_lab'/path.name,path.read_bytes())
        put(out/'source/scripts/develop_table_regrasp.py',Path(__file__).read_bytes())
        xml,layout=build_scene(seed=args.seed,scenario='dinner',dinner_preset='task')
        put(out/'scene.xml',xml.encode()); put(out/'layout.json',layout)
        model=mujoco.MjModel.from_xml_string(xml); data=mujoco.MjData(model)
        action=json.loads((args.source/'action.json').read_text())
        with np.load(args.source/'states.npz') as archive:
            values={key:archive[key] for key in ('stage','initial_integration','ctrl','qpos','qvel')}
        index=int(np.flatnonzero(values['stage']=='hold')[-1])
        state_spec=mujoco.mjtState.mjSTATE_INTEGRATION
        mujoco.mj_setState(model,data,values['initial_integration'],state_spec); mujoco.mj_forward(model,data)
        for i,ctrl in enumerate(values['ctrl'][:index]):
            data.ctrl[:]=ctrl; mujoco.mj_step(model,data)
            assert np.array_equal(data.qpos,values['qpos'][i+1])
            assert np.array_equal(data.qvel,values['qvel'][i+1])
        provenance={'source':str(args.source),'source_states_sha256':hashlib.sha256((args.source/'states.npz').read_bytes()).hexdigest(),
            'prefix_frames':index,'prefix_replay_exact':True,'preflight':preflight,
            'scope':'Exposed bimanual cutlery dependency; not whole-table coverage.'}
        put(out/'provenance.json',provenance)
        task=TableRegrasp(model,data,layout)
        task.start(side=action['arm'],object_id=action['item'])
        item=next(x for x in layout['objects'] if x['id']==action['item'])
        task._select_item(action['arm'],item)
        candidate=next(x for x in grasp_candidates(data,action['item']) if x.name==action['grasp_candidate'])
        task._configure(candidate); task.chosen=candidate
        q=data.qpos[task.offset:task.offset+5].copy()
        task._move('hold',np.array([q,q]),-.170,0.)
        task.motion.started=float(data.time)-1.
        targets=data.ctrl.copy(); targets[task.offset+5]=-.170
        if args.geometry_only:
            found,rows=coupled_pose_search(task,check_routes=not args.endpoint_only)
            put(out/'geometry.json',{'found':len(found),'checks':rows,'physical_success':False})
            report({'geometry_candidates':len(found),'checks':len(rows)})
            return
        require_space(out,384*1024**2)
        initial=np.empty(mujoco.mj_stateSize(model,state_spec)); mujoco.mj_getState(model,data,initial,state_spec)
        controls=[]; qpos=[data.qpos.copy()]; qvel=[data.qvel.copy()]; stages=[task.stage]; forces=[];finger_forces=[]
        began=time.perf_counter(); previous=None
        try:
            for tick in range(36000):
                if not task.active:break
                if previous!=task.stage:report({'stage':task.stage,'time_s':float(data.time)});previous=task.stage
                task.update(targets)
                if not task.active:break
                data.ctrl[:]=task.apply_gripper_limit(targets)
                assert model.neq==0 and not np.any(data.xfrc_applied) and not np.any(data.qfrc_applied)
                controls.append(data.ctrl.copy())
                mujoco.mj_step(model,data)
                qpos.append(data.qpos.copy());qvel.append(data.qvel.copy());stages.append(task.stage)
                forces.append(data.actuator_force.copy())
                finger_forces.append([*task.metrics.get('donor_finger_forces_n',(0.,0.)),
                                      *task.metrics.get('receiver_finger_forces_n',(0.,0.))])
            if task.active:task._finish('failed','Bimanual development tick budget exhausted.',True)
        except Exception as exc:
            task._finish('failed',f'{type(exc).__name__}: {exc}',True)
        finally:
            require_space(out,128*1024**2)
            with (out/'states.npz').open('xb') as stream:
                np.savez_compressed(stream,initial_integration=initial,ctrl=np.asarray(controls),qpos=np.asarray(qpos),qvel=np.asarray(qvel),stage=np.asarray(stages),actuator_force=np.asarray(forces),pre_step_finger_forces=np.asarray(finger_forces))
            put(out/'search.json',task.handoff_log)
            put(out/'action.json',task.action_metadata())
            replay=mujoco.MjData(model);mujoco.mj_setState(model,replay,initial,state_spec);mujoco.mj_forward(model,replay)
            maximum=0.
            for i,ctrl in enumerate(controls):
                replay.ctrl[:]=ctrl;mujoco.mj_step(model,replay)
                maximum=max(maximum,float(np.max(np.abs(replay.qpos-qpos[i+1]))),float(np.max(np.abs(replay.qvel-qvel[i+1]))))
            result={'status':task.status,'message':task.message,'metrics':task.metrics,'history':task.history,
                'frames':len(qpos),'replay_exact':maximum==0.,'replay_max_error':maximum,'wall_seconds':time.perf_counter()-began,
                'physical_success':task.status=='succeeded' and maximum==0.,'whole_table_complete':False}
            put(out/'result.json',result);report(result)


if __name__=='__main__':main()
