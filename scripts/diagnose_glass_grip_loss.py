"""Replay immutable motor commands; export actual contact forces without trial edits."""
import argparse, contextlib, hashlib, json, os, sys, time
from pathlib import Path
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from simulation_lab.storage import require_space
from scripts.develop_side_plate_grasp import put

def run(args):
    out=args.output.resolve();log=out.with_suffix('.log');source=args.source.resolve()
    if out.exists() or log.exists():raise FileExistsError('Output and log must both be unused.')
    preflight=require_space(out,128*1024**2);out.mkdir(parents=True)
    with log.open('x') as stream,contextlib.redirect_stdout(stream):
        started=time.perf_counter();manifest={}
        for path in (Path(__file__),ROOT/'simulation_lab/storage.py',ROOT/'scripts/develop_side_plate_grasp.py'):
            payload=path.read_bytes();name=path.relative_to(ROOT).as_posix();put(out/'source'/name,payload);manifest[name]=hashlib.sha256(payload).hexdigest()
        for name in ('scene.xml','parameters.json','result.json','source-manifest.json'):
            payload=(source/name).read_bytes();put(out/'input'/name,payload);manifest['input/'+name]=hashlib.sha256(payload).hexdigest()
        manifest['input/states.npz']=hashlib.sha256((source/'states.npz').read_bytes()).hexdigest()
        put(out/'source-manifest.json',manifest)
        with np.load(source/'states.npz') as archive:saved={key:archive[key] for key in ('initial_integration','ctrl','qpos','qvel','stage')}
        prefix=json.loads((source/'result.json').read_text())['prefix_motor_frames']
        m=mujoco.MjModel.from_xml_string((source/'scene.xml').read_text());d=mujoco.MjData(m)
        mujoco.mj_setState(m,d,saved['initial_integration'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
        glass=m.body('glass').id;tool=m.body('right_gripper').id;moving=m.body('right_moving_jaw_so101_v1').id
        jid=m.joint('right_gripper').id;qadr=m.jnt_qposadr[jid];dadr=m.jnt_dofadr[jid];aid=m.actuator('right_gripper').id
        rows=[];error=0.;renderer=None;images=[]
        snapshots={2305:'lift-start',2565:'lift-1.3s',2705:'lift-2.0s',2795:'lift-2.45s',2865:'lift-2.8s'}
        for i,ctrl in enumerate(saved['ctrl']):
            d.ctrl[:]=ctrl;mujoco.mj_step(m,d)
            error=max(error,float(np.max(np.abs(d.qpos-saved['qpos'][i+1]))),float(np.max(np.abs(d.qvel-saved['qvel'][i+1]))))
            if i<prefix:continue
            n=i-prefix;stage=str(saved['stage'][i+1])
            if n<1400:continue
            rg=d.xmat[glass].reshape(3,3);rt=d.xmat[tool].reshape(3,3)
            contacts=[];total=np.zeros(6)
            for k,c in enumerate(d.contact):
                a,b=int(c.geom1),int(c.geom2)
                if m.geom_bodyid[a]!=glass and m.geom_bodyid[b]!=glass:continue
                sign=1 if m.geom_bodyid[b]==glass else -1;other=a if sign==1 else b
                force=np.zeros(6);mujoco.mj_contactForce(m,d,k,force)
                frame=c.frame.reshape(3,3);normal=sign*frame[0];worldforce=sign*frame.T@force[:3]
                worldtorque=sign*frame.T@force[3:]+np.cross(c.pos-d.xipos[glass],worldforce)
                total+=np.r_[worldforce,worldtorque]
                category='fixed' if m.geom_bodyid[other]==tool else 'moving' if m.geom_bodyid[other]==moving else 'external'
                jp=np.zeros((3,m.nv));jr=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jp,jr,c.pos,glass)
                vg=jp@saved['qvel'][i];mujoco.mj_jac(m,d,jp,jr,c.pos,int(m.geom_bodyid[other]));relative_velocity=vg-jp@saved['qvel'][i]
                contacts.append(dict(jaw=category,geom_other=m.geom(other).name or str(other),geom_glass=m.geom(b if sign==1 else a).name,position_world=c.pos.tolist(),position_glass=(rg.T@(c.pos-d.xpos[glass])).tolist(),position_tool=(rt.T@(c.pos-d.xpos[tool])).tolist(),position_jaw=(d.xmat[m.geom_bodyid[other]].reshape(3,3).T@(c.pos-d.xpos[m.geom_bodyid[other]])).tolist(),normal_on_glass=normal.tolist(),force_world=worldforce.tolist(),normal_force=float(force[0]),friction_world=(worldforce-normal*force[0]).tolist(),torque_about_com_world=worldtorque.tolist(),relative_velocity_world=relative_velocity.tolist(),friction=c.friction.tolist(),dim=int(c.dim),distance=float(c.dist),cone_usage=float(np.sqrt(np.sum((force[1:]/np.maximum(c.friction,1e-12))**2))/max(force[0],1e-12))))
            rows.append(dict(step=n,time=n*m.opt.timestep,stage=stage,glass_origin=d.xpos[glass].tolist(),glass_origin_in_tool=(rt.T@(d.xpos[glass]-d.xpos[tool])).tolist(),glass_up_in_tool=(rt.T@rg[:,2]).tolist(),tilt_deg=float(np.degrees(np.arccos(np.clip(rg[2,2],-1,1)))),grip_q=float(saved['qpos'][i,qadr]),grip_qvel=float(saved['qvel'][i,dadr]),grip_ctrl=float(ctrl[aid]),grip_actuator_force=float(d.actuator_force[aid]),grip_qfrc_constraint=float(d.qfrc_constraint[dadr]),contact_wrench_world=total.tolist(),contacts=contacts))
            if n in snapshots:
                try:
                    from PIL import Image
                    if renderer is None:renderer=mujoco.Renderer(m,height=600,width=800)
                    camera=mujoco.MjvCamera();camera.lookat[:]=d.xpos[glass]+[0,0,.045];camera.distance=.34;camera.azimuth=115;camera.elevation=-12
                    option=mujoco.MjvOption();option.geomgroup[3:]=0
                    renderer.update_scene(d,camera=camera,scene_option=option)
                    path=out/(snapshots[n]+'.png');require_space(path,4*1024**2)
                    with path.open('xb') as imagefile:Image.fromarray(renderer.render()).save(imagefile,format='PNG')
                    images.append(path.name)
                except Exception as exc:print('Render error',repr(exc),flush=True)
        if renderer is not None:renderer.close()
        put(out/'contacts.json',rows)
        geom_info=[]
        for g in range(m.ngeom):
            if m.geom_bodyid[g] in (glass,tool,moving) and (m.geom_contype[g] or m.geom_conaffinity[g]):geom_info.append(dict(name=m.geom(g).name or str(g),body=m.body(int(m.geom_bodyid[g])).name,type=int(m.geom_type[g]),position=m.geom_pos[g].tolist(),size=m.geom_size[g].tolist(),friction=m.geom_friction[g].tolist(),priority=int(m.geom_priority[g])))
        report=dict(scope='Exact saved motor replay only; no new manipulation trial, physics/model/state edits after initial reset. Contact force/pose belongs to pre-integration dynamics state at reported continuation time; qpos/qvel replay gate checks post-integration state.',source=str(source),preflight=preflight,replay_exact=error==0,replay_max_state_error=error,motor_steps=len(saved['ctrl']),glass_mass=float(m.body_mass[glass]),glass_com_local=m.body_ipos[glass].tolist(),gravity=m.opt.gravity.tolist(),gripper_actuator_gain=m.actuator_gainprm[aid].tolist(),gripper_force_limit=m.actuator_forcerange[aid].tolist(),geom_info=geom_info,rows=len(rows),images=images,wall_s=time.perf_counter()-started)
        put(out/'report.json',report);print(json.dumps({k:v for k,v in report.items() if k!='geom_info'}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=Path('.run/glass-placement-v17-proof004-lower-band'));p.add_argument('--output',type=Path,required=True);run(p.parse_args())
