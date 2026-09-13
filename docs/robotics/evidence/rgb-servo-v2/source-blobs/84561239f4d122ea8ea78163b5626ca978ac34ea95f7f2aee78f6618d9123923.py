"""Physical evaluation of route-aware RGB control, retaining every refusal."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from PIL import Image
import torch
from simulation_lab.dinner_monitor import DinnerPhysicalMonitor
from simulation_lab.experiment_targets import with_bottle_destination
from simulation_lab.policy_control import apply_targets
from simulation_lab.rgb_servo_cameras import VIEWS,camera_argument,calibration
from simulation_lab.rgb_servo_openvino import OpenVinoBottleObserver,OpenVinoCartesianMotorPolicy
from simulation_lab.rgb_servo_routing import RoutedRgbServoBottle,plan_bottle_route
from simulation_lab.scene import HOME,build_scene
from simulation_lab.storage import require_space


def run(args):
    if args.output.exists():raise FileExistsError(args.output)
    protocol=json.loads(args.protocol.read_text());routing=json.loads(args.routing.read_text())
    allowed=(range(protocol['training_seed_range'][0],protocol['training_seed_range'][1]+1)
             if args.split=='training' else protocol[args.split+'_seeds'])
    if args.seed not in allowed:raise ValueError('Seed is outside the declared split.')
    require_space(args.output,16*1024**2);args.output.mkdir(parents=True)
    root=Path(__file__).resolve().parents[1]
    sources=list(json.loads((root/'docs/robotics/experiments/rgb-servo-final-evaluation-v1.json').read_text())['source_sha256'])
    sources+=['scripts/evaluate_rgb_servo_routing.py','simulation_lab/rgb_servo_routing.py']
    hashes={}
    for name in sources:
        payload=(root/name).read_bytes();out=args.output/'evaluated-source'/name
        require_space(out,len(payload)+1024**2);out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(payload)
        hashes[name]=hashlib.sha256(payload).hexdigest()
    rng=np.random.default_rng(args.seed);bounds=protocol['workspace_m']
    x,y,yaw=rng.uniform(*bounds['x']),rng.uniform(*bounds['y']),rng.uniform(-.6,.6)
    destination=protocol['destinations_m'][args.seed%3]
    xml,layout=build_scene(seed=args.seed,scenario='dinner',dinner_preset='task')
    model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model);model.vis.quality.offsamples=0
    data.qpos[:12]=HOME*2;data.ctrl[:]=HOME*2
    address=int(model.joint('bottle_free').qposadr[0]);body=model.body('bottle').id
    data.qpos[address:address+7]=[x,y,layout['table_z']+.001,math.cos(yaw/2),0.,0.,math.sin(yaw/2)]
    mujoco.mj_forward(model,data)
    overlap=max([-c.dist for c in data.contact if c.dist<0 and body in [model.geom_bodyid[c.geom1],model.geom_bodyid[c.geom2]]],default=0.)
    for _ in range(300):mujoco.mj_step(model,data)
    data.time=0.;scene=ET.fromstring(xml);compiler=scene.find('compiler')
    compiler.set('meshdir',os.path.relpath(compiler.get('meshdir'),args.output).replace('\\','/'))
    (args.output/'scene.xml').write_text(ET.tostring(scene,encoding='unicode'))
    report={'seed':args.seed,'split':args.split,'mode':args.mode,'initial_pose':[x,y,yaw],'destination_m':destination,
            'source_sha256':hashes,'source_parent_git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'protocol_sha256':hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
            'routing_sha256':hashlib.sha256(args.routing.read_bytes()).hexdigest(),
            'observer_sha256':hashlib.sha256((args.model/'observer.safetensors').read_bytes()).hexdigest(),
            'motor_sha256':hashlib.sha256((args.model/'motor.safetensors').read_bytes()).hexdigest(),
            'openvino_artifact_sha256':{name:hashlib.sha256((args.model/'openvino'/name).read_bytes()).hexdigest() for name in ('observer.xml','observer.bin','motor.xml','motor.bin')},
            'initial_overlap_m':float(overlap),'layout':layout,'inference_runtime':'OpenVINO '+args.device,
            'physics_state_writes_during_control':0,'hidden_forces':0,'inverse_solver_calls_during_control':0,
            'ablation':'Initial image frozen separately for each leg; next leg acquires fresh RGB only after physical release and parking.',
            'scope':'Route-aware upright bottle experiment; approach/descent RGB, proprioceptive carry, optional table-supported relay.'}
    if overlap>.001 or data.body('bottle').xmat[8]<.98:
        report.update(status='invalid_start',message='Initial overlap or tipped bottle; refused.',physical=None,simulation_seconds=0.)
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');return
    torch.set_num_threads(2)
    observer=OpenVinoBottleObserver(args.model/'openvino/observer.xml',minimum_views=2,device=args.device,rigid_geometry=True)
    motor=OpenVinoCartesianMotorPolicy(args.model/'openvino/motor.xml',args.model/'motor.safetensors',model,device=args.device)
    renderer=mujoco.Renderer(model,width=320,height=240);option=mujoco.MjvOption();option.geomgroup[3:]=0
    trace={k:[] for k in ('qpos','qvel','time','stage','targets')};observations=[];counterfactuals=[]
    target=np.asarray(HOME*2,dtype=float);controller=None;monitor=None;route=None;completed=[];leg_index=0;epoch=0.
    initial_others={o['id']:data.body(o['id']).xpos.copy() for o in layout['objects'] if o['id']!='bottle'}
    max_other=0.;began=time.perf_counter();image_seconds=0.;status='failed';message='Global trial timeout.'
    push=json.loads(args.push.read_text()) if args.push else None
    push_origin=None;push_settled=None;push_peak=0.;push_ticks=0;last_controller=None
    try:
        for tick in range(40001):
            now=float(data.time)
            if tick%20==0:
                started=time.perf_counter();images={};calibrations={}
                for name in VIEWS:
                    renderer.update_scene(data,camera=camera_argument(name),scene_option=option)
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
                    images[name]=renderer.render().copy();calibrations[name]=calibration(renderer)
                estimate=observer.observe(images,calibrations);image_seconds+=time.perf_counter()-started
                observations.append({'time_s':now,'leg':leg_index,'status':estimate['status'],
                    'grasp_point_m':estimate.get('grasp_point_m'),'reason':estimate.get('reason'),'used_views':estimate.get('used_views')})
                if tick%5000==0:
                    require_space(args.output,8*1024**2);Image.fromarray(np.concatenate(list(images.values()),axis=1)).save(args.output/f'views-{tick:05d}.png')
                if route is None and estimate['status']=='observed':
                    try:route=plan_bottle_route(motor,estimate['grasp_point_m'],destination,layout['table_z'],args.mode,routing)
                    except ValueError as exc:message=str(exc);break
                if route and controller is None and estimate['status']=='observed':
                    leg=route['legs'][leg_index];epoch=now
                    controller=RoutedRgbServoBottle(motor,leg['destination'],layout['table_z'],args.mode,routing,leg['side'])
                    monitor=DinnerPhysicalMonitor(model,data,with_bottle_destination(layout,leg['destination']),'bottle',leg['side'])
                    assert np.allclose(monitor.destination[:2],leg['destination'],atol=1e-12)
                if controller:
                    controller.accept_observation(estimate,now-epoch)
                    if estimate['status']=='observed' and controller.stage in ('approach','descend'):
                        try:
                            a=controller.preview_image_action(controller.current_rgb_estimate,now-epoch,data.qpos[:12].copy())
                            b=controller.preview_image_action(controller.initial_observation,now-epoch,data.qpos[:12].copy())
                            counterfactuals.append({'time_s':now,'leg':leg_index,'stage':controller.stage,
                                'same_measured_joint_positions':data.qpos[:12].tolist(),'live_image_action':a.tolist(),
                                'initial_image_action':b.tolist(),'max_action_delta_rad':float(np.max(np.abs(a-b)))})
                        except ValueError as exc:counterfactuals.append({'time_s':now,'leg':leg_index,'refused':str(exc)})
            q,v=data.qpos.copy(),data.qvel.copy()
            if controller:target=controller.update(now-epoch,data.qpos[:12].copy(),data.qvel[:12].copy())
            assert np.array_equal(q,data.qpos) and np.array_equal(v,data.qvel)
            stage=f'leg-{leg_index+1}:'+controller.stage if controller else 'observe'
            if tick%10==0:
                for key,value in (('qpos',q),('qvel',v),('time',now),('stage',stage),('targets',target.copy())):trace[key].append(value)
            max_other=max(max_other,max((float(np.linalg.norm(data.body(name).xpos-position)) for name,position in initial_others.items()),default=0.))
            failure=monitor.update() if monitor else None
            if max_other>.004:failure='Cumulative non-target displacement exceeded 4 mm.'
            if failure:message=failure;break
            if monitor and monitor.succeeded:
                completed.append({'leg':leg_index,'side':controller.side,'destination':route['legs'][leg_index]['destination'],
                    'physical':monitor.report(),'route_details':controller.route_details,'stages':controller.history,
                    'rgb_observations_accepted':controller.observations,'rgb_refusals':controller.refusals})
                last_controller=controller;leg_index+=1
                if leg_index==len(route['legs']):status='succeeded';message='Every leg physically released and parked.';break
                controller=None;monitor=None;target=data.qpos[:12].copy()
            elif controller and controller.status!='running':message=controller.message;break
            elif route is None and now>1.:message='No confident initial RGB estimate.';break
            data.xfrc_applied[:]=0.
            if push and push['starts_at_simulation_s']<=now<push['starts_at_simulation_s']+push['duration_s']:
                if push_origin is None:push_origin=data.body('bottle').xpos.copy()
                force=np.asarray(push['force_n']);application=data.body('bottle').xpos+[0.,0.,push['application_height_above_bottle_origin_m']]
                data.xfrc_applied[body,:3]=force;data.xfrc_applied[body,3:]=np.cross(application-data.xipos[body],force);push_ticks+=1
            if push_origin is not None and now<push['starts_at_simulation_s']+1.2:
                push_peak=max(push_peak,float(np.linalg.norm(data.body('bottle').xpos[:2]-push_origin[:2])*1000))
                push_settled=((data.body('bottle').xpos-push_origin)*1000).tolist()
            assert model.neq==0 and not np.any(data.qfrc_applied) and not np.any(np.delete(data.xfrc_applied,body,axis=0))
            if not push:assert not np.any(data.xfrc_applied)
            current=controller or last_controller
            data.ctrl[:]=apply_targets(model,data,target,.25,6 if current and current.side=='right' else 0)
            mujoco.mj_step(model,data)
    finally:renderer.close()
    report.update(status=status,message=message,route=route,completed_legs=completed,physical=monitor.report() if monitor else None,
        simulation_seconds=float(data.time),wall_seconds=time.perf_counter()-began,image_pipeline_seconds=image_seconds,
        stage=stage,arm=controller.side if controller else None,cumulative_non_target_displacement_m=max_other,
        disclosed_push_protocol=args.push.as_posix() if args.push else None,disclosed_push_ticks=push_ticks,
        push_translation_before_grasp_mm=push_settled,push_peak_translation_mm=push_peak,
        image_counterfactual_max_action_delta_rad=max((row.get('max_action_delta_rad',0.) for row in counterfactuals),default=0.))
    if controller:report.update(controller_status=controller.status,current_route_details=controller.route_details)
    require_space(args.output,16*1024**2)
    np.savez_compressed(args.output/'states.npz',**{key:np.asarray(values) for key,values in trace.items()})
    for name,content in (('observations.json',observations),('image-counterfactuals.json',counterfactuals),('report.json',report)):
        (args.output/name).write_text(json.dumps(content,indent=2)+'\n')
    print(json.dumps({key:report.get(key) for key in ('seed','status','message','route','simulation_seconds','wall_seconds','physical')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,default=Path('docs/robotics/experiments/rgb-servo-bottle-v2.json'))
    p.add_argument('--routing',type=Path,default=Path('docs/robotics/experiments/rgb-servo-routing-v2.json'))
    p.add_argument('--model',type=Path,default=Path('models/bottle_servo_v1'))
    p.add_argument('--seed',type=int,required=True);p.add_argument('--split',choices=['training','development','evaluation'],required=True)
    p.add_argument('--mode',choices=['live','frozen'],default='live');p.add_argument('--device',default='CPU')
    p.add_argument('--push',type=Path);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
