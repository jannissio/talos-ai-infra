"""Camera/proprioception-only ACT rollout with a separate physical outcome observer."""
import argparse,json,os,sys,time
from pathlib import Path
if sys.platform!='win32':os.environ.setdefault('MUJOCO_GL','egl')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco,numpy as np,torch
from simulation_lab.act_learning import CAMERAS,load_policy,normalize,denormalize
from simulation_lab.dinner_autonomy import DinnerTask
from scripts.prepare_bottle_data import STATE
from simulation_lab.policy_control import apply_targets
from simulation_lab.retrieval_policy import ObservationRejected

def observation(model,data,renderer,option,blank=False):
    # This dictionary is the entire policy interface. No layout, pose, stage or contacts.
    result={'observation.state':torch.tensor(np.r_[data.qpos[:12],data.qvel[:12]],dtype=torch.float32)[None]}
    for camera in CAMERAS:
        renderer.update_scene(data,camera=camera,scene_option=option)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
        rgb=renderer.render().copy()
        if blank:rgb[:]=0
        result['observation.images.'+camera]=torch.from_numpy(rgb).permute(2,0,1)[None].float()/255
    return result

def evaluate(a):
    state_recording=getattr(a,'record_states',None)
    recorded={'qpos':[],'qvel':[],'time':[],'stage':[]}
    if state_recording:
        from simulation_lab.storage import require_space
        require_space(state_recording,128*1024**2)
        if Path(state_recording).exists():raise FileExistsError('Choose a fresh state-recording folder.')
    torch.set_num_threads(4)
    folder=Path(a.episode).resolve();manifest=json.loads((folder/'manifest.json').read_text())
    model=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'));data=mujoco.MjData(model)
    mujoco.mj_setState(model,data,np.load(folder/'initial-integration-state.npy',allow_pickle=False),STATE)
    mujoco.mj_forward(model,data);assert model.neq==0
    start=float(data.time);parked=data.qpos[6:12].copy();initial_robot=data.qpos[:12].copy()
    observer=DinnerTask(model,data,manifest['layout'])
    observer._select_item('left',next(x for x in manifest['layout']['objects'] if x['id']=='bottle'))
    destination=observer.destination_position.copy()
    if a.seconds<=0 or not 0<a.gripper_cap<=.65:raise ValueError('Require positive timeout and gripper cap in (0, 0.65].')
    training_meta=Path(a.checkpoint)/'run.json'
    if training_meta.exists():
        recorded_caps=json.loads(training_meta.read_text()).get('dataset',{}).get('gripper_caps_nm',[])
        if len(recorded_caps)==1 and abs(recorded_caps[0]-a.gripper_cap)>1e-9:
            raise ValueError('Playback gripper cap differs from the checkpoint training contract.')
    policy,stats=load_policy(a.checkpoint,a.device)
    print('Policy loaded; initializing renderer.',flush=True)
    model.vis.quality.offsamples=0
    renderer=mujoco.Renderer(model,height=240,width=320)
    from OpenGL.GL import glGetString,GL_VENDOR,GL_RENDERER,GL_VERSION
    graphics={name:(glGetString(key) or b'unknown').decode(errors='replace')
              for name,key in [('vendor',GL_VENDOR),('renderer',GL_RENDERER),('version',GL_VERSION)]}
    print('Renderer initialized; starting physical rollout.',flush=True)
    option=mujoco.MjvOption();option.geomgroup[3:]=0
    latencies=[];held=best_hold=stable=0.;clipped=0;success=False;reason='timeout'
    max_other=0.;max_parked=0.;bad_gap=longest_gap=0.
    error=float(np.linalg.norm(data.body('bottle').xpos[:2]-destination[:2]));rejection=None
    began=time.perf_counter();stall_target=None;diagnostic_trace=[]
    video=stream=None
    if a.video:
        import av
        Path(a.video).parent.mkdir(parents=True,exist_ok=True)
        video=av.open(a.video,'w');stream=video.add_stream('libx264',rate=5)
        stream.width=320;stream.height=240;stream.pix_fmt='yuv420p';stream.options={'crf':'23'}
    try:
        for _cycle in range(int(np.ceil(a.seconds/.2))):
            if _cycle%25==0:print(json.dumps({'simulated_s':float(data.time-start),'cycle':_cycle}),flush=True)
            raw=observation(model,data,renderer,option,a.blank_images)
            if video:
                rgb=(raw['observation.images.overhead'][0].permute(1,2,0).numpy()*255).round().astype(np.uint8)
                for packet in stream.encode(av.VideoFrame.from_ndarray(rgb,format='rgb24')):video.mux(packet)
            batch=normalize(raw,stats,a.device)
            t=time.perf_counter()
            try:
                with torch.inference_mode():chunk=denormalize(policy.predict_action_chunk(batch),stats,raw['observation.state'])[0].cpu().numpy()
            except ObservationRejected as exc:
                rejection=str(exc);reason='policy_observation_rejected';break
            latencies.append(time.perf_counter()-t)
            if chunk.shape!=(20,12) or not np.isfinite(chunk).all():reason='invalid_policy_action';break
            bounded=np.clip(chunk,model.actuator_ctrlrange[:12,0],model.actuator_ctrlrange[:12,1])
            clipped+=int(np.count_nonzero(chunk!=bounded));chunk=bounded
            if getattr(a,'trace',None):
                diagnostic_trace.append({'elapsed_s':float(data.time-start),'joints':data.qpos[:12].tolist(),
                    'velocities':data.qvel[:12].tolist(),'targets':chunk[:5].tolist(),'observer':dict(observer.metrics),
                    'policy_features':policy.last_features.tolist() if hasattr(policy,'last_features') else None})
            # Four 50 ms intervals; the fifth prediction is the last interpolation endpoint.
            for k in range(40):
                if state_recording and k%10==0:
                    if _cycle%25==0:require_space(state_recording,32*1024**2)
                    recorded['qpos'].append(data.qpos.copy());recorded['qvel'].append(data.qvel.copy())
                    recorded['time'].append(float(data.time-start));recorded['stage'].append('learned bottle policy')
                u=k/10;i=int(u);target=chunk[i]*(1-(u-i))+chunk[i+1]*(u-i)
                elapsed=float(data.time-start)
                if a.stall_at_seconds<=elapsed<a.stall_at_seconds+a.stall_start_seconds:
                    if stall_target is None:stall_target=data.qpos[:12].copy()
                    target=stall_target
                data.ctrl[:]=apply_targets(model,data,target,a.gripper_cap)
                previous_time=float(data.time)
                mujoco.mj_step(model,data)
                if data.time<=previous_time:
                    reason='physics_reset_or_stalled';break
                assert not np.any(data.xfrc_applied) and not np.any(data.qfrc_applied)
                forces,lift,up,both=observer._observe()
                supported=both and lift>.05 and forces['external']<.02
                held=held+.005 if supported else 0.;best_hold=max(best_hold,held)
                bad_gap=bad_gap+.005 if lift>.012 and not both and forces['base']<.02 else 0.
                longest_gap=max(longest_gap,bad_gap)
                max_other=max(max_other,max(float(np.linalg.norm(data.body(n).xpos-p)) for n,p in observer.others.items()))
                max_parked=max(max_parked,float(np.rad2deg(np.max(np.abs(data.qpos[6:12]-parked)))))
                error=float(np.linalg.norm(data.body('bottle').xpos[:2]-destination[:2]))
                dof=model.joint('bottle_free').dofadr[0];speed=float(np.linalg.norm(data.qvel[dof:dof+3]))
                released=max(forces['fixed'],forces['moving'])<.02
                placed=error<.012 and up>np.cos(np.deg2rad(5)) and speed<.003 and forces['base']>.02 and released
                stable=stable+.005 if placed else 0.
                if best_hold>=1.49 and stable>=1. and max_other<.004 and max_parked<=1 and longest_gap<=.18:
                    success=True;reason='verified_lift_hold_release_and_place';break
                if not np.isfinite(data.qpos).all():reason='nonfinite_physics';break
            if success or reason!='timeout':break
    finally:
        renderer.close()
        if video:
            for packet in stream.encode():video.mux(packet)
            video.close()
    result={'episode':folder.name,'checkpoint':str(a.checkpoint),'success':success,'reason':reason,
            'simulated_s':float(data.time-start),'wall_s':time.perf_counter()-began,
            'best_supported_hold_s':best_hold,'stable_release_s':stable,'placement_error_m':error,
            'other_object_max_displacement_m':max_other,'parked_arm_max_motion_deg':max_parked,
            'longest_unsupported_contact_gap_s':longest_gap,'clipped_action_values':clipped,
            'inference_median_ms':float(np.median(latencies)*1000) if latencies else None,
            'inference_p95_ms':float(np.percentile(latencies,95)*1000) if latencies else None,'rejection_detail':rejection,
            'gripper_cap_nm':a.gripper_cap,'blank_images':a.blank_images,'teacher_updates':0,'state_resets':1,
            'fault_injection':{'actuator_hold_seconds':a.stall_start_seconds,'hold_begins_at_seconds':a.stall_at_seconds},
            'max_lift_cm':observer.metrics.get('max_lift_cm',0),'video':a.video,
            'policy_details':policy.details() if hasattr(policy,'details') else {'kind':'ACT'},
            'runtime':{'policy_device':str(next(policy.parameters()).device),
                       'torch_version':torch.__version__,'torch_threads':torch.get_num_threads(),
                       'mujoco_version':mujoco.__version__,'graphics':graphics},
            'timing':'Offline synchronous evaluation; physics waits during inference. Not a real-time claim.',
            'policy_inputs':['three RGB cameras','12 joint positions','12 joint velocities']}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    if state_recording:
        import xml.etree.ElementTree as ET
        require_space(state_recording,128*1024**2)
        output=Path(state_recording);output.mkdir(parents=True)
        scene=ET.fromstring((folder/'scene.xml').read_text());compiler=scene.find('compiler')
        # Preserve asset resolution when moving an XML into a new private folder.
        compiler.set('meshdir',str((folder/compiler.get('meshdir')).resolve()))
        (output/'scene.xml').write_text(ET.tostring(scene,encoding='unicode'))
        np.savez_compressed(output/'states.npz',**recorded)
        (output/'result.json').write_text(json.dumps(result,indent=2))
    if getattr(a,'save_final_state',None):
        saved=np.empty(mujoco.mj_stateSize(model,STATE));mujoco.mj_getState(model,data,saved,STATE)
        np.save(a.save_final_state,saved,allow_pickle=False)
    if getattr(a,'trace',None):
        Path(a.trace).write_text(json.dumps({'observer_fields_are_not_policy_inputs':True,'rows':diagnostic_trace}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--episode',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');p.add_argument('--seconds',type=float,default=60)
    p.add_argument('--gripper-cap',type=float,default=.25);p.add_argument('--blank-images',action='store_true');p.add_argument('--video')
    p.add_argument('--stall-start-seconds',type=float,default=0)
    p.add_argument('--save-final-state',help='Save final integration state for an explicitly separate recovery-data experiment.')
    p.add_argument('--trace',help='Optional offline command/state trace; observer metrics never enter the policy.')
    p.add_argument('--record-states',help='Fresh folder for compact actual-state capture and later HD video rendering.')
    p.add_argument('--stall-at-seconds',type=float,default=0);a=p.parse_args()
    if min(a.stall_start_seconds,a.stall_at_seconds)<0:p.error('Actuator stall timing must be nonnegative.')
    evaluate(a)
