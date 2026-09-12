"""Execute a typed or saved ASR command with explicitly programmed physical skills.

Record authoritative poses for later video rendering, not policy training labels.
Never starts the microphone or makes a speech API request.
"""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco,numpy as np
from simulation_lab.command_task import CommandSequence
from simulation_lab.language import parse_command
from simulation_lab.scene import HOME,build_scene
from simulation_lab.storage import require_space
from scripts.evaluate_dinner_scene import load


def run(a):
    out=Path(a.output);require_space(out,128*1024**2)
    if out.exists():raise FileExistsError('Preserve previous evaluation output; choose a new folder.')
    text=a.text
    source='typed command'
    if a.transcript_report:
        speech=json.loads(Path(a.transcript_report).read_text())
        text=speech['transcript'];source=speech['provider']+'; '+speech['audio_source']
    plan=parse_command(text)
    if plan.get('control'):raise ValueError('This standalone evaluator requires a movement command.')
    m,d,layout=load(a.seed)
    for _ in range(200):mujoco.mj_step(m,d)
    d.time=0.;target=np.array(HOME*2)
    if a.mode=='learned_bottle':
        if len(plan['steps'])!=1 or plan['steps'][0]['object_id']!='bottle' or plan['steps'][0]['destination']!={'kind':'default'}:
            raise ValueError('Learned recording currently requires the default bottle goal.')
        from simulation_lab.learned_task import LearnedBottleTask
        task=LearnedBottleTask(m,d,layout,**({'checkpoint':a.checkpoint} if a.checkpoint else {}))
    else:
        task=CommandSequence(m,d,layout);task.start_plan(plan)
    poses=[];velocities=[];times=[];stages=[];start=time.perf_counter()
    ticks=0
    while task.active and d.time<a.seconds:
        if ticks%10==0:
            if ticks%2000==0:require_space(out,32*1024**2)
            poses.append(d.qpos.copy());velocities.append(d.qvel.copy());times.append(float(d.time));stages.append(task.stage)
        before=d.qpos.copy();velocity=d.qvel.copy();task.update(target)
        assert np.array_equal(before,d.qpos) and np.array_equal(velocity,d.qvel),'Controller wrote physical state'
        assert m.neq==0 and not np.any(d.xfrc_applied) and not np.any(d.qfrc_applied)
        if not task.active:break
        d.ctrl[:]=task.apply_gripper_limit(target);mujoco.mj_step(m,d);ticks+=1
    if task.active:task.cancel(target)
    poses.append(d.qpos.copy());velocities.append(d.qvel.copy());times.append(float(d.time));stages.append(task.stage)
    report={'seed':a.seed,'input_source':source,'instruction':text,'plan':plan,'status':task.status,
            'simulated_seconds':float(d.time),'wall_seconds':time.perf_counter()-start,
            'controller':'OpenVINO learned bottle primitive; camera/motor inputs with independent simulator-state success monitor' if a.mode=='learned_bottle' else 'programmed physical skills; exact simulator state, not learned VLA',
            'physical_state_writes_during_control':0,'equality_constraints':int(m.neq),'task':task.snapshot(),
            'recording':'20 Hz authoritative physical state for video reconstruction; not a separate rollout'}
    require_space(out,128*1024**2);out.mkdir(parents=True)
    xml,_=build_scene(seed=a.seed,scenario='dinner',dinner_preset='task')
    (out/'scene.xml').write_text(xml)
    np.savez_compressed(out/'states.npz',qpos=poses,qvel=velocities,time=times,stage=stages)
    (out/'result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'status':task.status,'simulation_seconds':float(d.time),'message':task.message,'output':str(out)}))
    return 0 if task.status=='succeeded' else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--text');group.add_argument('--transcript-report');p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--seconds',type=float,default=300)
    p.add_argument('--mode',choices=['programmed','learned_bottle'],default='programmed')
    p.add_argument('--checkpoint',help='Optional explicit learned checkpoint; no programmed fallback')
    raise SystemExit(run(p.parse_args()))
