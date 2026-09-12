"""Exercise the same learned task class used by the live engine, without browser timing."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco,numpy as np
from simulation_lab.learned_task import LearnedBottleTask,DEFAULT_CHECKPOINT
from simulation_lab.scene import HOME
from scripts.evaluate_dinner_scene import load
from simulation_lab.storage import require_space


def run(a):
    require_space(a.output,32*1024**2)
    if a.episode:
        folder=Path(a.episode);layout=json.loads((folder/'manifest.json').read_text())['layout']
        m=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'));d=mujoco.MjData(m)
        mujoco.mj_setState(m,d,np.load(folder/'initial-integration-state.npy',allow_pickle=False),mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(m,d)
    else:
        m,d,layout=load(a.seed)
        for _ in range(200):mujoco.mj_step(m,d)
    task=LearnedBottleTask(m,d,layout,checkpoint=a.checkpoint);targets=np.array(HOME*2);ticks=0
    if a.blank:
        observation=task._observation
        task._observation=lambda: {k: v*0 for k,v in observation().items()}
    try:
        while task.active and ticks<12500:
            if a.cancel_after is not None and ticks>=a.cancel_after:task.cancel(targets);break
            before=d.qpos.copy();velocity=d.qvel.copy();task.update(targets)
            assert np.array_equal(before,d.qpos) and np.array_equal(velocity,d.qvel)
            assert m.neq==0 and not np.any(d.xfrc_applied) and not np.any(d.qfrc_applied)
            if task.active:d.ctrl[:]=task.apply_gripper_limit(targets);mujoco.mj_step(m,d)
            ticks+=1
        result=task.snapshot();result.update(input_case=Path(a.episode).name if a.episode else 'default-seed-'+str(a.seed),
                                            physical_state_writes=0,hidden_forces=0,equality_constraints=int(m.neq),
                                            renderer_closed=task.renderer is None)
        Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps({'status':task.status,'message':task.message,'metrics':task.metrics}))
    finally:task.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episode');p.add_argument('--seed',type=int,default=42);p.add_argument('--output',required=True)
    p.add_argument('--checkpoint',default=str(DEFAULT_CHECKPOINT))
    p.add_argument('--cancel-after',type=int);p.add_argument('--blank',action='store_true');run(p.parse_args())
