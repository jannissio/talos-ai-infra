"""No-step comparison of renderer replacement order on one exposed saved state."""
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from PIL import Image
import torch
from simulation_lab.learned_dinner import LearnedDinnerSequence
from simulation_lab.mug_visual_control import VisualMugSequence
from simulation_lab.retrieval_policy import KEYS,ObservationRejected
from simulation_lab.scene import build_scene,HOME
from simulation_lab.storage import require_space


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root=ROOT/'.run/mug-renderer-lifecycle-v1'
    if root.exists():raise FileExistsError('Preserve earlier diagnostic evidence.')
    preflight=require_space(root,4*1024**2);root.mkdir(parents=True)
    source=ROOT/'.run/mug-visual-correction-v1/development/2026100301-upright-baseline'
    with np.load(source/'states.npz',allow_pickle=False) as saved:
        q,v,time=saved['qpos'][0],saved['qvel'][0],float(saved['time'][0])
    assert time==0
    protocol=json.loads((ROOT/'docs/robotics/experiments/mug-visual-correction-v1.json').read_text())
    suite_file=ROOT/protocol['baseline_suite'];suite=json.loads(suite_file.read_text())
    checkpoints={k:suite_file.parent/value for k,value in suite.items()}
    steps=['bottle','plate','mug','drawer','fork','spoon'];rows=[];arrays=[]
    torch.set_num_threads(2)
    with patch.object(mujoco,'mj_step',side_effect=RuntimeError('No physics stepping in a renderer diagnostic.')):
        for mode in ('single_renderer','create_then_close','close_then_create'):
            require_space(root,1024**2)
            model=mujoco.MjModel.from_xml_path(str((source/'scene.xml').resolve()))
            data=mujoco.MjData(model);data.qpos[:]=q;data.qvel[:]=v;data.ctrl[:]=HOME*2;data.time=time
            mujoco.mj_forward(model,data)
            _,layout=build_scene(2026100301,3,None,scenario='dinner',dinner_preset='task',drawer_open=False)
            layout['bottle_start']='upright'
            old=LearnedDinnerSequence(model,data,layout,checkpoints,steps)
            if mode=='single_renderer':current=old
            elif mode=='create_then_close':
                current=VisualMugSequence(model,data,layout,checkpoints,steps,protocol,'live');old.close()
            else:
                old.close();current=VisualMugSequence(model,data,layout,checkpoints,steps,protocol,'live')
            batch=current.child._observation()
            images=np.stack([(batch[k][0].numpy().transpose(1,2,0)*255).round().clip(0,255).astype(np.uint8) for k in KEYS])
            row={'mode':mode,'rgb_sha256':[hashlib.sha256(image.tobytes()).hexdigest() for image in images],
                'rgb_minimum':int(images.min()),'rgb_maximum':int(images.max()),'rgb_mean':float(images.mean())}
            try:
                with torch.inference_mode():action=current.child.policy.predict_action_chunk(batch).numpy()
                row.update(observation_accepted=True,action_sha256=hashlib.sha256(action.tobytes()).hexdigest())
            except ObservationRejected as exc:row.update(observation_accepted=False,reason=str(exc))
            assert np.array_equal(data.qpos,q) and np.array_equal(data.qvel,v) and data.time==time
            with (root/(mode+'.png')).open('xb') as stream:Image.fromarray(np.concatenate(list(images),axis=1)).save(stream,format='PNG')
            current.close();rows.append(row);arrays.append(images)
    result={'schema':'talos.mug-renderer-lifecycle.v1','source_sha256':sha(__file__),
        'input_sha256':{(source/name).relative_to(ROOT).as_posix():sha(source/name) for name in ('scene.xml','states.npz')},
        'preflight':preflight,'new_physical_trials':0,'integration_steps':0,'rows':rows,
        'close_before_create_matches_single_rgb':bool(np.array_equal(arrays[0],arrays[2])),
        'close_after_create_matches_single_rgb':bool(np.array_equal(arrays[0],arrays[1])),
        'scope':'An already exposed saved scene at time zero, restored for rendering only. No model changes or physical controller trial.'}
    with (root/'report.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(result,flush=True)


if __name__=='__main__':main()
