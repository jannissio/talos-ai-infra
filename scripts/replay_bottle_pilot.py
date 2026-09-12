"""Replay saved nominal commands through physics; never restore observation poses."""
import argparse,gzip,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from scripts.prepare_bottle_data import replay,STATE,save
from simulation_lab.dinner_autonomy import DinnerTask

def replay_saved(folder,hz=20,open_gripper=False):
    folder=Path(folder)
    manifest=json.loads((folder/'manifest.json').read_text())
    model=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'))
    initial=np.load(folder/manifest['initial_state']['file'],allow_pickle=False)
    data=mujoco.MjData(model)
    mujoco.mj_setState(model,data,initial,STATE)
    mujoco.mj_forward(model,data)
    task=DinnerTask(model,data,manifest['layout'])
    item=next(x for x in manifest['layout']['objects'] if x['id']=='bottle')
    task._select_item(manifest['arm'],item)
    task.grip_torque=manifest['gripper_torque_cap_nm']
    with gzip.open(folder/manifest['actions'],'rt') as f:actions=[json.loads(x) for x in f]
    if open_gripper:
        for action in actions:action['target'][task.offset+5]=.85
    return replay(model,initial,actions,task,hz)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path);p.add_argument('--hz',type=int,choices=[20,200],default=20)
    p.add_argument('--open-gripper',action='store_true');p.add_argument('--output',type=Path)
    a=p.parse_args();result=replay_saved(a.folder,a.hz,a.open_gripper)
    if a.output:save(a.output,result)
    print(json.dumps(result,indent=2))
