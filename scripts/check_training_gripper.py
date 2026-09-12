"""Check a fixed, observation-independent gripper cap against physical pilot replay."""
import argparse,gzip,json,sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco,numpy as np
from scripts.prepare_bottle_data import OUT,ROOT,STATE,replay,save
from simulation_lab.dinner_autonomy import DinnerTask

def check(args):
    name,cap=args;folder=OUT/name
    manifest=json.loads((folder/'manifest.json').read_text())
    model=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'));data=mujoco.MjData(model)
    initial=np.load(folder/'initial-integration-state.npy',allow_pickle=False)
    mujoco.mj_setState(model,data,initial,STATE);mujoco.mj_forward(model,data)
    task=DinnerTask(model,data,manifest['layout'])
    task._select_item(manifest['arm'],next(x for x in manifest['layout']['objects'] if x['id']=='bottle'))
    task.grip_torque=cap
    with gzip.open(folder/'actions.jsonl.gz','rt') as f:actions=[json.loads(x) for x in f]
    return dict(episode=name,cap_nm=cap,**replay(model,initial,actions,task,20))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cap',type=float,default=.35);p.add_argument('--all',action='store_true');a=p.parse_args()
    names=[x['id'] for x in json.loads((ROOT/'docs/robotics/bottle-dataset-validation.json').read_text())['episodes']] if a.all else ['upright-01','sideways-01','sideways-03']
    with ProcessPoolExecutor(max_workers=3) as pool:results=list(pool.map(check,[(n,a.cap) for n in names]))
    save(ROOT/f'.run/training-gripper-{a.cap}.json',results)
    print(results)
