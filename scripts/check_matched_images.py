"""Check training-environment camera pixels against the matched initial capture."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco,numpy as np,torch
from PIL import Image
from scripts.evaluate_act import observation
from scripts.prepare_bottle_data import STATE
from simulation_lab.act_learning import CAMERAS

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',default='.run/bottle-matched-nvidia/sideways-01');a=p.parse_args()
    folder=Path(a.folder)
    model=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'));data=mujoco.MjData(model)
    mujoco.mj_setState(model,data,np.load(folder/'initial-integration-state.npy',allow_pickle=False),STATE)
    mujoco.mj_forward(model,data);model.vis.quality.offsamples=0
    torch.cuda.init()
    renderer=mujoco.Renderer(model,height=240,width=320);option=mujoco.MjvOption();option.geomgroup[3:]=0
    try:batch=observation(model,data,renderer,option)
    finally:renderer.close()
    result={}
    for camera in CAMERAS:
        actual=(batch['observation.images.'+camera][0].permute(1,2,0).numpy()*255).round().astype(np.uint8)
        expected=np.asarray(Image.open(folder/'images'/f'000000_{camera}.png'))
        delta=np.abs(actual.astype(int)-expected.astype(int))
        result[camera]={'max_channel_difference':int(delta.max()),'different_pixel_fraction':float(np.any(delta,axis=-1).mean())}
    Path('docs/robotics/act-matched-image-check.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    assert all(x['max_channel_difference']==0 for x in result.values())
