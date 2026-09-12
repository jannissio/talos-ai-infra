"""Compare saved initial RGB, live RGB and action fit on one training episode."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco,numpy as np,torch
from PIL import Image
from simulation_lab.learned_task import LearnedBottleTask
from simulation_lab.primitive_policy import primitive_image_vector
from simulation_lab.retrieval_policy import KEYS

def run(a):
    if a.cuda_render:torch.cuda.init()
    f=Path(a.episode);m=mujoco.MjModel.from_xml_path(str(f/'scene.xml'));d=mujoco.MjData(m)
    mujoco.mj_setState(m,d,np.load(f/'initial-integration-state.npy'),mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
    task=LearnedBottleTask(m,d,json.loads((f/'manifest.json').read_text())['layout'],checkpoint=a.checkpoint)
    try:
        obs=task._observation();live=[(obs[k][0].permute(1,2,0).numpy()*255).round().astype('uint8') for k in KEYS]
        saved=[np.array(Image.open(f/(k.removeprefix('observation.images.')+'.png'))) for k in KEYS]
        policy=task.policy;v=policy.visual;meta=policy.meta
        with np.load(f/'trajectory.npz') as z:truth=z['actions20'];seconds=torch.tensor(z['action_indices']/200,dtype=torch.float32)
        results=[]
        for label,images in [('saved',saved),('live',live)]:
            pixels=primitive_image_vector(images,meta.get('visual_preprocess','raw'))
            features=((pixels-v['mean'])@v['components'].T/v['scale']-meta['visual_mean'])/meta['visual_std']
            with torch.inference_mode():pred=policy.net(torch.tensor(features,dtype=torch.float32)[None].expand(len(seconds),-1),seconds).numpy()*meta['action_std']+meta['action_mean']
            results.append({'input':label,'action_rmse_by_joint':np.sqrt(np.mean((pred-truth)**2,axis=0)).tolist(),'max_action_error_by_joint':np.max(abs(pred-truth),axis=0).tolist()})
        from OpenGL.GL import glGetString,GL_RENDERER
        print(json.dumps({'renderer':glGetString(GL_RENDERER).decode(),'pixel_mae_per_camera':[float(np.mean(abs(x.astype(float)-y))) for x,y in zip(live,saved)],'fit':results}))
    finally:task.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--episode',required=True);p.add_argument('--cuda-render',action='store_true');run(p.parse_args())
