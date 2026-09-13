"""Measure camera observability through an existing physical bottle trace."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from PIL import Image
from simulation_lab.policy_cameras import camera_argument
from simulation_lab.rgb_servo_vision import observe_bottle_views
from simulation_lab.storage import require_space


def project_camera(renderer,point):
    # Calibration only; object point is scoring-only and never used by detection.
    cameras=renderer.scene.camera
    position=np.mean([c.pos for c in cameras],axis=0)
    forward=np.mean([c.forward for c in cameras],axis=0);forward/=np.linalg.norm(forward)
    up=np.mean([c.up for c in cameras],axis=0);up/=np.linalg.norm(up)
    right=np.cross(forward,up);right/=np.linalg.norm(right)
    camera=cameras[0];relative=point-position;depth=float(relative@forward)
    if depth<=0:return None
    top,bottom,near=map(float,[camera.frustum_top,camera.frustum_bottom,camera.frustum_near])
    halfwidth=(top-bottom)*320/240/2
    x=(float(relative@right)/depth*near+halfwidth)/(halfwidth*2)*320
    y=(top-float(relative@up)/depth*near)/(top-bottom)*240
    return [x,y]


def run(args):
    if args.output.exists():raise FileExistsError(args.output)
    require_space(args.output,32*1024**2);args.output.mkdir(parents=True)
    model=mujoco.MjModel.from_xml_path(str(args.recording/'scene.xml'));data=mujoco.MjData(model)
    with np.load(args.recording/'states.npz',allow_pickle=False) as source:states={k:source[k] for k in source.files}
    indices=np.flatnonzero(states['stage']=='bottle')
    if not len(indices):raise ValueError('Trace has no bottle skill.')
    selected=np.unique(indices[np.linspace(0,len(indices)-1,min(96,len(indices))).astype(int)])
    model.vis.quality.offsamples=0;renderer=mujoco.Renderer(model,width=320,height=240)
    option=mujoco.MjvOption();option.geomgroup[3:]=0
    rows=[];began=time.perf_counter()
    try:
        for j,index in enumerate(selected):
            require_space(args.output,1024**2)
            data.qpos[:]=states['qpos'][index];data.qvel[:]=states['qvel'][index]
            data.time=float(states['time'][index]);mujoco.mj_forward(model,data)
            images={};scores={}
            for camera in ['overhead','table_left','table_right']:
                renderer.update_scene(data,camera=camera_argument(camera),scene_option=option)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
                images[camera]=renderer.render().copy()
                scores[camera]=project_camera(renderer,data.body('bottle').xpos+data.body('bottle').xmat.reshape(3,3)@np.array([0,0,.07]))
                if j%12==0 or j==len(selected)-1:
                    Image.fromarray(images[camera]).save(args.output/f'frame-{j:03d}-{camera}.png')
            observation=observe_bottle_views(images)
            for camera,view in observation['views'].items():
                view['scoring_only_projected_bottle_center_px']=scores[camera]
                for candidate in view['candidates']:
                    candidate['scoring_only_center_error_px']=float(np.linalg.norm(np.array(candidate['centroid_px'])-scores[camera])) if scores[camera] is not None else None
            rows.append({'time_s':data.time,**observation})
    finally:renderer.close()
    result={'protocol':'docs/robotics/experiments/rgb-servo-bottle-v1.json','recording':args.recording.as_posix(),
            'scope':'Exposed recorded bottle motion, RGB candidate observability only. No corrective controller or generalization claim.',
            'frames':len(rows),'at_least_two_unique_views':sum(r['unique_views']>=2 for r in rows),
            'per_camera':{c:{s:sum(r['views'][c]['status']==s for r in rows) for s in ['unique','missing','ambiguous']} for c in ['overhead','table_left','table_right']},
            'rows':rows,'wall_seconds':time.perf_counter()-began}
    (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
