"""Render identical recorded states and save frozen camera calibration."""
import sys,json,gzip,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from PIL import Image,ImageDraw
from scripts.prepare_bottle_data import ROOT,OUT,save

if __name__=='__main__':
    folder=OUT/'sideways-01'
    if not folder.exists():folder=OUT/'upright-01'
    model=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'))
    data=mujoco.MjData(model)
    with gzip.open(folder/'observations.jsonl.gz','rt') as f: rows=[json.loads(x) for x in f]
    with gzip.open(folder/'actions.jsonl.gz','rt') as f: actions=[json.loads(x) for x in f]
    cameras=['overhead','opposite']
    stages=['approach','close','hold','rotate','lower']
    sheet=Image.new('RGB',(960,380*len(stages)),'#14212e')
    draw=ImageDraw.Draw(sheet)
    renderer=mujoco.Renderer(model,height=360,width=480)
    option=mujoco.MjvOption();option.geomgroup[3:]=0
    metrics=[]
    for r,stage in enumerate(stages):
        matches=[o for o in rows if o['action_index']<len(actions) and actions[o['action_index']]['stage']==stage]
        if not matches:continue
        row=matches[len(matches)//2]
        data.qpos[:]=row['qpos'];data.qvel[:]=row['qvel'];mujoco.mj_forward(model,data)
        for c,camera in enumerate(cameras):
            renderer.update_scene(data,camera=camera,scene_option=option)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
            sheet.paste(Image.fromarray(renderer.render()),(480*c,380*r+20))
            draw.text((480*c+10,380*r+3),f'{stage} | {camera} | t={row["time_s"]:.2f}s',fill='white')
            renderer.enable_segmentation_rendering()
            seg=renderer.render()
            bottle_geoms=np.flatnonzero(model.geom_bodyid==model.body('bottle').id)
            pixels=int(np.sum(np.isin(seg[:,:,0],bottle_geoms)&(seg[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))))
            renderer.disable_segmentation_rendering()
            metrics.append({'stage':stage,'camera':camera,'visible_bottle_pixels':pixels})
    renderer.close()
    dest=ROOT/'docs/robotics/bottle-camera-comparison.jpg';sheet.save(dest,quality=90)
    calibration=[]
    for name in ['overhead','opposite','left_wrist_cam','right_wrist_cam']:
        i=model.camera(name).id;fy=480/(2*math.tan(math.radians(model.cam_fovy[i])/2))
        calibration.append({'name':name,'body_id':int(model.cam_bodyid[i]),'local_position_m':model.cam_pos[i].tolist(),
            'local_quaternion_wxyz':model.cam_quat[i].tolist(),'vertical_fov_deg':float(model.cam_fovy[i]),
            'resolution':[640,480],'intrinsics_fx_fy_cx_cy':[fy,fy,320,240],
            'coordinates':'MuJoCo camera local +X right, +Y up, viewing along -Z. Wrist extrinsics vary with joints.'})
    save(ROOT/'docs/robotics/bottle-camera-calibration.json',{'selected':['overhead','left_wrist_cam','right_wrist_cam'],
        'comparison_episode':folder.name,'comparison_resolution':[480,360],'visibility':metrics,'calibration':calibration})
    print(dest)
