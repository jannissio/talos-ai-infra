"""Render actual recorded simulation states as an HD multiview video.

No physics is rerun and no actions or intermediate object poses are invented.
Training camera inputs remain unchanged. Outputs are labeled offline renders.
"""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import av,mujoco,numpy as np
from PIL import Image,ImageDraw,ImageFont
from simulation_lab.storage import require_space,GIB


def run(a):
    folder=Path(a.recording);out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    require_space(out,GIB)
    with np.load(folder/'states.npz',allow_pickle=False) as data:states={k:data[k] for k in data.files}
    m=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'));d=mujoco.MjData(m)
    width,height=1280,720;tile_width,tile_height=640,360
    m.vis.global_.offwidth=max(m.vis.global_.offwidth,tile_width);m.vis.global_.offheight=max(m.vis.global_.offheight,tile_height)
    m.vis.quality.offsamples=0;renderer=mujoco.Renderer(m,width=tile_width,height=tile_height)
    cameras=['opposite','overhead','left_wrist_cam','right_wrist_cam']
    opt=mujoco.MjvOption();opt.geomgroup[3:]=0
    font=ImageFont.load_default(size=18)
    out.parent.mkdir(parents=True,exist_ok=True);container=av.open(str(out),'w')
    stream=container.add_stream('libx264',rate=20);stream.width=width;stream.height=height;stream.pix_fmt='yuv420p'
    stream.options={'crf':'20','preset':'fast'};count=0;started=time.perf_counter()
    try:
        # The final observation can be off the 50 ms grid; regular samples alone
        # preserve the 20 Hz timing instead of extending a partial interval.
        for i,t in enumerate(states['time']):
            if abs(t*20-round(t*20))>1e-5:continue
            if a.max_frames and count>=a.max_frames:break
            if count%100==0:
                require_space(out,128*1024**2);print(json.dumps({'frame':count,'simulated_seconds':float(t)}),flush=True)
            d.qpos[:]=states['qpos'][i];d.qvel[:]=states['qvel'][i];d.time=t;mujoco.mj_forward(m,d)
            canvas=Image.new('RGB',(width,height))
            for n,camera in enumerate(cameras):
                renderer.update_scene(d,camera=camera,scene_option=opt);renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
                tile=Image.fromarray(renderer.render().copy());draw=ImageDraw.Draw(tile)
                draw.rectangle((0,0,640,27),fill=(11,23,35));draw.text((10,3),camera.replace('_cam','').replace('_',' '),font=font,fill='white')
                canvas.paste(tile,((n%2)*tile_width,(n//2)*tile_height))
            draw=ImageDraw.Draw(canvas);draw.rectangle((0,690,1280,720),fill=(11,23,35))
            draw.text((10,695),f'Talos | {t:.2f} s | {states["stage"][i]} | {a.label} | recorded physics',font=font,fill='white')
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.asarray(canvas),format='rgb24')):container.mux(packet)
            count+=1
        for packet in stream.encode():container.mux(packet)
    finally:renderer.close();container.close()
    report={'resolution':[width,height],'fps':20,'frames':count,'cameras':cameras,'wall_seconds':time.perf_counter()-started,
            'bytes':out.stat().st_size,'method':'Offline rendering of recorded authoritative states; training image resolution unchanged',
            'controller_label':a.label}
    out.with_suffix('.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--recording',required=True);p.add_argument('--output',required=True)
    p.add_argument('--label',default='programmed physical skills');p.add_argument('--max-frames',type=int,default=0)
    run(p.parse_args())
