"""Show every recorded seed simultaneously, with a shared simulation timeline."""
import argparse,json,math,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import av,mujoco,numpy as np
from PIL import Image,ImageDraw,ImageFont
from simulation_lab.storage import require_space,GIB

def run(a):
    folder=Path(a.evaluation);out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    require_space(out,GIB)
    report=json.loads((folder/'summary.json').read_text());seeds=report['protocol']['seeds']
    if len(seeds)!=10 or not report['complete']:raise ValueError('A complete ten-seed report is required.')
    cases=[];container=None;began=time.perf_counter()
    try:
        for seed in seeds:
            f=folder/str(seed);m=mujoco.MjModel.from_xml_path(str(f/'scene.xml'));d=mujoco.MjData(m)
            with np.load(f/'states.npz',allow_pickle=False) as z:states={k:z[k].copy() for k in z.files}
            result=json.loads((f/'result.json').read_text());m.vis.quality.offsamples=0
            renderer=mujoco.Renderer(m,width=256,height=192)
            cases.append((m,d,renderer,states,result))
        opt=mujoco.MjvOption();opt.geomgroup[3:]=0;small=ImageFont.load_default(size=17);font=ImageFont.load_default(size=25)
        out.parent.mkdir(parents=True,exist_ok=True);container=av.open(str(out),'w',options={'movflags':'+faststart'})
        stream=container.add_stream('libx264',rate=10);stream.width=1280;stream.height=720;stream.pix_fmt='yuv420p';stream.options={'crf':'19','preset':'fast'}
        duration=max(float(c[3]['time'][-1]) for c in cases)+2;frames=int(math.ceil(duration*10))
        for n in range(frames):
            t=n/10
            if n%50==0:require_space(out,128*1024**2);print(json.dumps({'frame':n,'simulation_time_s':t}),flush=True)
            canvas=Image.new('RGB',(1280,720),(13,28,43));draw=ImageDraw.Draw(canvas)
            draw.text((25,20),f'Talos upright bottle | all ten recorded trials | {t:.1f} s | 1x timeline',font=font,fill='white')
            draw.text((25,65),'OpenVINO neural trajectory + initial RGB localization + motor-feedback guard',font=small,fill='#66e4d5')
            for i,(_,d,renderer,states,result) in enumerate(cases):
                index=max(0,min(len(states['time'])-1,int(np.searchsorted(states['time'],t,side='right'))-1))
                d.qpos[:]=states['qpos'][index];d.qvel[:]=states['qvel'][index];d.time=float(states['time'][index]);mujoco.mj_forward(cases[i][0],d)
                renderer.update_scene(d,camera='overhead',scene_option=opt);renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
                x=(i%5)*256;y=115+(i//5)*250;canvas.paste(Image.fromarray(renderer.render().copy()),(x,y))
                finished=t>=float(states['time'][-1]);status=result['status'].upper() if finished else 'RUNNING'
                draw.text((x+8,y+198),f'{result["seed"]}  {status}',font=small,fill='#66e4d5' if finished and status=='SUCCEEDED' else 'white')
                if finished:draw.text((x+8,y+220),f'{result["task"]["metrics"]["placement_error_mm"]:.2f} mm final error',font=small,fill='white')
            draw.text((25,635),f'Final result: {report["successes"]}/10 | new task-preset seeds, small upright bottle jitter',font=font,fill='white')
            draw.text((25,677),'Overhead views of authoritative recorded physics. Finished trials remain at their actual final state.',font=small,fill='#b8cbd5')
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.asarray(canvas),format='rgb24')):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
    finally:
        for _,_,renderer,_,_ in cases:renderer.close()
        if container:container.close()
    out.with_suffix('.json').write_text(json.dumps({'seeds':seeds,'successes':report['successes'],'duration_s':frames/10,'fps':10,'resolution':[1280,720],'bytes':out.stat().st_size,'wall_seconds':time.perf_counter()-began,'method':'All ten recorded trajectories, rendered simultaneously from overhead; common 1x simulation timeline. No physics rerun or omitted trial.'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evaluation',required=True);p.add_argument('--output',required=True);run(p.parse_args())
