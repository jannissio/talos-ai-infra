"""Render all ten declared physical traces on one labeled simulation timeline."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import av
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from simulation_lab.storage import GIB, require_space


def render(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8-sig'))
    if len(manifest['trials']) != 10 or len({r['seed'] for r in manifest['trials']}) != 10:
        raise ValueError('Include every one of the ten distinct declared trials.')
    if args.speed not in (1, 2, 4):
        raise ValueError('Use a labeled 1x, 2x or 4x playback speed.')
    require_space(output, GIB)
    cases, container = [], None
    started = time.perf_counter()
    try:
        for entry in manifest['trials']:
            folder = Path(entry['recording'])
            report = json.loads((folder/'report.json').read_text())
            if report['seed'] != entry['seed'] or report['active']:
                raise ValueError('Every trace needs its matching terminal report.')
            model = mujoco.MjModel.from_xml_path(str(folder/'scene.xml'))
            model.vis.quality.offsamples = 0
            renderer = mujoco.Renderer(model, width=256, height=192)
            with np.load(folder/'states.npz', allow_pickle=False) as data:
                states = {name: data[name].copy() for name in data.files}
            cases.append((model, mujoco.MjData(model), renderer, states, report))
        output.parent.mkdir(parents=True, exist_ok=True)
        container = av.open(str(output), 'w', options={'movflags':'faststart'})
        stream = container.add_stream('libx264', rate=10)
        stream.width, stream.height, stream.pix_fmt = 1280, 720, 'yuv420p'
        stream.options = {'crf':'19','preset':'fast'}
        small, large = ImageFont.load_default(size=17), ImageFont.load_default(size=25)
        option = mujoco.MjvOption(); option.geomgroup[3:] = 0
        successes = sum(case[4]['status'] == 'succeeded' for case in cases)
        duration = max(float(case[3]['time'][-1]) for case in cases)
        frames = math.ceil((duration/args.speed+2)*10)
        for frame_index in range(frames):
            seconds = frame_index/10*args.speed
            if frame_index % 50 == 0:
                require_space(output, 128*1024**2)
                print(json.dumps({'frame':frame_index,'simulation_seconds':seconds}), flush=True)
            canvas = Image.new('RGB', (1280,720), '#0d1c2b')
            draw = ImageDraw.Draw(canvas)
            draw.text((25,20), manifest['title'], font=large, fill='white')
            draw.text((25,65), f'All ten recorded trials   {seconds:.1f} simulated seconds   {args.speed}x playback', font=small, fill='#66e4d5')
            for i, (model, data, renderer, states, report) in enumerate(cases):
                n = int(np.clip(np.searchsorted(states['time'], seconds, side='right')-1, 0, len(states['time'])-1))
                data.qpos[:] = states['qpos'][n]; data.qvel[:] = states['qvel'][n]
                data.time = float(states['time'][n]); mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera='overhead', scene_option=option)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                x, y = i%5*256, 115+i//5*250
                canvas.paste(Image.fromarray(renderer.render().copy()), (x,y))
                finished = seconds >= float(states['time'][-1])
                status = report['status'].upper() if finished else str(states['stage'][n]).replace('reverse_', '').replace('relay_', '')
                draw.text((x+6,y+198), f'{report["seed"]}  {status[:12]}', font=small, fill='#ffae8e' if finished and report['status']!='succeeded' else '#66e4d5')
                if finished:
                    completed = len(report.get('completed_steps', [])); total = len(report.get('steps', []))
                    draw.text((x+6,y+220), f'{completed}/{total} physical steps completed', font=small, fill='white')
            draw.text((25,635), f'Final result: {successes}/10. '+manifest['scope'], font=large, fill='white')
            draw.text((25,677), 'Actual recorded states. Every trial retained. Finished scenes hold their recorded final state.', font=small, fill='#b8cbd5')
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.asarray(canvas), format='rgb24')):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        for _, _, renderer, _, _ in cases:
            renderer.close()
        if container:
            container.close()
    result = {'manifest':manifest,'resolution':[1280,720],'fps':10,'playback_speed':args.speed,
              'duration_seconds':frames/10,'bytes':output.stat().st_size,'successes':successes,'trials':10,
              'wall_seconds':time.perf_counter()-started,'method':'Offline rendering of every recorded physical trace. No omitted trial or invented object motion.'}
    require_space(output, 1024**2)
    output.with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='manifest'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--speed', type=int, default=1)
    render(parser.parse_args())
