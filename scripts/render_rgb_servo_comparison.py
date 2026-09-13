"""Render a labeled paired comparison from immutable physical state recordings."""
import argparse
import hashlib
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
from simulation_lab.rgb_servo_cameras import camera_argument
from simulation_lab.storage import require_space


def run(args):
    if args.output.exists() or args.output.with_suffix('.json').exists():
        raise FileExistsError('Use a new output filename; preserve earlier media.')
    require_space(args.output, 128*1024**2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for case in ('push-live', 'push-frozen'):
        folder = args.evidence/f'{args.seed}-{case}'
        report = json.loads((folder/'report.json').read_text())
        with np.load(folder/'states.npz', allow_pickle=False) as data:
            states = {key: data[key] for key in data.files}
        model = mujoco.MjModel.from_xml_path(str(folder/'scene.xml'))
        model.vis.quality.offsamples = 0
        records.append({'case': case, 'report': report, 'states': states, 'model': model,
                        'data': mujoco.MjData(model), 'renderer': mujoco.Renderer(model, width=640, height=440),
                        'states_sha256': hashlib.sha256((folder/'states.npz').read_bytes()).hexdigest()})
    font_path = Path('C:/Windows/Fonts/arial.ttf')
    font = lambda size: ImageFont.truetype(str(font_path), size) if font_path.exists() else ImageFont.load_default(size=size)
    large, normal, small = font(36), font(26), font(20)
    option = mujoco.MjvOption(); option.geomgroup[3:] = 0
    container = av.open(str(args.output), 'w', options={'movflags': 'faststart'})
    stream = container.add_stream('libx264', rate=20)
    stream.width = 1280; stream.height = 720; stream.pix_fmt = 'yuv420p'
    stream.options = {'crf': '19', 'preset': 'fast'}
    frames = 0; began = time.perf_counter()

    def emit(canvas):
        nonlocal frames
        if frames % 100 == 0:
            require_space(args.output, 64*1024**2)
        for packet in stream.encode(av.VideoFrame.from_ndarray(np.asarray(canvas), format='rgb24')):
            container.mux(packet)
        frames += 1

    def card(title, lines, seconds):
        canvas = Image.new('RGB', (1280,720), '#0d1c2b'); draw = ImageDraw.Draw(canvas)
        draw.text((50,65), title, font=large, fill='#66e4d5')
        for index,line in enumerate(lines):
            draw.text((50,185+index*70), line, font=normal, fill='white')
        for _ in range(seconds*20):
            emit(canvas)

    try:
        card('Talos / experimental RGB correction', [
            'One paired physical trial, selected after the full evaluation.',
            'Seed 2026095309 passed both undisturbed controls.',
            'Same disclosed push; live images versus the initial image frozen.',
            'Experimental only. The six-skill dinner baseline is unchanged.'], 5)
        duration = max(record['report']['simulation_seconds'] for record in records)
        for frame in range(math.ceil(duration*20)+41):
            seconds = min(frame/20, math.ceil(duration*20)/20)
            canvas = Image.new('RGB', (1280,720), '#0d1c2b'); draw = ImageDraw.Draw(canvas)
            draw.text((25,18), 'Live RGB correction  /  initial-image control', font=large, fill='white')
            draw.text((25,65), f'Seed {args.seed}  |  Recorded physics at 1x speed  |  {seconds:.2f} simulated seconds', font=small, fill='#b8cbd5')
            for n, record in enumerate(records):
                states = record['states']; index = int(np.clip(np.searchsorted(states['time'], seconds, side='right')-1,0,len(states['time'])-1))
                data = record['data']; data.qpos[:] = states['qpos'][index]; data.qvel[:] = states['qvel'][index]
                data.time = float(states['time'][index]); mujoco.mj_forward(record['model'], data)
                renderer = record['renderer']; renderer.update_scene(data, camera=camera_argument('servo_left'), scene_option=option)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                canvas.paste(Image.fromarray(renderer.render().copy()), (n*640,135))
                draw.text((n*640+20,102), 'LIVE IMAGES' if n == 0 else 'INITIAL IMAGE FROZEN', font=normal, fill='#66e4d5' if n == 0 else '#ffd282')
                finished = seconds >= record['report']['simulation_seconds']
                report = record['report']; passed = report['physical']['passed']
                label = f'{"PASS" if passed else "FAIL"} / {report["physical"]["metrics"]["placement_error_mm"]:.2f} mm placement error' if finished else 'Stage: '+str(states['stage'][index])
                draw.text((n*640+20,591), label, font=normal, fill='#66e4d5' if finished and passed else 'white')
            push_label = 'Disclosed push: +0.75 N for 0.1 s at t=1.5 s; measured translation 8.77 mm.'
            draw.text((25,645), push_label, font=small, fill='#ffd282')
            draw.text((25,678), 'Full evaluation: live 2/12 nominal and 3/12 pushed. Frozen 1/12 nominal and 0/12 pushed.', font=small, fill='white')
            emit(canvas)
            if frame % 200 == 0:
                print(json.dumps({'frame': frames, 'simulation_seconds': seconds}), flush=True)
        card('The complete result limits the claim', [
            '48 trials: all 12 seeds under all four conditions.',
            'Live: 2/12 undisturbed, 3/12 pushed. Frozen: 1/12 and 0/12.',
            'Seven seeds were refused before movement; no case was excluded.',
            'Only this seed passed both undisturbed controls.',
            'Useful correction here; broad bottle coverage remains unfinished.'], 7)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        for record in records:
            record['renderer'].close()
        container.close()
    result = {'schema': 'talos.rgb-servo-comparison-video.v1', 'seed': args.seed,
              'resolution': [1280,720], 'fps': 20, 'frames': frames, 'duration_seconds': frames/20,
              'playback_speed': 1, 'final_result_hold_seconds': 2,
              'bytes': args.output.stat().st_size, 'wall_seconds': time.perf_counter()-began,
              'recordings': [{'case': r['case'], 'states_sha256': r['states_sha256']} for r in records],
              'selection': 'Illustration selected after evaluation: the only seed where both unperturbed controls pass. All 48 outcomes are packaged.',
              'method': 'Offline rendering of authoritative saved states. No new physics, interpolated object poses or invented actions.',
              'audio': 'No audio; all explanations are captioned.', 'status': 'Experimental, not the selected dinner baseline.'}
    require_space(args.output, 1024**2)
    args.output.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, default=Path('docs/robotics/evidence/rgb-servo-v1'))
    parser.add_argument('--seed', type=int, choices=[2026095309], default=2026095309)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
