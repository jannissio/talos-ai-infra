"""Render all ten frozen composed-command trials, including every failure."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import av
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from simulation_lab.storage import require_space


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    audit = json.loads((args.evidence / 'audit.json').read_text())
    rows = [r for r in audit['outcomes'] if r['split'] == 'evaluation']
    if len(rows) != 10:
        raise ValueError('Every frozen trial is required.')
    require_space(args.output, 128 * 1024**2)
    cases = []
    container = None
    fps, speed = 10, 4
    title, small = ImageFont.load_default(size=29), ImageFont.load_default(size=17)
    try:
        for row in rows:
            folder = (args.evidence / row['report']).parent
            model = mujoco.MjModel.from_xml_path(str(folder / 'scene.xml'))
            model.vis.quality.offsamples = 0
            with np.load(folder / 'states.npz', allow_pickle=False) as arrays:
                states = {k: arrays[k] for k in arrays.files}
            cases.append((row, model, mujoco.MjData(model), mujoco.Renderer(model, width=256, height=192), states))
        duration = max(float(c[4]['time'][-1]) for c in cases)
        frames = math.ceil((duration / speed + 3) * fps)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        container = av.open(str(args.output), 'w', options={'movflags': 'faststart'})
        stream = container.add_stream('libx264', rate=fps)
        stream.width, stream.height, stream.pix_fmt = 1280, 720, 'yuv420p'
        stream.options = {'crf': '19', 'preset': 'fast'}
        option = mujoco.MjvOption()
        option.geomgroup[3:] = 0
        for frame in range(frames):
            seconds = min(frame / fps * speed, duration)
            if frame % 100 == 0:
                require_space(args.output, 64 * 1024**2)
                print(json.dumps({'frame': frame, 'frames': frames}), flush=True)
            canvas = Image.new('RGB', (1280, 720), '#0d1c2b')
            draw = ImageDraw.Draw(canvas)
            draw.text((25, 18), 'Talos | all ten composed dinner-command trials', font=title, fill='white')
            draw.text((25, 64), f'"Set the table" | finite left-reach bottle preset | 4x playback | {seconds:.1f} simulated seconds', font=small, fill='#66e4d5')
            for number, (row, model, data, renderer, states) in enumerate(cases):
                index = max(0, min(len(states['time'])-1, int(np.searchsorted(states['time'], seconds, side='right'))-1))
                data.qpos[:] = states['qpos'][index]
                data.qvel[:] = states['qvel'][index]
                data.time = float(states['time'][index])
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera='overhead', scene_option=option)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                x, y = number % 5 * 256, 107 + number // 5 * 249
                canvas.paste(Image.fromarray(renderer.render().copy()), (x, y))
                finished = seconds >= float(states['time'][-1])
                status = ('PASS' if row['passed'] else 'FAIL') if finished else 'RUNNING'
                color = '#66e4d5' if finished and row['passed'] else '#ffb18f' if finished else 'white'
                draw.text((x+7, y+198), f'{row["seed"]}  {status}', font=small, fill=color)
                stage = str(states['stage'][index])
                stage = {'reverse_bottle_right': 'Bottle relay: first arm', 'reverse_bottle_left': 'Bottle relay: second arm'}.get(stage, stage.capitalize())
                text = f'{len(row["completed_steps"])} / 7 steps completed' if finished else stage
                draw.text((x+7, y+220), text, font=small, fill='white')
            draw.text((25, 625), f'Full workflow: {audit["passed_by_split"]["evaluation"]} / 10 | both relay legs + plate, mug, drawer, fork, spoon', font=small, fill='white')
            draw.text((25, 662), 'Recorded physical states; each completed step releases and parks. Failed trials remain at their actual final state.', font=small, fill='#b8cbd5')
            draw.text((25, 693), 'Classical RGB plan + learned motor skills + encoder feedback. Fixed destinations; no arbitrary-position claim.', font=small, fill='#b8cbd5')
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.asarray(canvas), format='rgb24')):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        for _, _, _, renderer, _ in cases:
            renderer.close()
        if container:
            container.close()
    info = {'fps': fps, 'frames': frames, 'duration_seconds': frames/fps, 'resolution': [1280, 720],
            'playback_speed': speed, 'all_seeds': [r['seed'] for r in rows], 'passed': audit['passed_by_split']['evaluation'],
            'bytes': args.output.stat().st_size, 'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
            'state_sha256': {str(r['seed']): hashlib.sha256(((args.evidence/r['report']).parent/'states.npz').read_bytes()).hexdigest() for r in rows},
            'method': 'All ten authoritative state traces at a shared 4x timeline. No simulation rerun or omitted trial. Captions only; no audio.'}
    require_space(args.output, 1024**2)
    args.output.with_suffix('.json').write_bytes((json.dumps(info, indent=2)+'\n').encode())
    print(json.dumps({k: v for k, v in info.items() if k != 'state_sha256'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, default=Path('docs/robotics/evidence/composed-dinner-v1'))
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
