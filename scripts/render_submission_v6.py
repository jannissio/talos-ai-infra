"""Render the six-skill submission baseline from its saved physical states."""
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
from simulation_lab.storage import GIB, require_space


def render(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    require_space(args.output, GIB)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    font_path = Path('C:/Windows/Fonts/arial.ttf')
    font = lambda size: ImageFont.truetype(str(font_path), size) if font_path.exists() else ImageFont.load_default(size=size)
    title_font, body_font, small_font = font(42), font(29), font(21)
    container = av.open(str(args.output), 'w', options={'movflags': 'faststart'})
    stream = container.add_stream('libx264', rate=20)
    stream.width, stream.height, stream.pix_fmt = 1280, 720, 'yuv420p'
    stream.options = {'crf': '19', 'preset': 'fast'}
    frames, chapters = 0, []
    started = time.perf_counter()

    def emit(canvas):
        nonlocal frames
        if frames % 200 == 0:
            require_space(args.output, 128*1024**2)
        for packet in stream.encode(av.VideoFrame.from_ndarray(np.asarray(canvas), format='rgb24')):
            container.mux(packet)
        frames += 1

    def card(title, lines, seconds):
        chapters.append({'title': title, 'type': 'explanation', 'start_s': frames/20, 'duration_s': seconds})
        canvas = Image.new('RGB', (1280, 720), '#0d1c2b'); draw = ImageDraw.Draw(canvas)
        draw.text((65, 80), title, font=title_font, fill='#66e4d5')
        for index, line in enumerate(lines):
            draw.text((65, 230+index*65), line, font=body_font, fill='white')
        draw.text((65, 655), 'TALOS  /  AI Infra Summit Hackathon  /  September 2026', font=small_font, fill='#b8cbd5')
        for _ in range(seconds*20):
            emit(canvas)

    def trace(folder, instruction, speed):
        require_space(args.output, 256*1024**2)
        report = json.loads((folder/'report.json').read_text())
        if report['active'] or report['status'] != 'succeeded':
            raise ValueError('Main demonstration needs a terminal successful trace; failures remain in the all-seed montage.')
        with np.load(folder/'states.npz', allow_pickle=False) as source:
            states = {k: source[k] for k in source.files}
        model = mujoco.MjModel.from_xml_path(str(folder/'scene.xml')); data = mujoco.MjData(model)
        model.vis.quality.offsamples = 0
        renderer = mujoco.Renderer(model, width=640, height=480)
        option = mujoco.MjvOption(); option.geomgroup[3:] = 0
        duration = float(states['time'][-1]); first = frames
        chapters.append({'title': instruction, 'type': 'recorded_physics', 'start_s': first/20,
                         'seed': report['seed'], 'playback_speed': speed, 'simulation_seconds': duration,
                         'source_recording': folder.as_posix(), 'recording_sha256': hashlib.sha256((folder/'states.npz').read_bytes()).hexdigest(),
                         'completed_steps': report['completed_steps'], 'status': report['status']})
        try:
            for frame in range(math.ceil((duration/speed+2)*20)):
                seconds = min(frame/20*speed, duration)
                index = int(np.clip(np.searchsorted(states['time'], seconds, side='right')-1, 0, len(states['time'])-1))
                data.qpos[:] = states['qpos'][index]; data.qvel[:] = states['qvel'][index]
                data.time = float(states['time'][index]); mujoco.mj_forward(model, data)
                canvas = Image.new('RGB', (1280, 720), '#0d1c2b'); draw = ImageDraw.Draw(canvas)
                draw.text((25, 20), instruction, font=title_font, fill='white')
                draw.text((25, 75), f'Learned control   |   Seed {report["seed"]}   |   {speed}x playback   |   {seconds:.1f} simulated seconds', font=small_font, fill='#66e4d5')
                for n, camera in enumerate(['opposite', 'overhead']):
                    renderer.update_scene(data, camera=camera, scene_option=option)
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                    canvas.paste(Image.fromarray(renderer.render().copy()), (n*640, 115))
                    draw.text((n*640+12, 602), 'Opposite side' if n == 0 else 'Overhead', font=small_font, fill='#b8cbd5')
                skill = str(states['stage'][index]).replace('_', ' ')
                label = 'PASS: physical release and parked arms verified' if seconds >= duration else 'Current skill: '+skill
                draw.text((25, 642), label, font=body_font, fill='#66e4d5')
                draw.text((25, 685), 'Recorded physical states. Objects move through contact. Table-supported relays release before regrasping.', font=small_font, fill='white')
                emit(canvas)
        finally:
            renderer.close()
        chapters[-1]['duration_s'] = (frames-first)/20
        print(json.dumps(chapters[-1]), flush=True)

    try:
        card('Talos: learned dinner-table robotics', [
            'Two SO-101 arms, one physical MuJoCo scene.',
            'Six learned skills and bottle relays in both directions.',
            'Supported typed commands and Speechmatics voice input.'], 6)
        card('From instruction to movement', [
            'A bounded language parser selects supported skills.',
            'Initial RGB features condition each neural motor policy.',
            'Motor feedback regulates progress; contact checks score outcomes.',
            'The following clips show the selected neural baseline.'], 8)
        trace(args.dinner, '“Set the table” — six learned skills', 4)
        trace(args.forward, '“Pass the bottle to the right arm”', 2)
        trace(args.reverse, '“Pass the bottle to the left arm”', 2)
        card('All frozen outcomes are retained', [
            'Full learned dinner sequence: 8 / 10 randomized task starts.',
            'Failures: mug 14.59 mm and spoon 13.61 mm placement error.',
            'Bottle relays: 5 / 5 in each direction, fixed destinations.',
            'Separate ten-seed videos show every trial, including failures.'], 10)
        card('Intel deployment evidence', [
            'Original bottle: Intel i7-10850H CPU 0.157 ms median.',
            'Intel UHD iGPU: 0.688 ms median. FP32 network inference only.',
            'These are historical laptop results, not the new suite.',
            'Final-suite Intel execution and hardware eligibility remain open.'], 9)
        card('What is still being developed', [
            'Continuous RGB correction and broader placement coverage.',
            'Other-object handoffs, pouring and unrestricted language.',
            'This baseline uses initial images and bounded starting regions.',
            'No airborne handoff, liquids or physical robot is claimed.'], 8)
        card('Reproduce and inspect', [
            'Code, licensed assets, models and exact compact training inputs.',
            'Fresh local browser trials; private hosted demo verification.',
            'github.com/jannissio/talos-ai-infra',
            'Public release is reserved for the final submission step.'], 6)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
    result = {'resolution': [1280, 720], 'fps': 20, 'duration_seconds': frames/20,
              'bytes': args.output.stat().st_size, 'wall_seconds': time.perf_counter()-started,
              'audio': 'No audio track; instructions and explanations are visible captions.', 'chapters': chapters,
              'scope': 'Versioned six-skill submission baseline, not a claim that broader development or Intel verification is complete.'}
    if result['duration_seconds'] > 300 or result['bytes'] >= 300_000_000:
        raise ValueError('Video exceeds submission limits.')
    require_space(args.output, 1024**2)
    args.output.with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'chapters'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dinner', type=Path, default=Path('.run/final-goal/dinner-final-2026091901-v6'))
    parser.add_argument('--forward', type=Path, default=Path('.run/final-goal/relay-final-2026092501-v3'))
    parser.add_argument('--reverse', type=Path, default=Path('.run/final-goal/reverse-final-2026092601-v3'))
    parser.add_argument('--output', type=Path, required=True)
    render(parser.parse_args())
