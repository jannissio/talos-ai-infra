"""Score the frozen RGB observer gate on a dataset or exposed physical trace."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from PIL import Image, ImageDraw
import torch
from simulation_lab.rgb_bottle_observer import RgbBottleObserver
from simulation_lab.rgb_servo_cameras import VIEWS, KEYPOINTS, camera_argument, calibration, project
from simulation_lab.storage import require_space


def dataset_rows(folder):
    manifest = json.loads((folder/'manifest.json').read_text())
    for shard in manifest['shards']:
        with np.load(folder/shard['file'], allow_pickle=False) as data:
            for i in range(len(data['rgb'])):
                yield {name: data['rgb'][i, j] for j, name in enumerate(VIEWS)}, manifest['calibrations'], data['world_points'][i], bool(data['present'][i]), None


def trace_rows(folder):
    model = mujoco.MjModel.from_xml_path(str(folder/'scene.xml')); data = mujoco.MjData(model)
    model.vis.quality.offsamples = 0
    with np.load(folder/'states.npz', allow_pickle=False) as source:
        states = {key: source[key] for key in source.files}
    candidates = np.flatnonzero(states['stage'] == 'bottle')
    if not len(candidates):
        raise ValueError('This trace has no bottle skill.')
    indices = np.unique(candidates[np.linspace(0, len(candidates)-1, 256).astype(int)])
    renderer = mujoco.Renderer(model, width=320, height=240)
    option = mujoco.MjvOption(); option.geomgroup[3:] = 0
    try:
        for index in indices:
            data.qpos[:] = states['qpos'][index]; data.qvel[:] = states['qvel'][index]
            data.time = float(states['time'][index]); mujoco.mj_forward(model, data)
            images, calibrations = {}, {}
            for name in VIEWS:
                renderer.update_scene(data, camera=camera_argument(name), scene_option=option)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                images[name] = renderer.render().copy(); calibrations[name] = calibration(renderer)
            # Scoring only; never passed to RgbBottleObserver.observe.
            truth = KEYPOINTS@data.body('bottle').xmat.reshape(3, 3).T+data.body('bottle').xpos
            yield images, calibrations, truth, True, data.time
    finally:
        renderer.close()


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    require_space(args.output, 32*1024**2); args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    observer = RgbBottleObserver(args.checkpoint, args.device)
    records = trace_rows(args.recording) if args.recording else dataset_rows(args.dataset)
    rows = []; began = time.perf_counter()
    for index, (images, calibrations, truth, present, seconds) in enumerate(records):
        row = observer.observe(images, calibrations)
        row.update(index=index, scoring_only_present=present, simulation_seconds=seconds)
        if row['status'] == 'observed' and present:
            row['scoring_only_error_mm'] = float(np.linalg.norm(np.asarray(row['keypoints_m'])-truth, axis=1).max()*1000)
        rows.append(row)
        if index % 32 == 0:
            require_space(args.output, 4*1024**2)
            canvas = Image.new('RGB', (960, 280), '#0d1c2b'); draw = ImageDraw.Draw(canvas)
            for j, name in enumerate(VIEWS):
                canvas.paste(Image.fromarray(images[name]), (j*320, 0))
                for u, v in row['views'][name]['keypoints_px']:
                    draw.ellipse((j*320+u-3, v-3, j*320+u+3, v+3), outline='lime')
                if present:
                    for u, v in project(truth, calibrations[name]['projection']):
                        draw.rectangle((j*320+u-2, v-2, j*320+u+2, v+2), outline='red')
            draw.text((10, 246), f'{index}: {row["status"]}; green RGB prediction; red scoring-only reference', fill='white')
            canvas.save(args.output/f'frame-{index:03d}.png')
    errors = [r['scoring_only_error_mm'] for r in rows if 'scoring_only_error_mm' in r]
    quantiles = {'median': float(np.median(errors)), 'p95': float(np.quantile(errors, .95)), 'max': max(errors)} if errors else None
    result = {'protocol': 'docs/robotics/experiments/rgb-servo-confidence-v1.json',
              'source': (args.recording or args.dataset).as_posix(), 'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'frames': len(rows), 'present': sum(r['scoring_only_present'] for r in rows),
              'accepted_present': sum(r['status'] == 'observed' and r['scoring_only_present'] for r in rows),
              'false_accepted_absent': sum(r['status'] == 'observed' and not r['scoring_only_present'] for r in rows),
              'refused': sum(r['status'] == 'refused' for r in rows), 'error_mm': quantiles,
              'wall_seconds': time.perf_counter()-began, 'rows': rows,
              'scope': 'RGB perception accuracy only, not a physical corrective-control result.'}
    require_space(args.output, 4*1024**2)
    (args.output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--dataset', type=Path)
    source.add_argument('--recording', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
