"""Render RGB supervision from replay-verified physical training motions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from simulation_lab.rgb_servo_cameras import VIEWS, KEYPOINTS, camera_argument, calibration, project
from simulation_lab.storage import GIB, require_space


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    summary = json.loads((args.source/'summary.json').read_text())
    eligible = [r for r in summary['rows'] if r['training_eligible']]
    if summary['attempts'] != summary['nominal_seeds_planned'] or not eligible:
        raise ValueError('Finish the bounded motion batch and its replay gates first.')
    require_space(args.output, GIB); args.output.mkdir(parents=True)
    rng = np.random.default_rng(2026095406)
    manifest = {'schema': 'talos.rgb-servo-motion-labels.v1', 'split': 'train', 'source': args.source.as_posix(),
                'source_attempts': summary['attempts'], 'eligible_motions': len(eligible), 'states_per_motion': 128,
                'states': len(eligible)*128, 'completed_states': 0, 'shards': [], 'views': list(VIEWS),
                'label_usage': 'Privileged keypoints/segmentation are labels only. Inference receives current RGB.',
                'physical_action_replay_verified': True, 'seed': 2026095406, 'reserve_gib': 10}
    started = time.perf_counter()
    for entry in eligible:
        require_space(args.output, 128*3*240*320*4)
        folder = args.source/f'seed-{entry["seed"]}'
        model = mujoco.MjModel.from_xml_path(str(folder/'scene.xml')); data = mujoco.MjData(model)
        model.vis.quality.offsamples = 0
        with np.load(folder/'replay-states.npz', allow_pickle=False) as source:
            states = {k: source[k] for k in source.files}
        selected = np.linspace(0, len(states['time'])-1, 128).astype(int)
        renderer = mujoco.Renderer(model, width=320, height=240)
        option = mujoco.MjvOption(); option.geomgroup[3:] = 0
        body = model.body('bottle').id; geoms = np.flatnonzero(model.geom_bodyid == body)
        original_light, original_ambient = model.light_diffuse.copy(), model.vis.headlight.ambient.copy()
        rows = {k: [] for k in ['rgb', 'mask', 'keypoints_px', 'visible', 'world_points', 'present']}
        try:
            for index in selected:
                data.qpos[:] = states['qpos'][index]; data.qvel[:] = states['qvel'][index]
                model.light_diffuse[:] = original_light*rng.uniform(.85, 1.15)
                model.vis.headlight.ambient[:] = original_ambient*rng.uniform(.85, 1.15)
                mujoco.mj_forward(model, data)
                points = KEYPOINTS@data.body('bottle').xmat.reshape(3, 3).T+data.body('bottle').xpos
                images, masks, pixels, visible, calibrations = [], [], [], [], {}
                for name in VIEWS:
                    renderer.update_scene(data, camera=camera_argument(name), scene_option=option)
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                    calibrations[name] = calibration(renderer)
                    images.append(renderer.render().copy())
                    renderer.enable_segmentation_rendering(); segmentation = renderer.render().copy(); renderer.disable_segmentation_rendering()
                    mask = np.isin(segmentation[:, :, 0], geoms) & (segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
                    xy = project(points, calibrations[name]['projection'])
                    valid = mask.sum() >= 40 and ((xy[:, 0] >= 2) & (xy[:, 0] < 318) & (xy[:, 1] >= 2) & (xy[:, 1] < 238)).all()
                    masks.append(mask.astype('uint8')); pixels.append(xy); visible.append(valid)
                rows['rgb'].append(images); rows['mask'].append(masks); rows['keypoints_px'].append(pixels)
                rows['visible'].append(visible); rows['world_points'].append(points); rows['present'].append(True)
        finally:
            renderer.close()
        require_space(args.output, 128*3*240*320*4)
        filename = f'seed-{entry["seed"]}.npz'
        dtypes = {'rgb': 'uint8', 'mask': 'uint8', 'keypoints_px': 'float32', 'visible': 'bool', 'world_points': 'float32', 'present': 'bool'}
        np.savez_compressed(args.output/filename, **{k: np.asarray(v, dtype=dtypes[k]) for k, v in rows.items()})
        manifest['shards'].append({'file': filename, 'states': 128, 'seed': entry['seed'], 'sha256': hashlib.sha256((args.output/filename).read_bytes()).hexdigest()})
        manifest['completed_states'] += 128
        manifest.update(calibrations=calibrations, wall_seconds=time.perf_counter()-started)
        (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        print(json.dumps({'seed': entry['seed'], 'completed_states': manifest['completed_states'], 'wall_seconds': manifest['wall_seconds']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
