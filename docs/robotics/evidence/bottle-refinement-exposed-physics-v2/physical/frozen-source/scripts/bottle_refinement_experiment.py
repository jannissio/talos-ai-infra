"""Bounded coarse-to-fine bottle vision experiment for wider manipulation."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from PIL import Image
import torch
from safetensors.torch import load_file, save_file
from simulation_lab.scene import HOME, build_scene
from simulation_lab.rgb_servo_cameras import VIEWS, KEYPOINTS, camera_argument, calibration, project
from simulation_lab.rgb_servo_network import BottleKeypointNet, decode_heatmaps
from simulation_lab.bottle_refinement import KeypointCropRefiner, image_crops, decode, reconstruct, refinement_loss
from simulation_lab.storage import require_space


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    payload = (json.dumps(value, indent=2)+'\n').encode() if not isinstance(value, bytes) else value
    require_space(path, len(payload)+1024)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(payload)


def arrays(path, **values):
    if path.exists():
        raise FileExistsError('Preserve earlier arrays.')
    require_space(path, sum(np.asarray(v).nbytes for v in values.values())+1024**2)
    with path.open('xb') as stream:
        np.savez_compressed(stream, **values)


def checked_protocol(path):
    p = read(path)
    if p['schema'] != 'talos.bottle-refinement.v1':
        raise ValueError('Unknown refinement protocol.')
    for name, digest in p['source_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('A declared dependency changed: '+name)
    return p


def space(p, destination, expected):
    folders = [ROOT/p[key] for key in ('raw_root', 'model_package', 'training_package', 'evidence_package')]
    used = sum(f.stat().st_size for folder in folders if folder.exists() for f in folder.rglob('*') if f.is_file())
    if used+expected > p['budget']['maximum_raw_and_packaged_gib']*1024**3:
        raise ValueError('The declared cumulative storage budget is exhausted.')
    return require_space(destination, expected)


def declare(path):
    if path.exists():
        raise FileExistsError(path)
    coverage = ROOT/'docs/robotics/evidence/manipulation-coverage-v1'
    evidence = read(coverage/'audit.json')
    assert evidence['all_200_outcomes_retained']
    pool = []
    for row in evidence['cases']:
        if row['case']['skill'] == 'bottle' and row['outcomes']['learned']['valid_start']:
            for mode in ('teacher', 'learned'):
                pool.append((coverage/(row['case']['id']+'-'+mode)/'states.npz').relative_to(ROOT).as_posix())
    coarse = 'docs/robotics/evidence/rgb-servo-observer-camera-v1/candidates/step-003000.safetensors'
    camera = 'docs/robotics/experiments/rgb-servo-visibility-v1.json'
    sources = [Path(__file__).relative_to(ROOT).as_posix(), 'simulation_lab/bottle_refinement.py',
        'simulation_lab/rgb_servo_network.py', 'simulation_lab/rgb_servo_cameras.py', 'simulation_lab/rgb_servo_geometry.py',
        'simulation_lab/scene.py', 'simulation_lab/dinner.py', coarse, camera,
        (coverage/'audit.json').relative_to(ROOT).as_posix(), *pool]
    p = {'schema': 'talos.bottle-refinement.v1', 'declared_on': '2026-09-14',
        'purpose': 'Improve wider bottle manipulation with current-image refinement and the existing target-conditioned neural motor/route controller. The unchanged preset policy is the fallback and a physical comparator.',
        'evidence': 'Coverage diagnostic: 0/24 bottle position-grid successes (two invalid resets); exact-state controller demonstrates 15/24. Earlier broad RGB control is limited by missing/inaccurate image estimates. The separate camera-adapted coarse model failed its 6 mm maximum-error test; it remains frozen and unpromoted.',
        'raw_root': '.run/bottle-refinement-v1', 'model_package': 'models/bottle_refinement_v1',
        'training_package': 'training/bottle_refinement_v1', 'evidence_package': 'docs/robotics/evidence/bottle-refinement-v1',
        'coarse_checkpoint': coarse, 'camera_protocol': camera, 'camera_configuration': 'opposite_obliques',
        'state_pool': pool, 'source_sha256': {name: sha(ROOT/name) for name in sources},
        'data': {'training_states': 4096, 'development_states': 512, 'evaluation_states': 512,
            'training_rng_seed': 2026111001, 'development_rng_seed': 2026111002, 'evaluation_rng_seed': 2026111003,
            'scene_seed': 2026111000, 'states_per_shard': 64, 'absent_fraction': .10,
            'physical_context_fraction': .50, 'source_xy_m': [[-.16, -.20], [.23, .09]], 'lift_m': [0., .08],
            'yaw_rad': [-.6, .6], 'tilt_rad': [-.07, .07], 'robot_joint_jitter_std_rad': .025,
            'physical_object_translation_jitter_m': [.012, .012, .006],
            'minimum_foreground_pixels': 40, 'crop_size': 64, 'training_crop_jitter_px': 4.,
            'method': 'Half independently posed broad source/lift examples; half sampled actual physical context frames from every valid exposed bottle diagnostic, including failed controllers. Add bounded reset-only pose/joint/light variations. Every absent and failed observation is retained. No training on old held-out perception arrays; new seeds sample all three splits.'},
        'training': {'seed': 2026111004, 'updates': 8000, 'batch_size': 64, 'checkpoints': [4000, 8000],
            'learning_rate': .001, 'final_learning_rate': .00002, 'weight_decay': .0001,
            'loss': 'One local heatmap with 0.9 px target sigma, coordinate regression and visibility. Frozen coarse network; only the new crop refiner is fitted.',
            'selection': 'Finish both checkpoints. Among complete development-gate passes choose most accepted present states, then lowest p95, then earliest checkpoint. No threshold revision or further fit in this protocol.'},
        'inference': {'minimum_visibility_probability': .9, 'minimum_heatmap_peak': .03,
            'maximum_heatmap_variance_px2': 12., 'minimum_views': 2,
            'grasp_workspace_m': [[-.20, -.25, .86], [.28, .14, 1.0]],
            'contract': 'Three RGB views -> frozen coarse image points -> 64 px bilinear RGB crops with semantic point role -> new heatmaps/visibility -> current image points -> fixed calibration and unchanged 103 mm rigid geometry. No object state, depth, segmentation, teacher or IK at inference.'},
        'perception_gate': {'minimum_present_acceptance_fraction': .95, 'maximum_false_absent_accepts': 0,
            'maximum_p95_error_mm': 2., 'maximum_error_mm': 4.},
        'physical': {'workspace_m': {'x': [-.14, .16], 'y': [-.18, -.06], 'yaw': [-.6, .6]},
            'destinations_xy_m': [[.10, -.115], [.16, -.06], [.20, .05]],
            'development_seeds': list(range(2026111101, 2026111107)),
            'evaluation_seeds': list(range(2026111201, 2026111225)),
            'motor': 'docs/robotics/evidence/rgb-servo-v5/physical/model',
            'routing': 'Frozen V2 R1 neural route selection and original phase timings; no new motor fit or physical threshold change.',
            'gates': 'Only after full development/fresh perception and FP32 export parity pass: freeze integration sources before all six development cases, live and unchanged preset-baseline conditions. Require at least 4/6 live and strict improvement. Freeze before 24 fresh cases with live, frozen-image and unchanged preset-baseline conditions; require at least 18/24 nominal live, at least six more successes than preset baseline, and no failure on six exposed task-preset regressions. Retain invalid starts/refusals in planned physical denominators. Any visual-feedback benefit needs its own live/frozen comparison, not just higher success than a different controller.'},
        'budget': {'maximum_raw_and_packaged_gib': 4, 'maximum_generation_minutes_per_split': 12,
            'maximum_fit_minutes': 12, 'maximum_torch_reserved_gib': 10, 'maximum_new_physical_trials': 90, 'reserve_gib': 10},
        'stop_rule': 'Stop this protocol before subsequent reserved stages if a gate fails. Preserve both candidates and all inputs/outcomes. Do not promote a perception-only result or tune on the final physical set. Continue the broader goal through a separately declared follow-on if needed.'}
    # Count six anchors plus 6x2 development and 24x3 final comparisons = 90.
    write(path, p)
    print(json.dumps({'protocol_sha256': sha(path), 'physical_context_traces': len(pool), 'training_states': p['data']['training_states']}), flush=True)


def collect(args):
    p = checked_protocol(args.protocol); settings = p['data']
    output = ROOT/p['raw_root']/args.split
    if output.exists():
        raise FileExistsError('Preserve earlier data splits.')
    if args.split == 'evaluation':
        selection = read(ROOT/p['raw_root']/'fit/selection.json')
        if not selection['development_gate_passed']:
            raise ValueError('Fresh perception requires a frozen passing selection.')
    preflight = space(p, output, 1024**3 if args.split == 'training' else 192*1024**2)
    output.mkdir(parents=True)
    poses = []
    for name in p['state_pool']:
        with np.load(ROOT/name, allow_pickle=False) as trace:
            poses.append(trace['qpos'][::10].copy())
    poses = np.concatenate(poses)
    xml, layout = build_scene(seed=settings['scene_seed'], scenario='dinner', dinner_preset='task')
    model = mujoco.MjModel.from_xml_string(xml); model.vis.quality.offsamples = 0
    data = mujoco.MjData(model); data.qpos[:12] = HOME*2; data.ctrl[:] = HOME*2
    initial = data.qpos.copy()
    address = int(model.joint('bottle_free').qposadr[0]); body_id = model.body('bottle').id
    geom_ids = np.flatnonzero(model.geom_bodyid == body_id)
    original_light = model.light_diffuse.copy(); original_ambient = model.vis.headlight.ambient.copy()
    camera_settings = read(ROOT/p['camera_protocol'])['camera_configurations'][p['camera_configuration']]
    rng = np.random.default_rng(settings[args.split+'_rng_seed'])
    count = settings[args.split+'_states']; started = time.perf_counter()
    renderer = mujoco.Renderer(model, width=320, height=240)
    option = mujoco.MjvOption(); option.geomgroup[3:] = 0
    recipes = {k: [] for k in ('qpos', 'light_diffuse', 'ambient', 'present', 'physical_context', 'rgb_sha256')}
    batch = {k: [] for k in ('rgb', 'mask', 'keypoints_px', 'visible', 'world_points', 'present')}
    shards, calibrations = [], []
    try:
        for index in range(count):
            space(p, output, 4*1024**2)
            if time.perf_counter()-started > p['budget']['maximum_generation_minutes_per_split']*60:
                raise TimeoutError('Declared generation time reached.')
            physical = rng.random() < settings['physical_context_fraction']
            sampled = poses[int(rng.integers(len(poses)))]
            data.qpos[:] = sampled if physical else initial
            data.qvel[:] = 0.; data.time = 0.
            data.qpos[:12] = sampled[:12] if rng.random() < .8 else np.array(HOME*2)
            data.qpos[:12] += rng.normal(0, settings['robot_joint_jitter_std_rad'], 12)
            data.qpos[:12] = np.clip(data.qpos[:12], model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
            if physical:
                data.qpos[address:address+3] += rng.uniform(-1, 1, 3)*settings['physical_object_translation_jitter_m']
                data.qpos[address+2] = np.clip(data.qpos[address+2], layout['table_z']+.001, layout['table_z']+.081)
            else:
                xy = rng.uniform(*np.asarray(settings['source_xy_m']))
                lift = 0. if rng.random() < .4 else rng.uniform(*settings['lift_m'])
                angle, tilt = rng.uniform(*settings['yaw_rad']), rng.uniform(*settings['tilt_rad'])
                yaw_q = np.array([np.cos(angle/2), 0, 0, np.sin(angle/2)])
                tilt_q = np.array([np.cos(tilt/2), np.sin(tilt/2), 0, 0]); quat = np.zeros(4)
                mujoco.mju_mulQuat(quat, yaw_q, tilt_q)
                data.qpos[address:address+7] = [*xy, layout['table_z']+.001+lift, *quat]
            present = rng.random() >= settings['absent_fraction']
            if not present:
                data.qpos[address:address+3] = [2., 2., 2.]
            model.light_diffuse[:] = original_light*rng.uniform(.65, 1.25, (model.nlight, 1))*rng.uniform(.92, 1.08, (1, 3))
            model.vis.headlight.ambient[:] = original_ambient*rng.uniform(.7, 1.3)
            mujoco.mj_forward(model, data)
            world = KEYPOINTS@data.body('bottle').xmat.reshape(3, 3).T+data.body('bottle').xpos
            images, masks, points, visible, digests, camera_rows = [], [], [], [], [], []
            for name in VIEWS:
                camera = camera_argument(name)
                camera.azimuth, camera.elevation, camera.distance = camera_settings[name]
                renderer.update_scene(data, camera=camera, scene_option=option)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                image = renderer.render().copy(); cal = calibration(renderer)
                renderer.enable_segmentation_rendering(); segmentation = renderer.render().copy(); renderer.disable_segmentation_rendering()
                mask = np.isin(segmentation[:, :, 0], geom_ids) & (segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
                xy = project(world, cal['projection'])
                confident_label = bool(present and mask.sum() >= settings['minimum_foreground_pixels'] and np.all(xy >= 2) and np.all(xy < [318, 238]))
                images.append(image); masks.append(mask.astype('uint8')); points.append(xy)
                visible.append(confident_label); digests.append(hashlib.sha256(image.tobytes()).hexdigest()); camera_rows.append(cal)
            if not calibrations:
                calibrations = camera_rows
            if index in (0, 1, 7, 31, 127, count-1):
                space(p, output, 2*1024**2)
                with (output/f'example-{index:04d}.png').open('xb') as stream:
                    Image.fromarray(np.concatenate(images, axis=1)).save(stream, format='PNG')
            for key, value in zip(recipes, (data.qpos.copy(), model.light_diffuse.copy(), model.vis.headlight.ambient.copy(), present, physical, digests)):
                recipes[key].append(value)
            for key, value in zip(batch, (images, masks, points, visible, world, present)):
                batch[key].append(value)
            if len(batch['rgb']) == settings['states_per_shard'] or index == count-1:
                space(p, output, 64*1024**2)
                name = f'shard-{len(shards):03d}.npz'
                arrays(output/name, rgb=np.asarray(batch['rgb'], np.uint8), mask=np.asarray(batch['mask'], np.uint8),
                    keypoints_px=np.asarray(batch['keypoints_px'], np.float32), visible=np.asarray(batch['visible'], bool),
                    world_points=np.asarray(batch['world_points'], np.float64), present=np.asarray(batch['present'], bool))
                shards.append({'file': name, 'states': len(batch['rgb']), 'sha256': sha(output/name)})
                batch = {key: [] for key in batch}
                if index % 256 == 255 or index == count-1:
                    print(json.dumps({'split': args.split, 'rendered': index+1, 'wall_seconds': time.perf_counter()-started}), flush=True)
    finally:
        renderer.close()
    space(p, output, 8*1024**2)
    arrays(output/'recipes.npz', **{key: np.asarray(value) for key, value in recipes.items()})
    scene = ET.fromstring(xml); scene.find('compiler').set('meshdir', os.path.relpath(ROOT/'simulation_lab/assets/so101/assets', output).replace('\\', '/'))
    write(output/'scene.xml', ET.tostring(scene, encoding='utf-8'))
    write(output/'manifest.json', {'schema': p['schema'], 'protocol_sha256': sha(args.protocol), 'split': args.split,
        'states': count, 'completed_states': count, 'views': list(VIEWS), 'shards': shards, 'calibrations': calibrations,
        'recipes_sha256': sha(output/'recipes.npz'), 'source_sha256': p['source_sha256'], 'preflight': preflight,
        'wall_seconds': time.perf_counter()-started, 'scope': 'Synthetic posed RGB perception data; no new physics success is claimed.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declare', type=Path)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--action', choices=['collect'])
    parser.add_argument('--split', choices=['training', 'development', 'evaluation'])
    args = parser.parse_args()
    if args.declare:
        declare(args.declare)
    elif args.action == 'collect':
        collect(args)
    else:
        parser.error('Choose declare or collect.')
