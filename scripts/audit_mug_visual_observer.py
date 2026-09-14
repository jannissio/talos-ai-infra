"""Regenerate every compact observer example and independently check its labels."""
import argparse
import hashlib
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from simulation_lab.mug_visual_observer import encode_images
from simulation_lab.rgb_servo_cameras import calibration
from scripts.mug_observer_experiment import load_protocol, read, sha, space, write_json
from scripts.probe_mug_visual_visibility import fixed_camera


def audit(args):
    p = load_protocol(args.protocol)
    folder = ROOT/p['raw_root']/args.split
    output = folder/'audit.json'
    if output.exists():
        raise FileExistsError('Preserve the existing data audit.')
    manifest = read(folder/'manifest.json')
    if manifest['protocol_sha256'] != sha(args.protocol) or sha(folder/'data.npz') != manifest['data_sha256']:
        raise ValueError('Declared data changed.')
    for name, digest in manifest['source_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('A recorded data source changed: '+name)
    with np.load(folder/'data.npz', allow_pickle=False) as archive:
        rows = {name: archive[name] for name in archive.files}
    count = p['data'][args.split+'_states']
    if len(rows['qpos']) != count:
        raise ValueError('Every declared example is required.')
    preflight = space(p, output, 4*1024**2)
    visibility = read(ROOT/p['visibility_protocol'])
    cameras = visibility['camera_configurations'][p['camera_configuration']]
    maximum_label_error = 0.
    maximum_feature_error = 0.
    mismatches, checked = [], 0
    started = time.perf_counter()
    for episode, name in enumerate(manifest['episodes']):
        indices = np.flatnonzero(rows['episode_index'] == episode)
        if not len(indices):
            continue
        model = mujoco.MjModel.from_xml_path(str(ROOT/p['initial_scene_root']/name/'scene.xml'))
        model.vis.quality.offsamples = 0
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, width=320, height=240)
        option = mujoco.MjvOption()
        option.geomgroup[3:] = 0
        try:
            for index in indices:
                data.qpos[:] = rows['qpos'][index]
                data.qvel[:] = 0
                data.time = 0
                mujoco.mj_forward(model, data)
                label = data.body('mug').xpos+data.body('mug').xmat.reshape(3, 3)@np.asarray(p['data']['target_local_m'])
                error = float(np.max(abs(label-rows['labels_m'][index])))
                maximum_label_error = max(maximum_label_error, error)
                if error > 1e-12:
                    mismatches.append([int(index), 'geometric label'])
                images, calibrations = [], []
                for slot, camera in enumerate(cameras):
                    renderer.update_scene(data, camera=fixed_camera(camera), scene_option=option)
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                    image = np.round(renderer.render().astype(float)*rows['exposure_scale'][index]).clip(0, 255).astype(np.uint8)
                    if hashlib.sha256(image.tobytes()).hexdigest() != rows['rgb_sha256'][index, slot]:
                        mismatches.append([int(index), slot, 'RGB bytes'])
                    images.append(image)
                    calibrations.append(calibration(renderer))
                if calibrations != manifest['calibration_by_episode'][name]:
                    mismatches.append([int(index), 'fixed calibration'])
                feature, observations = encode_images(images, calibrations, visibility['rgb_method'])
                difference = float(np.max(abs(feature-rows['features'][index])))
                maximum_feature_error = max(maximum_feature_error, difference)
                if difference != 0 or sum(r['usable'] for r in observations) != int(rows['usable_views'][index]):
                    mismatches.append([int(index), 'image features or acceptance'])
                if not np.array_equal([r['rgb_pixels'] for r in observations], rows['rgb_component_pixels'][index]):
                    mismatches.append([int(index), 'component count'])
                if bool(rows['present'][index]) != (int(index) % p['data']['absent_every_nth_state'] != 0):
                    mismatches.append([int(index), 'presence label'])
                checked += 1
        finally:
            renderer.close()
        print(f'Audited {args.split}: {checked}/{count} compact states', flush=True)
    result = {'schema': p['schema'], 'protocol_sha256': sha(args.protocol), 'source_sha256': sha(Path(__file__)),
        'split': args.split, 'data_sha256': sha(folder/'data.npz'), 'states': checked, 'rgb_views': checked*3,
        'maximum_geometric_label_difference_m': maximum_label_error, 'maximum_feature_difference': maximum_feature_error,
        'mismatches': mismatches, 'passed': checked == count and not mismatches, 'wall_seconds': time.perf_counter()-started,
        'preflight': preflight, 'scope': 'Every stored image hash, fixed calibration, RGB feature, acceptance and geometric label independently reconstructed from the compact state recipe. No physical trial or fitting.'}
    space(p, output, 4*1024**2)
    write_json(output, result)
    print({key: result[key] for key in ('split', 'states', 'rgb_views', 'maximum_geometric_label_difference_m', 'maximum_feature_difference', 'passed', 'wall_seconds')})
    if not result['passed']:
        raise ValueError('Compact data reproduction failed; preserve the audit.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT/'docs/robotics/experiments/mug-visual-observer-v1.json')
    parser.add_argument('--split', choices=['training', 'development', 'evaluation'], required=True)
    audit(parser.parse_args())
