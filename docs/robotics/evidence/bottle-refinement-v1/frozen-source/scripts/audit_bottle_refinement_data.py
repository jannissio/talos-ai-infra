"""Independent geometry/hash audit and fixed-index RGB regeneration of each split."""
import argparse
import hashlib
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from scripts.bottle_refinement_experiment import checked_protocol, read, sha, write, space
from simulation_lab.rgb_servo_cameras import KEYPOINTS, VIEWS, camera_argument, project


def run(args):
    p = checked_protocol(args.protocol); folder = ROOT/p['raw_root']/args.split
    output = folder/'audit.json'
    if output.exists():
        raise FileExistsError('Preserve prior data audits.')
    manifest = read(folder/'manifest.json')
    assert manifest['protocol_sha256'] == sha(args.protocol)
    assert sha(folder/'recipes.npz') == manifest['recipes_sha256']
    space(p, output, 8*1024**2)
    with np.load(folder/'recipes.npz', allow_pickle=False) as z:
        recipes = {key: z[key].copy() for key in z.files}
    count = manifest['states']; assert len(recipes['qpos']) == count
    model = mujoco.MjModel.from_xml_path(str(folder/'scene.xml')); model.vis.quality.offsamples = 0
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=320, height=240)
    option = mujoco.MjvOption(); option.geomgroup[3:] = 0
    settings = read(ROOT/p['camera_protocol'])['camera_configurations'][p['camera_configuration']]
    sample = {0, 1, 7, 31, 127, count//2, count-1}
    checked, regenerated, max_world, max_pixel = 0, 0, 0., 0.
    try:
        for shard in manifest['shards']:
            path = folder/shard['file']; assert sha(path) == shard['sha256']
            with np.load(path, allow_pickle=False) as stored:
                # NpzFile access decompresses an entire array each time. Keep
                # one bounded shard in memory while retaining identical checks.
                z = {key: stored[key] for key in stored.files}
                for row in range(len(z['rgb'])):
                    index = checked
                    data.qpos[:] = recipes['qpos'][index]; data.qvel[:] = 0.; data.time = 0.
                    model.light_diffuse[:] = recipes['light_diffuse'][index]
                    model.vis.headlight.ambient[:] = recipes['ambient'][index]
                    mujoco.mj_forward(model, data)
                    body = data.body('bottle')
                    expected = KEYPOINTS@body.xmat.reshape(3, 3).T+body.xpos
                    world_error = float(np.max(np.abs(expected-z['world_points'][row])))
                    max_world = max(max_world, world_error)
                    assert world_error < 1e-12
                    assert bool(z['present'][row]) == bool(recipes['present'][index])
                    for slot, name in enumerate(VIEWS):
                        image = z['rgb'][row, slot]
                        assert hashlib.sha256(image.tobytes()).hexdigest() == recipes['rgb_sha256'][index, slot]
                        pixel = project(expected, manifest['calibrations'][slot]['projection'])
                        error = float(np.max(np.abs(pixel-z['keypoints_px'][row, slot])))
                        max_pixel = max(max_pixel, error)
                        assert error < 1e-3
                        visible = bool(z['present'][row] and z['mask'][row, slot].sum() >= p['data']['minimum_foreground_pixels']
                            and np.all(pixel >= 2) and np.all(pixel < [318, 238]))
                        assert visible == bool(z['visible'][row, slot])
                        if index in sample:
                            camera = camera_argument(name)
                            camera.azimuth, camera.elevation, camera.distance = settings[name]
                            renderer.update_scene(data, camera=camera, scene_option=option)
                            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                            assert np.array_equal(renderer.render(), image), (index, name)
                            regenerated += 1
                    checked += 1
    finally:
        renderer.close()
    assert checked == count
    result = {'schema': p['schema'], 'split': args.split, 'passed': True,
        'states_geometry_checked': checked, 'rgb_hashes_checked': checked*3,
        'rgb_views_independently_regenerated': regenerated, 'regenerated_state_indices': sorted(sample),
        'maximum_world_label_discrepancy_m': max_world, 'maximum_projection_discrepancy_px': max_pixel,
        'recipes_sha256': sha(folder/'recipes.npz'), 'data_manifest_sha256': sha(folder/'manifest.json'),
        'protocol_sha256': sha(args.protocol), 'auditor_sha256': sha(Path(__file__)),
        'scope': 'Every stored image hash, pose/keypoint/visibility label and camera projection checked; seven fixed states independently re-rendered. Synthetic posed observation audit, not physical manipulation.'}
    write(output, result)
    print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--split', choices=['training', 'development', 'evaluation'], required=True)
    run(parser.parse_args())
