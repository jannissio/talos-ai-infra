"""Re-render preserved camera recipes and compare pixels, labels and CPU scores."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from simulation_lab.rgb_servo_cameras import VIEWS, KEYPOINTS, camera_argument, calibration, project
from simulation_lab.rgb_servo_openvino import OpenVinoBottleObserver
from simulation_lab.storage import require_space
from scripts.package_rgb_servo_camera_training import arrays, read, sha


def run(args):
    if args.output.exists():
        raise FileExistsError('Use a fresh replay output; preserve previous evidence.')
    folder = args.package/'validation'
    for entry in read(folder/'manifest.json')['files']:
        if sha(folder/entry['file']) != entry['sha256']:
            raise ValueError('Packaged input changed: '+entry['file'])
    sources = read(folder/'reproduction-inputs.json')
    for name in ('simulation_lab/rgb_servo_cameras.py', 'simulation_lab/rgb_servo_geometry.py',
                 'simulation_lab/rgb_servo_openvino.py', 'simulation_lab/rgb_bottle_observer.py',
                 'simulation_lab/rgb_servo_network.py'):
        if sha(ROOT/name) != sources['source_sha256'][name]:
            raise ValueError('Use the frozen observer runtime for this reproduction.')
    for name, digest in sources['repository_asset_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('A frozen scene asset changed.')
    current = folder/'data'/args.dataset
    manifest, states, labels = read(current/'manifest.json'), arrays(current/'sampled-states.npz'), arrays(current/'labels.npz')
    image_hashes = read(current/'rgb-sha256.json')['states']
    indices = args.indices if args.indices is not None else list(range(manifest['states']))
    if len(set(indices)) != len(indices) or any(i < 0 or i >= manifest['states'] for i in indices):
        raise ValueError('Replay indices must be unique and inside the preserved dataset.')
    require_space(args.output, 16*1024**2)
    args.output.mkdir(parents=True)
    model = mujoco.MjModel.from_xml_path(str((folder/'scene.xml').resolve()))
    model.vis.quality.offsamples = 0
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=320, height=240)
    option = mujoco.MjvOption()
    option.geomgroup[3:] = 0
    camera_config = read(folder/'camera-protocol.json')['camera_configurations'][manifest['camera_configuration']]
    geoms = np.flatnonzero(model.geom_bodyid == model.body('bottle').id)
    observer = OpenVinoBottleObserver(folder/'openvino/observer.xml', minimum_views=2, rigid_geometry=True)
    expected_path = folder/(manifest['split']+'-cpu-'+manifest['camera_configuration'])/'results.json'
    expected = read(expected_path)['rows'] if expected_path.is_file() else None
    rows, started = [], time.perf_counter()
    try:
        for index in indices:
            require_space(args.output, 1024**2)
            data.qpos[:] = states['qpos'][index]
            data.qvel[:] = 0
            model.light_diffuse[:] = states['light_diffuse'][index]
            model.vis.headlight.ambient[:] = states['headlight_ambient'][index]
            mujoco.mj_forward(model, data)
            truth = KEYPOINTS @ data.body('bottle').xmat.reshape(3,3).T + data.body('bottle').xpos
            images, calibrations, views = {}, {}, []
            for j, name in enumerate(VIEWS):
                camera = camera_argument(name)
                camera.azimuth, camera.elevation, camera.distance = camera_config[name]
                renderer.update_scene(data, camera=camera, scene_option=option)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                images[name] = renderer.render().copy()
                calibrations[name] = calibration(renderer)
                renderer.enable_segmentation_rendering()
                segmentation = renderer.render().copy()
                renderer.disable_segmentation_rendering()
                mask = np.isin(segmentation[:,:,0], geoms) & (segmentation[:,:,1] == int(mujoco.mjtObj.mjOBJ_GEOM))
                points = project(truth, calibrations[name]['projection'])
                in_frame = ((points[:,0] >= 2) & (points[:,0] < 318) & (points[:,1] >= 2) & (points[:,1] < 238)).all()
                visible = bool(states['present'][index] and mask.sum() >= 40 and in_frame)
                views.append(dict(view=name, exact_rgb_sha256=hashlib.sha256(images[name].tobytes()).hexdigest() == image_hashes[index][j],
                                  exact_mask=bool(np.array_equal(mask, labels['mask'][index,j])),
                                  exact_projected_labels=bool(np.array_equal(points.astype(np.float32), labels['keypoints_px'][index,j])),
                                  exact_visibility=visible == bool(labels['visible'][index,j])))
            observation = observer.observe(images, calibrations)
            error = float(np.linalg.norm(np.asarray(observation['keypoints_m'])-labels['world_points'][index], axis=1).max()*1000) if observation['status'] == 'observed' and states['present'][index] else None
            old = expected[index] if expected else None
            rows.append(dict(index=index, views=views, observation=observation, scoring_only_error_mm=error,
                             same_acceptance=None if old is None else old['status'] == observation['status'],
                             absolute_score_difference_mm=abs(error-old['scoring_only_error_mm']) if error is not None and old and old.get('scoring_only_error_mm') is not None else None))
    finally:
        renderer.close()
    result = dict(dataset=args.dataset, states=len(rows), views=3*len(rows), source_sha256=sha(Path(__file__)),
                  sampled_states_sha256=sha(current/'sampled-states.npz'), observer_sha256=read(args.package/'selection.json')['checkpoint_sha256'],
                  all_pixels_exact=all(v['exact_rgb_sha256'] for row in rows for v in row['views']),
                  all_labels_exact=all(v[k] for row in rows for v in row['views'] for k in ('exact_mask','exact_projected_labels','exact_visibility')),
                  acceptance_changes=sum(row['same_acceptance'] is False for row in rows),
                  maximum_score_difference_mm=max((row['absolute_score_difference_mm'] for row in rows if row['absolute_score_difference_mm'] is not None), default=None),
                  wall_seconds=time.perf_counter()-started, rows=rows,
                  scope='Reproduction on previously exposed samples, not another independent test. Driver-dependent pixel differences are reported explicitly.')
    require_space(args.output, 4*1024**2)
    (args.output/'replay.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}))
    return result['all_labels_exact'] and result['acceptance_changes'] == 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=ROOT/'docs/robotics/evidence/rgb-servo-observer-camera-v1')
    parser.add_argument('--dataset', choices=[s+'-'+c for s in ('train','development','evaluation') for c in ('original','opposite_obliques')], required=True)
    parser.add_argument('--indices', type=int, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args) else 1)
