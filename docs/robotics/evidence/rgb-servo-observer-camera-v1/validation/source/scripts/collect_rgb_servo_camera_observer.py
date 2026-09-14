"""Collect declared paired camera data without changing the original generator."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from simulation_lab.rgb_servo_cameras import VIEWS, KEYPOINTS, camera_argument, calibration, project
from simulation_lab.scene import build_scene, HOME
from simulation_lab.storage import GIB, require_space


def collect(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads(args.protocol.read_text())
    root=Path(__file__).resolve().parents[1]
    experiment=root/'.run/rgb-servo-observer-camera-v1'
    if not args.output.resolve().is_relative_to(experiment):
        raise ValueError('Keep this experiment inside its declared data directory.')
    if protocol['schema']!='talos.rgb-servo-observer-camera-training.v1' or args.camera_configuration not in protocol['camera_configurations']:
        raise ValueError('Unexpected protocol or camera configuration.')
    if args.split=='evaluation':
        if not args.selection:
            raise ValueError('A frozen training selection must pass development before fresh perception is generated.')
        selection=json.loads(args.selection.read_text())
        if selection['protocol_sha256']!=hashlib.sha256(args.protocol.read_bytes()).hexdigest() or not selection['development_gate_passed']:
            raise ValueError('Selection does not qualify for fresh perception.')
        if hashlib.sha256((root/selection['checkpoint']).read_bytes()).hexdigest()!=selection['checkpoint_sha256']:
            raise ValueError('Selected observer changed.')
    camera_protocol_path=root/protocol['camera_protocol']
    camera_protocol=json.loads(camera_protocol_path.read_text())
    def configured_camera(name):
        camera=camera_argument(name)
        camera.azimuth,camera.elevation,camera.distance=camera_protocol['camera_configurations'][args.camera_configuration][name]
        return camera
    def budget(expected):
        folders=[experiment,root/'docs/robotics/evidence/rgb-servo-observer-camera-v1',root/'models/bottle_servo_camera_v1']
        used=sum(p.stat().st_size for folder in folders if folder.exists() for p in folder.rglob('*') if p.is_file())
        used+=sum(p.stat().st_size for p in (root/'.run/final-goal').glob('rgb-servo-observer-camera-v1*') if p.is_file())
        if used+expected>protocol['budget']['maximum_new_data_gib']*GIB:
            raise ValueError('The cumulative experiment data budget would be exceeded.')
        require_space(args.output,expected)
    budget(128*1024**2)
    count = args.states or protocol[f'{args.split}_states']
    if count > protocol[f'{args.split}_states']:
        raise ValueError('The declared split budget cannot be exceeded.')
    require_space(args.output, min(2*GIB, count*3*240*320*4+32*1024**2))
    args.output.mkdir(parents=True)
    rng = np.random.default_rng(protocol[f'{args.split}_rng_seed'])
    xml, layout = build_scene(seed=42, scenario='dinner', dinner_preset='task')
    model = mujoco.MjModel.from_xml_string(xml)
    model.vis.quality.offsamples = 0
    data = mujoco.MjData(model)
    data.qpos[:12] = HOME*2
    data.ctrl[:] = HOME*2
    mujoco.mj_forward(model, data)
    initial = data.qpos.copy()
    address = int(model.joint('bottle_free').qposadr[0])
    bottle = model.body('bottle').id
    geom_ids = np.flatnonzero(model.geom_bodyid == bottle)
    pose_arrays=[]
    for name in protocol['arm_pose_sources']:
        with np.load(root/name,allow_pickle=False) as archive:
            pose_arrays.append(archive['qpos'][:,:12].copy())
    arm_poses=np.concatenate(pose_arrays)
    renderer = mujoco.Renderer(model, width=320, height=240)
    option = mujoco.MjvOption(); option.geomgroup[3:] = 0
    original_lights = model.light_diffuse.copy()
    original_ambient = model.vis.headlight.ambient.copy()
    calibrations = {}
    manifest = {'protocol': args.protocol.as_posix(), 'protocol_sha256': hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
                'split': args.split, 'states': count, 'views': list(VIEWS), 'shards': [],
                'labels_only': 'Projected body keypoints and geometry segmentation never enter the deployed observer.',
                'static_perception_examples': True, 'physical_demonstrations': False,
                'robot_pose_sources': protocol['arm_pose_sources'], 'reserve_gib': 10,
                'camera_configuration':args.camera_configuration,
                'camera_protocol_sha256':hashlib.sha256(camera_protocol_path.read_bytes()).hexdigest(),
                'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'parent_git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                'arm_pose_source_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in protocol['arm_pose_sources']}}
    recipe={name:[] for name in ('qpos','light_diffuse','headlight_ambient','present')}

    started = time.perf_counter()
    image_rows, mask_rows, point_rows, visible_rows, world_rows, present_rows = [], [], [], [], [], []
    try:
        for index in range(count):
            require_space(args.output, 4*1024**2)
            data.qpos[:] = initial
            data.qvel[:] = 0
            data.qpos[:12] = arm_poses[int(rng.integers(len(arm_poses)))] if rng.random() < .8 else np.array(HOME*2)
            data.qpos[:12] += rng.normal(0, .025, 12)
            data.qpos[:12] = np.clip(data.qpos[:12], model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
            present = bool(rng.random() >= .1)
            x, y, lift = rng.uniform(-.16, .23), rng.uniform(-.20, .09), rng.uniform(0., .08)
            if rng.random() < .4:
                lift = 0.
            angle = rng.uniform(-.6, .6)
            tilt = rng.uniform(-.07, .07)
            yaw = np.array([np.cos(angle/2), 0., 0., np.sin(angle/2)])
            lean = np.array([np.cos(tilt/2), np.sin(tilt/2), 0., 0.])
            quat = np.zeros(4); mujoco.mju_mulQuat(quat, yaw, lean)
            data.qpos[address:address+7] = [x, y, layout['table_z']+lift+.001, *quat]
            if not present:
                data.qpos[address:address+3] = [2., 2., 2.]
            model.light_diffuse[:] = original_lights*rng.uniform(.65, 1.25, (model.nlight, 1))*rng.uniform(.92, 1.08, (1, 3))
            model.vis.headlight.ambient[:] = original_ambient*rng.uniform(.7, 1.3)
            mujoco.mj_forward(model, data)
            points = KEYPOINTS@data.body('bottle').xmat.reshape(3, 3).T+data.body('bottle').xpos
            for name,value in [('qpos',data.qpos.copy()),('light_diffuse',model.light_diffuse.copy()),
                               ('headlight_ambient',model.vis.headlight.ambient.copy()),('present',present)]:
                recipe[name].append(value)
            images, masks, pixels, visibility = [], [], [], []
            for name in VIEWS:
                renderer.update_scene(data, camera=configured_camera(name), scene_option=option)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                calibrations.setdefault(name, calibration(renderer))
                rgb = renderer.render().copy()
                renderer.enable_segmentation_rendering()
                segmentation = renderer.render().copy()
                renderer.disable_segmentation_rendering()
                mask = np.isin(segmentation[:, :, 0], geom_ids) & (segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
                keypoints = project(points, calibrations[name]['projection'])
                in_frame = ((keypoints[:, 0] >= 2) & (keypoints[:, 0] < 318) & (keypoints[:, 1] >= 2) & (keypoints[:, 1] < 238)).all()
                visible = present and mask.sum() >= 40 and in_frame
                images.append(rgb); masks.append(mask.astype('uint8')); pixels.append(keypoints); visibility.append(visible)
                if index < 6:
                    preview = Image.fromarray(rgb); draw = ImageDraw.Draw(preview)
                    if present:
                        for u, v in keypoints:
                            draw.ellipse((u-3, v-3, u+3, v+3), outline='lime', width=1)
                    preview.save(args.output/f'preview-{index:03d}-{name}.png')
            image_rows.append(images); mask_rows.append(masks); point_rows.append(pixels)
            visible_rows.append(visibility); world_rows.append(points); present_rows.append(present)
            if len(image_rows) == 64 or index == count-1:
                budget(len(image_rows)*3*240*320*4+8*1024**2)
                filename = f'shard-{len(manifest["shards"]):03d}.npz'
                np.savez_compressed(args.output/filename, rgb=np.asarray(image_rows, dtype='uint8'),
                                    mask=np.asarray(mask_rows, dtype='uint8'), keypoints_px=np.asarray(point_rows, dtype='float32'),
                                    visible=np.asarray(visible_rows, dtype='bool'), world_points=np.asarray(world_rows, dtype='float32'),
                                    present=np.asarray(present_rows, dtype='bool'))
                manifest['shards'].append({'file': filename, 'states': len(image_rows), 'sha256': hashlib.sha256((args.output/filename).read_bytes()).hexdigest()})
                image_rows.clear(); mask_rows.clear(); point_rows.clear(); visible_rows.clear(); world_rows.clear(); present_rows.clear()
                np.savez_compressed(args.output/'sampled-states.npz',**{name:np.asarray(values) for name,values in recipe.items()})
                manifest['sampled_states_sha256']=hashlib.sha256((args.output/'sampled-states.npz').read_bytes()).hexdigest()
                manifest.update(calibrations=calibrations, completed_states=index+1, wall_seconds=time.perf_counter()-started)
                (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
                print(json.dumps({'split': args.split, 'completed_states': index+1, 'wall_seconds': manifest['wall_seconds']}), flush=True)
    finally:
        renderer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=Path('docs/robotics/experiments/rgb-servo-observer-camera-training-v1.json'))
    parser.add_argument('--split', choices=['train', 'development', 'evaluation'], required=True)
    parser.add_argument('--states', type=int, default=0)
    parser.add_argument('--camera-configuration',choices=['original','opposite_obliques'],required=True)
    parser.add_argument('--selection',type=Path)
    parser.add_argument('--output', type=Path, required=True)
    collect(parser.parse_args())
