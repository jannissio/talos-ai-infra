"""Physical development/evaluation of the RGB observer and neural motor map.

The runner owns scene setup and a privileged stop/score monitor. The controller
receives only RGB estimates, table calibration, requested goal and robot joints.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import subprocess
import time
import xml.etree.ElementTree as ET
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from PIL import Image
import torch
from simulation_lab.dinner_monitor import DinnerPhysicalMonitor
from simulation_lab.experiment_targets import with_bottle_destination
from simulation_lab.policy_control import apply_targets
from simulation_lab.rgb_bottle_observer import RgbBottleObserver
from simulation_lab.rgb_servo_cameras import VIEWS, camera_argument, calibration
from simulation_lab.rgb_servo_controller import RgbServoBottle
from simulation_lab.rgb_servo_motor import CartesianMotorPolicy
from simulation_lab.scene import HOME, build_scene
from simulation_lab.storage import require_space


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads(args.protocol.read_text())
    allowed = (range(protocol['training_seed_range'][0], protocol['training_seed_range'][1]+1)
               if args.split == 'training' else protocol[args.split+'_seeds'])
    if args.seed not in allowed:
        raise ValueError('Seed is not in the declared split.')
    require_space(args.output, 48*1024**2); args.output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    source_files = ['scripts/evaluate_rgb_servo_bottle.py', 'simulation_lab/rgb_servo_controller.py',
                    'simulation_lab/rgb_servo_motor.py', 'simulation_lab/rgb_bottle_observer.py',
                    'simulation_lab/rgb_servo_cameras.py', 'simulation_lab/rgb_servo_network.py',
                    'simulation_lab/rgb_servo_geometry.py',
                    'simulation_lab/experiment_targets.py', 'simulation_lab/policy_control.py']
    if args.openvino:
        source_files.append('simulation_lab/rgb_servo_openvino.py')
    source_hashes = {}
    for name in source_files:
        require_space(args.output, 1024**2)
        payload = (root/name).read_bytes(); copied = args.output/'evaluated-source'/name
        copied.parent.mkdir(parents=True, exist_ok=True); copied.write_bytes(payload)
        source_hashes[name] = hashlib.sha256(payload).hexdigest()
    rng = np.random.default_rng(args.seed); bounds = protocol['workspace_m']
    x, y, yaw = rng.uniform(*bounds['x']), rng.uniform(*bounds['y']), rng.uniform(-.6, .6)
    destination = protocol['destinations_m'][args.seed % 3]
    xml, layout = build_scene(seed=args.seed, scenario='dinner', dinner_preset='task')
    layout = with_bottle_destination(layout, destination)
    model = mujoco.MjModel.from_xml_string(xml); data = mujoco.MjData(model); model.vis.quality.offsamples = 0
    data.qpos[:12] = HOME*2; data.ctrl[:] = HOME*2
    address = int(model.joint('bottle_free').qposadr[0]); body = model.body('bottle').id
    data.qpos[address:address+7] = [x, y, layout['table_z']+.001, math.cos(yaw/2), 0., 0., math.sin(yaw/2)]
    mujoco.mj_forward(model, data)
    overlap = max([-c.dist for c in data.contact if c.dist < 0 and body in [model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]]], default=0.)
    for _ in range(300):
        mujoco.mj_step(model, data)
    data.time = 0.
    scene = ET.fromstring(xml); compiler = scene.find('compiler')
    compiler.set('meshdir', os.path.relpath(compiler.get('meshdir'), args.output).replace('\\', '/'))
    (args.output/'scene.xml').write_text(ET.tostring(scene, encoding='unicode'))
    report = {'seed': args.seed, 'split': args.split, 'mode': args.mode, 'initial_pose': [x, y, yaw],
              'source_sha256': source_hashes, 'source_parent_git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'destination_m': destination, 'initial_overlap_m': float(overlap), 'layout': layout,
              'observer_sha256': hashlib.sha256(args.observer.read_bytes()).hexdigest(),
              'motor_sha256': hashlib.sha256(args.motor.read_bytes()).hexdigest(), 'physics_state_writes_during_control': 0,
              'hidden_forces': 0, 'inverse_solver_calls_during_control': 0,
              'scope': 'Experimental RGB approach/descent correction; post-grasp Cartesian carry uses the observed closure offset and robot proprioception.'}
    if overlap > .001 or data.body('bottle').xmat[8] < .98:
        report.update(status='invalid_start', message='Initial overlap or tipped bottle; refused.')
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n'); print(json.dumps(report)); return
    torch.set_num_threads(2)
    if args.openvino:
        from simulation_lab.rgb_servo_openvino import OpenVinoBottleObserver, OpenVinoCartesianMotorPolicy
        parity = json.loads((args.openvino/'parity.json').read_text())
        if parity['observer_sha256'] != report['observer_sha256'] or parity['motor_sha256'] != report['motor_sha256']:
            raise ValueError('OpenVINO exports do not match the selected weights.')
        if not all(record['parity_passed'] for record in parity['networks'].values()):
            raise ValueError('OpenVINO exports have not passed numerical parity.')
        observer = OpenVinoBottleObserver(args.openvino/'observer.xml', minimum_views=args.minimum_views,
                                          device=args.openvino_device, rigid_geometry=args.rigid_geometry)
        motor = OpenVinoCartesianMotorPolicy(args.openvino/'motor.xml', args.motor, model, device=args.openvino_device)
        report['inference_runtime'] = 'OpenVINO '+args.openvino_device
        report['openvino_artifact_sha256'] = {name: hashlib.sha256((args.openvino/name).read_bytes()).hexdigest()
                                            for name in ('observer.xml', 'observer.bin', 'motor.xml', 'motor.bin')}
    else:
        observer = RgbBottleObserver(args.observer, args.device, args.minimum_views, args.rigid_geometry)
        motor = CartesianMotorPolicy(args.motor, model)
        report['inference_runtime'] = 'PyTorch '+args.device+' observer; PyTorch CPU motor'
    report['observer_minimum_views'] = args.minimum_views
    report['rigid_geometry'] = args.rigid_geometry
    controller = RgbServoBottle(motor, destination, layout['table_z'], args.mode)
    renderer = mujoco.Renderer(model, width=320, height=240); option = mujoco.MjvOption(); option.geomgroup[3:] = 0
    observations = []; trace = {k: [] for k in ('qpos', 'qvel', 'time', 'stage', 'targets')}
    monitor = None; target = np.asarray(HOME*2, dtype=float); began = time.perf_counter(); image_time = 0.
    push = json.loads(args.push.read_text()) if args.push else None
    push_origin = None; push_settled = None; push_peak_mm = 0.; push_tilt_deg = 0.; push_ticks = 0
    counterfactuals = []
    try:
        for tick in range(15001):
            now = float(data.time)
            if tick % 20 == 0:
                image_began = time.perf_counter(); images, calibrations = {}, {}
                for name in VIEWS:
                    renderer.update_scene(data, camera=camera_argument(name), scene_option=option)
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                    images[name] = renderer.render().copy(); calibrations[name] = calibration(renderer)
                estimate = observer.observe(images, calibrations)
                image_time += time.perf_counter()-image_began
                controller.accept_observation(estimate, now)
                if estimate['status'] == 'observed' and controller.initial_observation is not None and controller.stage in ('approach', 'descend'):
                    try:
                        live = controller.preview_image_action(controller.current_rgb_estimate, now, data.qpos[:12].copy())
                        frozen = controller.preview_image_action(controller.initial_observation, now, data.qpos[:12].copy())
                        counterfactuals.append({'time_s': now, 'stage': controller.stage,
                                               'same_measured_joint_positions': data.qpos[:12].tolist(),
                                               'live_image_action': live.tolist(), 'initial_image_action': frozen.tolist(),
                                               'max_action_delta_rad': float(np.max(np.abs(live-frozen))),
                                               'observed_shift_mm': float(np.linalg.norm(controller.current_rgb_estimate-controller.initial_observation)*1000)})
                    except ValueError as exc:
                        counterfactuals.append({'time_s': now, 'refused': str(exc)})
                observations.append({'time_s': now, 'stage': controller.stage, **estimate})
                if tick % 1000 == 0:
                    require_space(args.output, 16*1024**2)
                    Image.fromarray(np.concatenate(list(images.values()), axis=1)).save(args.output/f'views-{tick:05d}.png')
            q, v = data.qpos.copy(), data.qvel.copy()
            target = controller.update(now, data.qpos[:12].copy(), data.qvel[:12].copy())
            assert np.array_equal(q, data.qpos) and np.array_equal(v, data.qvel)
            if tick % 10 == 0:
                for key, value in (('qpos', q), ('qvel', v), ('time', now), ('stage', controller.stage), ('targets', target.copy())):
                    trace[key].append(value)
            if monitor is None and controller.side:
                monitor = DinnerPhysicalMonitor(model, data, layout, 'bottle', controller.side)
                if not np.allclose(monitor.destination[:2], destination, atol=1e-12):
                    raise ValueError('Physical monitor did not receive the declared destination.')
            failure = monitor.update() if monitor else None
            if failure or (monitor and monitor.succeeded) or controller.status != 'running':
                break
            data.xfrc_applied[:] = 0.
            if push and push['starts_at_simulation_s'] <= now < push['starts_at_simulation_s']+push['duration_s']:
                if push_origin is None:
                    push_origin = data.body('bottle').xpos.copy()
                force = np.asarray(push['force_n'])
                application = data.body('bottle').xpos+[0., 0., push['application_height_above_bottle_origin_m']]
                data.xfrc_applied[body, :3] = force
                data.xfrc_applied[body, 3:] = np.cross(application-data.xipos[body], force)
                push_ticks += 1
            if push_origin is not None and now < push['starts_at_simulation_s']+1.2:
                push_peak_mm = max(push_peak_mm, float(np.linalg.norm(data.body('bottle').xpos[:2]-push_origin[:2])*1000))
                push_tilt_deg = max(push_tilt_deg, float(np.degrees(np.arccos(np.clip(data.body('bottle').xmat[8], -1, 1)))))
                push_settled = ((data.body('bottle').xpos-push_origin)*1000).tolist()
            assert model.neq == 0 and not np.any(data.qfrc_applied)
            assert not np.any(np.delete(data.xfrc_applied, body, axis=0))
            if not push:
                assert not np.any(data.xfrc_applied)
            data.ctrl[:] = apply_targets(model, data, target, .25, 0 if controller.side != 'right' else 6)
            mujoco.mj_step(model, data)
    finally:
        renderer.close()
    passed = bool(monitor and monitor.succeeded)
    report.update(status='succeeded' if passed else 'failed', controller_status=controller.status,
                  message=(monitor.failure if monitor and monitor.failure else controller.message),
                  stage=controller.stage, arm=controller.side, simulation_seconds=float(data.time),
                  wall_seconds=time.perf_counter()-began, image_pipeline_seconds=image_time,
                  physical=monitor.report() if monitor else None, stages=controller.history,
                  rgb_observations_accepted=controller.observations, rgb_refusals=controller.refusals)
    report.update(disclosed_push_protocol=args.push.as_posix() if args.push else None, disclosed_push_ticks=push_ticks,
                  rgb_filter={'type': 'exponential', 'new_observation_weight': .35, 'observation_hz': 10},
                  fixed_jaw_clearance_m=controller.fixed_jaw_clearance_m,
                  push_translation_before_grasp_mm=push_settled, push_peak_translation_mm=push_peak_mm,
                  push_peak_tilt_deg=push_tilt_deg,
                  image_counterfactual_max_action_delta_rad=max((r.get('max_action_delta_rad', 0.) for r in counterfactuals), default=0.))
    require_space(args.output, 32*1024**2)
    np.savez_compressed(args.output/'states.npz', **{k: np.asarray(v) for k, v in trace.items()})
    (args.output/'observations.json').write_text(json.dumps(observations, indent=2)+'\n')
    (args.output/'image-counterfactuals.json').write_text(json.dumps(counterfactuals, indent=2)+'\n')
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('layout', 'stages')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, default=Path('docs/robotics/experiments/rgb-servo-bottle-v1.json'))
    p.add_argument('--seed', type=int, default=2026095101)
    p.add_argument('--split', choices=['training', 'development', 'evaluation'], default='training')
    p.add_argument('--mode', choices=['live', 'frozen'], default='live')
    p.add_argument('--observer', type=Path, default=Path('.run/rgb-servo-observer-fit-v3/step-004000/model.safetensors'))
    p.add_argument('--motor', type=Path, default=Path('.run/rgb-servo-motor-fit-v1/model.safetensors'))
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--openvino', type=Path)
    p.add_argument('--openvino-device', default='CPU')
    p.add_argument('--minimum-views', type=int, choices=[2, 3], default=3)
    p.add_argument('--rigid-geometry', action='store_true')
    p.add_argument('--push', type=Path)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
