"""Frozen physical follow-through for bottle refinement V1, with paired controls.

The experimental controller receives RGB estimates, the explicit requested XY,
fixed robot geometry and measured joints. Privileged state only resets, stops or
scores a rollout. No selected dinner runtime is changed by this experiment.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from PIL import Image
import torch
from scripts.bottle_refinement_experiment import checked_protocol, read, sha, space, write, arrays
from simulation_lab.bottle_refinement_runtime import RefinedBottleObserver
from simulation_lab.dinner_monitor import DinnerPhysicalMonitor
from simulation_lab.experiment_targets import with_bottle_destination
from simulation_lab.learned_dinner import LearnedDinnerTask
from simulation_lab.policy_control import apply_targets
from simulation_lab.rgb_servo_cameras import VIEWS, KEYPOINTS, camera_argument, calibration
from simulation_lab.rgb_servo_openvino import OpenVinoCartesianMotorPolicy
from simulation_lab.rgb_servo_routing import RoutedRgbServoBottle, plan_bottle_route
from simulation_lab.scene import HOME, build_scene


def integration_path(p):
    return ROOT / p['raw_root'] / 'physical/integration.json'


def checked_integration(p, protocol):
    path = integration_path(p)
    spec = read(path)
    if spec['protocol_sha256'] != sha(protocol):
        raise ValueError('Physical integration belongs to another protocol.')
    for name, digest in spec['frozen_sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('A frozen physical dependency changed: '+name)
    return spec


def declare(protocol):
    p = checked_protocol(protocol)
    raw = ROOT / p['raw_root']
    folder = raw / 'physical'
    if folder.exists():
        raise FileExistsError('Preserve the physical integration declaration.')
    result = read(raw / 'fresh-perception.json')
    if not result['passed'] or result['protocol_sha256'] != sha(protocol):
        raise ValueError('The reserved perception gate must pass before physical exposure.')
    if result['freeze_sha256'] != sha(raw / 'fresh-perception-freeze.json'):
        raise ValueError('The perception selection freeze changed.')
    if result['evaluation_manifest_sha256'] != sha(raw / 'evaluation/manifest.json') or result['evaluation_audit_sha256'] != sha(raw / 'evaluation/audit.json'):
        raise ValueError('The fresh perception data or its audit changed.')
    RefinedBottleObserver(raw / 'openvino')
    motor = ROOT / p['physical']['motor']
    routing = ROOT / 'docs/robotics/experiments/rgb-servo-routing-v2.json'
    source_names = [path.relative_to(ROOT).as_posix() for path in (ROOT / 'simulation_lab').glob('*.py')]
    source_names += ['scripts/evaluate_refined_bottle_physics.py', 'scripts/bottle_refinement_experiment.py']
    inputs = [raw / 'fresh-perception.json', raw / 'fresh-perception-freeze.json',
        raw / 'evaluation/audit.json', raw / 'openvino/parity-development.json', routing,
        motor / 'experiment.json', motor / 'motor.safetensors', motor / 'openvino/motor.xml', motor / 'openvino/motor.bin']
    inputs += list((raw / 'openvino').glob('*'))
    inputs += [path for path in (ROOT / 'models/bottle_visual').rglob('*') if path.is_file()]
    frozen = {name: sha(ROOT / name) for name in source_names}
    frozen.update({path.relative_to(ROOT).as_posix(): sha(path) for path in inputs if path.is_file()})
    preflight = space(p, folder, 512 * 1024**2)
    spec = {'schema': 'talos.bottle-refinement-physics.v1', 'protocol_sha256': sha(protocol),
        'source_parent_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'frozen_sha256': frozen, 'routing': routing.relative_to(ROOT).as_posix(),
        'observer': (raw / 'openvino').relative_to(ROOT).as_posix(),
        'motor': p['physical']['motor'], 'baseline': 'models/bottle_visual',
        'regression_seeds': list(range(2026091301, 2026091307)),
        'development_modes': ['live', 'baseline'], 'evaluation_modes': ['live', 'frozen', 'baseline'],
        'maximum_workers': 2, 'maximum_simulated_seconds': 100., 'maximum_wall_seconds': 180.,
        'settle_steps': 300, 'preflight': preflight,
        'gate': p['physical']['gates'],
        'reset': 'Fresh declared dinner task scene seed; replace bottle source XY/yaw once before 300 settling steps. No resampling. Exposed regressions retain the original task-preset source.',
        'validity': 'Initial bottle penetration >1 mm against another body, tipping, unsupported or unsettled reset is invalid. Preserve all planned denominators. Refusal is a failure, not an invalid scene.',
        'requested_destination': 'Explicit semantic XY supplied to routed control and the independent monitor. Preset baseline receives its unchanged RGB/joint inputs; visual target markers retain their original scene coordinates.',
        'frozen_ablation': 'The initial RGB estimate is frozen separately for each leg; the next leg observes only after physical release and parking. Current RGB is still scored but cannot update frozen motor inputs.',
        'scope': 'Upright bottle, wider source and three destinations; V2 R1 phases/routes and V5 motor unchanged. No perturbation, other-object/general language or continuous carry-feedback claim.'}
    write(integration_path(p), spec)
    for name in source_names:
        write(folder / 'frozen-source' / name, (ROOT / name).read_bytes())
    print({'physical_integration_sha256': sha(integration_path(p)), 'maximum_trials': 90}, flush=True)


def setup(p, spec, seed, split, folder):
    xml, layout = build_scene(seed=seed, scenario='dinner', dinner_preset='task')
    model = mujoco.MjModel.from_xml_string(xml)
    model.vis.quality.offsamples = 0
    data = mujoco.MjData(model)
    data.qpos[:12] = HOME*2
    data.ctrl[:] = HOME*2
    address = int(model.joint('bottle_free').qposadr[0])
    if split == 'regression':
        destination = next(t['position_m'][:2] for t in layout['targets'] if t['object_id'] == 'bottle')
        source = data.qpos[address:address+7].copy()
    else:
        rng = np.random.default_rng(seed)
        bounds = p['physical']['workspace_m']
        x, y, yaw = rng.uniform(*bounds['x']), rng.uniform(*bounds['y']), rng.uniform(*bounds['yaw'])
        source = np.array([x, y, layout['table_z']+.001, math.cos(yaw/2), 0., 0., math.sin(yaw/2)])
        data.qpos[address:address+7] = source
        destination = p['physical']['destinations_xy_m'][seed % 3]
    mujoco.mj_forward(model, data)
    body = model.body('bottle').id
    penetration = max((-float(c.dist) for c in data.contact
        if model.geom_bodyid[c.geom1] != model.geom_bodyid[c.geom2]
        and body in (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2])), default=0.)
    for _ in range(spec['settle_steps']):
        mujoco.mj_step(model, data)
    data.time = 0.
    mujoco.mj_forward(model, data)
    dof = int(model.joint('bottle_free').dofadr[0])
    speed = float(np.linalg.norm(data.qvel[dof:dof+3]))
    valid = penetration <= .001 and data.body('bottle').xmat[8] > .98 and data.body('bottle').xpos[2] >= layout['table_z']-.004 and speed < .005
    scene = ET.fromstring(xml)
    scene.find('compiler').set('meshdir', os.path.relpath(ROOT/'simulation_lab/assets/so101/assets', folder).replace('\\', '/'))
    write(folder / 'scene.xml', ET.tostring(scene, encoding='utf-8'))
    state = {'valid': bool(valid), 'requested_source_qpos': source.tolist(),
        'settled_source_xyz_m': data.body('bottle').xpos.tolist(), 'initial_penetration_m': penetration,
        'settled_upright_z': float(data.body('bottle').xmat[8]), 'settled_speed_m_s': speed,
        'reason': None if valid else 'Initial overlap, tipped/unsupported bottle or unsettled reset.'}
    return model, data, layout, destination, state


def trial(args):
    p = checked_protocol(args.protocol)
    spec = checked_integration(p, args.protocol)
    allowed = spec['regression_seeds'] if args.split == 'regression' else p['physical'][args.split+'_seeds']
    modes = ['live'] if args.split == 'regression' else spec[args.split+'_modes']
    if args.seed not in allowed or args.mode not in modes:
        raise ValueError('Trial is outside the declared conditions.')
    if args.split in ('evaluation', 'regression'):
        physical_root = ROOT / p['raw_root'] / 'physical'
        gate = read(physical_root / 'development/gate.json')
        frozen = read(physical_root / 'evaluation-freeze.json')
        if not gate['passed'] or frozen['development_gate_sha256'] != sha(physical_root/'development/gate.json') or frozen['integration_sha256'] != sha(integration_path(p)):
            raise ValueError('Reserved physical trials require the passing, frozen development gate.')
    root = ROOT / p['raw_root'] / 'physical' / args.split
    folder = root / f'{args.seed}-{args.mode}'
    if folder.exists():
        raise FileExistsError('Preserve every physical attempt.')
    preflight = space(p, folder, 24*1024**2)
    folder.mkdir(parents=True)
    torch.set_num_threads(2)
    trace = {k: [] for k in ('qpos', 'qvel', 'time', 'stage', 'targets')}
    observations, counterfactuals, completed = [], [], []
    task = renderer = controller = monitor = route = None
    began = time.perf_counter()
    report = {'seed': args.seed, 'split': args.split, 'mode': args.mode,
        'protocol_sha256': sha(args.protocol), 'integration_sha256': sha(integration_path(p)),
        'preflight': preflight, 'status': 'failed', 'message': 'Global trial timeout.',
        'physics_state_writes_during_control': 0, 'hidden_forces': 0, 'inverse_solver_calls_during_control': 0,
        'inference_runtime': 'OpenVINO CPU FP32' if args.mode != 'baseline' else 'Unchanged CPU preset policy',
        'scope': spec['scope']}
    image_seconds = 0.
    try:
        model, data, layout, destination, state = setup(p, spec, args.seed, args.split, folder)
        report.update(setup=state, destination_m=destination, layout=layout)
        target = data.ctrl.copy()
        initial_others = {o['id']: data.body(o['id']).xpos.copy() for o in layout['objects'] if o['id'] != 'bottle'}
        maximum_other = 0.
        if not state['valid']:
            report.update(status='invalid_start', message=state['reason'])
        elif args.mode == 'baseline':
            task = LearnedDinnerTask(model, data, with_bottle_destination(layout, destination), ROOT/spec['baseline'], 'bottle')
        else:
            observer = RefinedBottleObserver(ROOT/spec['observer'])
            motor_folder = ROOT/spec['motor']
            motor = OpenVinoCartesianMotorPolicy(motor_folder/'openvino/motor.xml', motor_folder/'motor.safetensors', model, device='CPU')
            routing = read(ROOT/spec['routing'])
            renderer = mujoco.Renderer(model, width=320, height=240)
            option = mujoco.MjvOption(); option.geomgroup[3:] = 0
            camera_settings = read(ROOT/p['camera_protocol'])['camera_configurations'][p['camera_configuration']]
        leg_index, epoch, waiting_since = 0, 0., 0.
        for tick in range(int(spec['maximum_simulated_seconds']/model.opt.timestep)+1):
            now = float(data.time)
            stage = 'invalid_start' if not state['valid'] else 'observe'
            q, v = data.qpos.copy(), data.qvel.copy()
            if state['valid'] and task is not None:
                task.update(target)
                stage = task.stage
            elif state['valid']:
                if tick % 20 == 0:
                    start = time.perf_counter()
                    images, calibrations = {}, {}
                    for name in VIEWS:
                        camera = camera_argument(name)
                        camera.azimuth, camera.elevation, camera.distance = camera_settings[name]
                        renderer.update_scene(data, camera=camera, scene_option=option)
                        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                        images[name] = renderer.render().copy()
                        calibrations[name] = calibration(renderer)
                    estimate = observer.observe(images, calibrations)
                    image_seconds += time.perf_counter()-start
                    truth = KEYPOINTS[1] @ data.body('bottle').xmat.reshape(3, 3).T + data.body('bottle').xpos
                    observations.append({'time_s': now, 'leg': leg_index, 'observation': estimate,
                        'scoring_only_grasp_point_m': truth.tolist(),
                        'scoring_only_error_mm': float(np.linalg.norm(np.asarray(estimate['grasp_point_m'])-truth)*1000) if estimate['status'] == 'observed' else None})
                    if tick % 5000 == 0:
                        space(p, folder, 8*1024**2)
                        with (folder/f'views-{tick:05d}.png').open('xb') as stream:
                            Image.fromarray(np.concatenate(list(images.values()), axis=1)).save(stream, format='PNG')
                    if route is None and estimate['status'] == 'observed':
                        try:
                            route = plan_bottle_route(motor, estimate['grasp_point_m'], destination, layout['table_z'], args.mode, routing)
                        except ValueError as exc:
                            report['message'] = str(exc)
                            break
                    if route and controller is None and estimate['status'] == 'observed':
                        leg = route['legs'][leg_index]
                        epoch = now
                        controller = RoutedRgbServoBottle(motor, leg['destination'], layout['table_z'], args.mode, routing, leg['side'])
                        monitor = DinnerPhysicalMonitor(model, data, with_bottle_destination(layout, leg['destination']), 'bottle', leg['side'])
                    if controller:
                        controller.accept_observation(estimate, now-epoch)
                        if estimate['status'] == 'observed' and controller.stage in ('approach', 'descend'):
                            try:
                                live = controller.preview_image_action(controller.current_rgb_estimate, now-epoch, data.qpos[:12].copy())
                                frozen = controller.preview_image_action(controller.initial_observation, now-epoch, data.qpos[:12].copy())
                                counterfactuals.append({'time_s': now, 'leg': leg_index, 'stage': controller.stage,
                                    'measured_joint_positions': data.qpos[:12].tolist(), 'live_action': live.tolist(),
                                    'initial_image_action': frozen.tolist(), 'maximum_delta_rad': float(np.max(np.abs(live-frozen)))})
                            except ValueError as exc:
                                counterfactuals.append({'time_s': now, 'leg': leg_index, 'refused': str(exc)})
                if controller:
                    target = controller.update(now-epoch, data.qpos[:12].copy(), data.qvel[:12].copy())
                    stage = f'leg-{leg_index+1}:'+controller.stage
            if not np.array_equal(q, data.qpos) or not np.array_equal(v, data.qvel):
                raise ValueError('A controller wrote authoritative state.')
            if model.neq or np.any(data.xfrc_applied) or np.any(data.qfrc_applied):
                raise ValueError('Hidden forces or equality constraints.')
            failure = None
            if state['valid']:
                maximum_other = max(maximum_other, max((float(np.linalg.norm(data.body(name).xpos-position)) for name, position in initial_others.items()), default=0.))
                failure = monitor.update() if monitor else None
                if maximum_other > .004:
                    failure = 'Cumulative non-target displacement exceeded 4 mm.'
            terminal = not state['valid'] or bool(failure) or (task is not None and not task.active)
            terminal = terminal or bool(controller and (monitor.succeeded or controller.status != 'running'))
            if tick % 10 == 0 or terminal:
                if tick % 2000 == 0:
                    space(p, folder, 24*1024**2)
                for key, value in zip(trace, (q, v, now, stage, target.copy())):
                    trace[key].append(value)
            if not state['valid']:
                break
            if failure:
                report['message'] = failure
                break
            if task is not None:
                if not task.active:
                    report.update(status=task.status, message=task.message)
                    break
                data.ctrl[:] = task.apply_gripper_limit(target)
            else:
                if monitor and monitor.succeeded:
                    completed.append({'leg': leg_index, 'side': controller.side, 'destination': route['legs'][leg_index]['destination'],
                        'physical': monitor.report(), 'route_details': controller.route_details, 'stages': controller.history,
                        'accepted_rgb': controller.observations, 'rgb_refusals': controller.refusals})
                    leg_index += 1
                    if leg_index == len(route['legs']):
                        report.update(status='succeeded', message='Every leg physically released and parked.')
                        break
                    controller = monitor = None
                    target = data.qpos[:12].copy()
                    waiting_since = now
                elif controller and controller.status != 'running':
                    report['message'] = controller.message
                    break
                elif route is None and now > 1.:
                    report['message'] = 'No confident initial RGB estimate.'
                    break
                elif route and controller is None and now-waiting_since > 2.:
                    report['message'] = 'No fresh RGB observation for the next relay leg.'
                    break
                data.ctrl[:] = apply_targets(model, data, target, .25, 6 if controller and controller.side == 'right' else 0)
            if time.perf_counter()-began > spec['maximum_wall_seconds']:
                report['message'] = 'Declared physical wall-time limit.'
                break
            mujoco.mj_step(model, data)
        # Retain the actual terminal state, including an early route refusal.
        if not trace['time'] or trace['time'][-1] != float(data.time):
            for key, value in zip(trace, (data.qpos.copy(), data.qvel.copy(), float(data.time), stage, target.copy())):
                trace[key].append(value)
        report.update(simulation_seconds=float(data.time), route=route, completed_legs=completed,
            physical=task.monitor.report() if task is not None else monitor.report() if monitor else None,
            stage=stage, cumulative_non_target_displacement_m=maximum_other,
            controller_status=controller.status if controller else None,
            controller_stages=controller.history if controller else None)
        if task is not None:
            report['task'] = task.snapshot()
    except Exception as exc:
        report.update(status='harness_error', error_type=type(exc).__name__, message=str(exc))
    finally:
        if task is not None:
            task.close()
        if renderer is not None:
            renderer.close()
        report.update(wall_seconds=time.perf_counter()-began, image_pipeline_seconds=image_seconds, trace_frames=len(trace['time']))
        space(p, folder, 24*1024**2)
        arrays(folder/'states.npz', **{key: np.asarray(value) for key, value in trace.items()})
        write(folder/'observations.json', observations)
        write(folder/'image-counterfactuals.json', counterfactuals)
        write(folder/'report.json', report)
    print({key: report.get(key) for key in ('seed', 'mode', 'status', 'message', 'simulation_seconds', 'wall_seconds')}, flush=True)
    return int(report['status'] == 'harness_error')


def batch(args):
    p = checked_protocol(args.protocol)
    spec = checked_integration(p, args.protocol)
    raw = ROOT/p['raw_root']/'physical'
    if args.action == 'evaluation':
        gate = read(raw/'development/gate.json')
        if not gate['passed'] or gate['integration_sha256'] != sha(integration_path(p)):
            raise ValueError('The complete physical development gate must pass.')
        if (raw/'evaluation-freeze.json').exists():
            raise FileExistsError('Preserve the original final-physical freeze.')
    splits = ['development'] if args.action == 'development' else ['evaluation', 'regression']
    for split in splits:
        if (raw/split).exists():
            raise FileExistsError('Preserve preceding physical batches.')
    space(p, raw, 512*1024**2)
    if args.action == 'evaluation':
        write(raw/'evaluation-freeze.json', {'protocol_sha256': sha(args.protocol),
            'integration_sha256': sha(integration_path(p)), 'development_gate_sha256': sha(raw/'development/gate.json'),
            'selection': 'Unchanged passing integration; no fitting, rerouting or threshold changes after development.',
            'evaluation_seeds': p['physical']['evaluation_seeds'], 'regression_seeds': spec['regression_seeds']})
    def launch(item):
        split, seed, mode = item
        folder = raw/split/f'{seed}-{mode}'
        log = raw/split/f'{seed}-{mode}.log'
        if folder.exists() or log.exists():
            raise FileExistsError('Preserve every physical folder and console log.')
        space(p, log, 24*1024**2)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('xb') as stream:
            outcome = subprocess.run([sys.executable, '-I', str(Path(__file__)), '--protocol', str(args.protocol.resolve()),
                '--action', 'trial', '--split', split, '--seed', str(seed), '--mode', mode],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=spec['maximum_wall_seconds']+90)
        report_path = folder/'report.json'
        result = read(report_path) if report_path.exists() else {'status': 'harness_error', 'message': 'No terminal report.'}
        row = {'split': split, 'seed': seed, 'mode': mode, 'exit_code': outcome.returncode,
            'status': result['status'], 'message': result['message'], 'report_sha256': sha(report_path) if report_path.exists() else None}
        print(row, flush=True)
        return row
    rows = []
    for split in splits:
        seeds = spec['regression_seeds'] if split == 'regression' else p['physical'][split+'_seeds']
        modes = ['live'] if split == 'regression' else spec[split+'_modes']
        with ThreadPoolExecutor(max_workers=spec['maximum_workers']) as workers:
            rows += list(workers.map(launch, [(split, seed, mode) for seed in seeds for mode in modes]))
    counts = {mode: sum(row['status'] == 'succeeded' and row['mode'] == mode and row['split'] == args.action for row in rows)
        for mode in spec[args.action+'_modes']}
    errors = sum(row['status'] == 'harness_error' or row['exit_code'] != 0 for row in rows)
    if args.action == 'development':
        passed = len(rows) == 12 and not errors and counts['live'] >= 4 and counts['live'] > counts['baseline']
    else:
        regressions = [row for row in rows if row['split'] == 'regression']
        passed = len(rows) == 78 and not errors and counts['live'] >= 18 and counts['live'] >= counts['baseline']+6 and len(regressions) == 6 and all(row['status'] == 'succeeded' for row in regressions)
    write(raw/args.action/'gate.json', {'schema': spec['schema'], 'integration_sha256': sha(integration_path(p)),
        'passed': bool(passed), 'counts': counts, 'harness_errors': errors, 'rows': rows,
        'denominator_note': 'Invalid starts and refusals remain in all planned denominators. Live/frozen comparison is separate from improvement over the preset controller.'})
    print({'stage': args.action, 'passed': bool(passed), 'successes': counts}, flush=True)
    return int(not passed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--action', choices=['declare', 'development', 'evaluation', 'trial'], required=True)
    parser.add_argument('--split', choices=['development', 'evaluation', 'regression'])
    parser.add_argument('--seed', type=int)
    parser.add_argument('--mode', choices=['live', 'frozen', 'baseline'])
    args = parser.parse_args()
    if args.action == 'declare':
        declare(args.protocol)
    else:
        raise SystemExit(trial(args) if args.action == 'trial' else batch(args))
