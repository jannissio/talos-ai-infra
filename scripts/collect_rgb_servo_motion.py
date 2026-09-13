"""Collect bounded varied bottle motions and exact saved-action replay gates.

Compact physical states can later be rendered for current-image supervision.
This first batch contains nominal motions; recovery demonstrations are reserved
for the final eight declared training seeds after disturbance calibration.
"""
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from simulation_lab.dinner_autonomy import DinnerTask
from simulation_lab.dinner_monitor import DinnerPhysicalMonitor
from simulation_lab.policy_control import apply_targets
from simulation_lab.scene import build_scene, HOME
from simulation_lab.storage import require_space
from scripts.prepare_bottle_data import STATE, state


def collect(seed, protocol, folder):
    require_space(folder, 32*1024**2); folder.mkdir()
    rng = np.random.default_rng(seed)
    bounds = protocol['workspace_m']
    x, y, yaw = rng.uniform(*bounds['x']), rng.uniform(*bounds['y']), rng.uniform(-.6, .6)
    destination = protocol['destinations_m'][seed % 3]
    xml, layout = build_scene(seed=seed, scenario='dinner', dinner_preset='task')
    layout = deepcopy(layout)
    for target in layout['targets']:
        if target['object_id'] == 'bottle':
            target['position_m'] = [*destination, layout['table_z']]
    model = mujoco.MjModel.from_xml_string(xml); data = mujoco.MjData(model)
    data.qpos[:12] = HOME*2; data.ctrl[:] = HOME*2
    address = int(model.joint('bottle_free').qposadr[0])
    data.qpos[address:address+7] = [x, y, layout['table_z']+.001, math.cos(yaw/2), 0., 0., math.sin(yaw/2)]
    mujoco.mj_forward(model, data)
    body = model.body('bottle').id
    overlap = max([-c.dist for c in data.contact if c.dist < 0 and body in [model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]]], default=0.)
    for _ in range(300):
        mujoco.mj_step(model, data)
    data.time = 0.
    initial = state(model, data)
    np.save(folder/'initial-integration-state.npy', initial, allow_pickle=False)
    scene = ET.fromstring(xml); compiler = scene.find('compiler')
    compiler.set('meshdir', os.path.relpath(compiler.get('meshdir'), folder).replace('\\', '/'))
    (folder/'scene.xml').write_text(ET.tostring(scene, encoding='unicode'))
    report = {'seed': seed, 'split': 'training', 'initial_pose': [x, y, yaw], 'destination_m': destination,
              'layout': layout, 'initial_overlap_m': float(overlap), 'training_eligible': False,
              'recovery_demonstration': False, 'controller': 'exact-state training teacher only',
              'state_writes_during_control': 0, 'hidden_forces': 0, 'replay': None}
    if overlap > .001 or data.body('bottle').xmat[8] < .98:
        report.update(status='invalid_start', message='Initial overlap or tipped bottle; preserve and refuse this start.')
    else:
        task = DinnerTask(model, data, layout); task.start(side='auto', object_id='bottle')
        targets = np.array(HOME*2); actions, stages = [], []
        trace = {k: [] for k in ['qpos', 'qvel', 'time', 'stage']}
        for tick in range(20000):
            q, v = data.qpos.copy(), data.qvel.copy(); task.update(targets)
            assert np.array_equal(q, data.qpos) and np.array_equal(v, data.qvel)
            if task.side:
                task.grip_torque = .25; task.metrics['gripper_torque_limit_nm'] = .25
            if tick % 10 == 0 or not task.active:
                for key, value in [('qpos', data.qpos.copy()), ('qvel', data.qvel.copy()), ('time', float(data.time)), ('stage', task.stage)]:
                    trace[key].append(value)
            if tick % 1000 == 0:
                require_space(folder, 32*1024**2)
            if not task.active:
                break
            actions.append(targets.copy()); stages.append(task.stage)
            data.ctrl[:] = apply_targets(model, data, targets, .25, 0 if task.side != 'right' else 6)
            mujoco.mj_step(model, data)
            assert model.neq == 0 and not np.any(data.xfrc_applied) and not np.any(data.qfrc_applied)
        if task.active:
            task.cancel(targets)
        report.update(status=task.status, message=task.message, arm=task.side, outcome=task.snapshot())
        require_space(folder, 32*1024**2)
        np.savez_compressed(folder/'teacher-states.npz', **{k: np.asarray(v) for k, v in trace.items()})
        if actions:
            raw = np.asarray(actions)
            indices = np.unique(np.r_[np.arange(0, len(raw), 10), len(raw)-1])
            endpoints = np.clip(raw[indices], model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]).astype('float32')
            np.savez_compressed(folder/'trajectory.npz', actions20=endpoints, action_indices=indices, stages=np.asarray(stages)[indices])
            if task.status == 'succeeded':
                replay = mujoco.MjData(model); mujoco.mj_setState(model, replay, initial, STATE); mujoco.mj_forward(model, replay)
                monitor = DinnerPhysicalMonitor(model, replay, layout, 'bottle', task.side)
                replay_trace = {k: [] for k in ['qpos', 'qvel', 'time', 'stage']}
                for tick in range(int(indices[-1])+301):
                    failure = monitor.update()
                    if tick % 10 == 0:
                        for key, value in [('qpos', replay.qpos.copy()), ('qvel', replay.qvel.copy()), ('time', float(replay.time)), ('stage', str(stages[min(tick, len(stages)-1)]))]:
                            replay_trace[key].append(value)
                    if failure or monitor.succeeded:
                        break
                    target = np.asarray([np.interp(tick, indices, endpoints[:, j]) for j in range(12)])
                    replay.ctrl[:] = apply_targets(model, replay, target, .25, 0 if task.side == 'left' else 6)
                    mujoco.mj_step(model, replay)
                monitor.update(); report['replay'] = monitor.report()
                report['training_eligible'] = report['replay']['passed']
                require_space(folder, 32*1024**2)
                np.savez_compressed(folder/'replay-states.npz', **{k: np.asarray(v) for k, v in replay_trace.items()})
    (folder/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    return {k: report.get(k) for k in ['seed', 'status', 'message', 'arm', 'training_eligible', 'initial_pose', 'destination_m', 'replay']}


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 1 <= args.count <= 24:
        raise ValueError('At most the first 24 nominal seeds; reserve the final eight for recovery data.')
    protocol = json.loads(args.protocol.read_text())
    require_space(args.output, args.count*32*1024**2); args.output.mkdir(parents=True)
    rows = []
    for seed in range(protocol['training_seed_range'][0], protocol['training_seed_range'][0]+args.count):
        rows.append(collect(seed, protocol, args.output/f'seed-{seed}'))
        result = {'protocol': args.protocol.as_posix(), 'nominal_seeds_planned': args.count, 'recovery_seeds_reserved': [2026095125, 2026095132],
                  'attempts': len(rows), 'eligible': sum(r['training_eligible'] for r in rows), 'rows': rows,
                  'scope': 'Training teacher and replay results only; not learned execution or a robustness benchmark.'}
        (args.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(rows[-1]), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=Path('docs/robotics/experiments/rgb-servo-bottle-v1.json'))
    parser.add_argument('--count', type=int, default=24)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
