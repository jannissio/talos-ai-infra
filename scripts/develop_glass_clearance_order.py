"""Declared V19 glass-first intervention before the known mug swept-path failure.

At most two independent glass alternatives, each from exact original motor
prefix replay. A passing glass branch continues without reset into at most one
mug attempt. Failed alternatives remain separate; no live object pose writes.
"""
import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['OPENBLAS_NUM_THREADS'] = '1'; os.environ['OMP_NUM_THREADS'] = '1'
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from scripts.develop_glass_depth_two import prepare, freeze, put, arrays
from simulation_lab.storage import require_space

DEADLINE = datetime(2026, 9, 14, 15, 45, tzinfo=timezone.utc)


def remaining_time():
    if datetime.now(timezone.utc) >= DEADLINE:
        raise RuntimeError('Declared15:45UTC endpoint reached')


def swept_bounds(model):
    from simulation_lab.random_dinner import body_bounds
    path = ROOT / '.run/mug-wall-clearance-physics-v3/continuous-states.npz'
    adr = int(model.joint('mug_free').qposadr[0])
    rows = []
    with np.load(path) as archive:
        indices = np.flatnonzero(archive['stage'] == 'align')
        for index in indices[::10]:
            pose = archive['qpos'][index, adr:adr + 7]
            lo, hi = body_bounds(model, 'mug', pose[3:])
            rows.append((pose[:3] + lo, pose[:3] + hi))
    return rows


def clears_sweep(model, position, quaternion, sweep):
    from simulation_lab.random_dinner import body_bounds
    lo, hi = body_bounds(model, 'glass', quaternion)
    lo += position; hi += position
    return not any(np.all(hi + .006 > a) and np.all(b + .006 > lo) for a, b in sweep)


def geometry(args):
    out = args.output.resolve(); log = prepare(out)
    with log.open('x') as stream, contextlib.redirect_stdout(stream):
        teacher, whole = freeze(out)
        put(out / 'source/scripts/develop_glass_clearance_order.py', Path(__file__).read_bytes())
        source = ROOT / '.run/whole-table-development-v19-seed4002'
        accepted = json.loads((source / 'accepted-action-00.json').read_text())
        prefix_path = source / accepted['path'] / 'states.npz'
        baseline = ROOT / '.run/mug-wall-approach-geometry-v1'
        for name in ('scene.xml', 'layout.json', 'actual-source-state.npz'):
            put(out / name, (baseline / name).read_bytes())
        protocol = dict(scope=__doc__, declared_before_new_physics=True, deadline_utc='2026-09-14 15:45:00 UTC',
                        source='V19 seed2026114002 after original accepted mug-buffer action00',
                        prefix_path=prefix_path.relative_to(ROOT).as_posix(), prefix_sha256=hashlib.sha256(prefix_path.read_bytes()).hexdigest(),
                        glass_physical_budget=2, mug_physical_budget=1,
                        alternatives='Common final glass setting first, then common temporary buffers that clear the recorded actual mug swept AABBs by6mm.',
                        failed_branches='Each failed glass alternative retains its full initial-reset+exact motor-prefix+new controls. Only a successful branch may continue to mug, without any reset.',
                        hold_gate_s=1.8, manipulated_external_penetration_m=.001, unexpected_robot_penetration_m=.0008,
                        unrelated_motion_m=.004, ordinary_physics=True, camera_control=False, learned_ordering=False)
        put(out / 'protocol.json', protocol)
        model = mujoco.MjModel.from_xml_string((out / 'scene.xml').read_text()); layout = json.loads((out / 'layout.json').read_text())
        data = mujoco.MjData(model)
        with np.load(out / 'actual-source-state.npz') as archive:
            data.qpos[:] = archive['qpos']; data.qvel[:] = archive['qvel']; data.ctrl[:] = archive['ctrl']
        mujoco.mj_forward(model, data)
        prior = json.loads((source / 'report.json').read_text())
        excluded = sorted({(r['arm'], r['candidate'], r['initial_wrist_roll']) for r in prior['physical_lookahead']
                           if r['item'] == 'glass' and r['action_index'] == 1 and r['metrics']['hold_verified_s'] < 1.8})
        put(out / 'prior-failed-grasp-exclusions.json', dict(source=source.relative_to(ROOT).as_posix(),
              meaning='Reuse exposed same-state source failures before spending the new budget; placement failures after verified hold are not excluded.', excluded=excluded))
        sweep = swept_bounds(model)
        put(out / 'recorded-mug-swept-bounds.json', [[a.tolist(), b.tolist()] for a, b in sweep])
        options = whole.candidates(model, data, layout, ['glass'])
        # Ensure unchanged final setting is explicitly considered even when the
        # common blocking prefilter removed it from its direct option list.
        if not any(r[2] == 'final_setting' for r in options):
            options.insert(0, ('glass', teacher.destination(layout, 'glass'), 'final_setting', None))
        cache, rows, proofs = {}, [], []
        from simulation_lab.scene import HOME
        for index, (_, target, role, quaternion) in enumerate(options):
            remaining_time()
            quat = np.array([1., 0., 0., 0.]) if quaternion is None else quaternion
            row = dict(index=index, target=target.tolist(), quaternion=quat.tolist(), role=role,
                       clears_recorded_sweep=clears_sweep(model, target, quat, sweep), planned=False)
            if not row['clears_recorded_sweep']:
                row['rejection'] = 'Target glass bounds still overlap recorded mug swept path+6mm.'
            else:
                task = teacher.TableTeacher(model, data, layout)
                task.start(object_id='glass', target=target, target_quaternion=quaternion,
                           excluded_grasps=excluded, source_pose_cache=cache, allow_measured_buffer=False)
                task.update(np.asarray(HOME * 2))
                row.update(planned=task.active, message=task.message, search=task.search_log)
                if task.active:
                    row.update(arm=task.side, candidate=task.chosen.name, roll=task.chosen_roll, excluded=excluded)
                    proofs.append(row); put(out / f'proof-{len(proofs):03d}.json', row)
            rows.append(row); put(out / f'check-{index:03d}.json', row)
            print({k: v for k, v in row.items() if k not in ('search', 'excluded')}, flush=True)
            if len(proofs) == 2:
                break
        put(out / 'report.json', dict(protocol=protocol, rows=rows, proofs=proofs,
                                      unexamined_common_options=len(options) - len(rows), physics_steps=0))


def physics(args):
    out = args.output.resolve(); log = prepare(out)
    with log.open('x') as stream, contextlib.redirect_stdout(stream):
        teacher, whole = freeze(out, args.geometry)
        put(out / 'source/scripts/develop_glass_clearance_order.py', Path(__file__).read_bytes())
        for name in ('scene.xml', 'layout.json', 'protocol.json'):
            put(out / name, (args.geometry / name).read_bytes())
        geo = json.loads((args.geometry / 'report.json').read_text()); protocol = geo['protocol']
        model = mujoco.MjModel.from_xml_string((out / 'scene.xml').read_text()); layout = json.loads((out / 'layout.json').read_text())
        prefix_path = ROOT / protocol['prefix_path']
        assert hashlib.sha256(prefix_path.read_bytes()).hexdigest() == protocol['prefix_sha256']
        with np.load(prefix_path) as archive:
            prefix = {k: archive[k].copy() for k in ('initial_integration', 'qpos', 'qvel', 'ctrl', 'stage')}
        from simulation_lab.scene import HOME
        from simulation_lab.table_observation import TableObserver
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        results = []
        extra_excluded = []
        # These alternatives replay an identical source state in the same
        # compiled model. Retain solved q values so the common teacher can
        # recognize when a new initial roll converges to a failed branch.
        # This prospective fix was not used by preserved physical-v1.
        source_pose_cache = {}
        alternatives = (geo['proofs'] + geo['proofs'][-1:])[:2]
        put(out / 'physical-selection-protocol.json', dict(
            alternatives='Try up to two checked target/grasp plans. If only one target has geometry, replan its second branch with the first failed wrist/grasp excluded.',
            final_setting_check='.run/glass-clearance-final-geometry-v1/report.json',
            branch_continuity='Each independent alternative replays the original accepted prefix; a passing branch continues without reset.'))
        for attempt, proof in enumerate(alternatives):
            remaining_time()
            branch = out / f'branch-{attempt + 1:02d}'
            if branch.exists() or branch.with_suffix('.log').exists():
                raise FileExistsError('Preserve branch output/log')
            require_space(branch, 700 * 1024**2); require_space(branch.with_suffix('.log'), 16 * 1024**2)
            branch.mkdir()
            with branch.with_suffix('.log').open('x') as branch_log, contextlib.redirect_stdout(branch_log):
                data = mujoco.MjData(model); mujoco.mj_setState(model, data, prefix['initial_integration'], spec); mujoco.mj_forward(model, data)
                for index, ctrl in enumerate(prefix['ctrl']):
                    data.ctrl[:] = ctrl; mujoco.mj_step(model, data)
                    assert np.array_equal(data.qpos, prefix['qpos'][index + 1]) and np.array_equal(data.qvel, prefix['qvel'][index + 1])
                arrays(branch / 'prefix-states.npz', **prefix)
                task = teacher.TableTeacher(model, data, layout)
                task.start(object_id='glass', target=proof['target'], target_quaternion=proof['quaternion'],
                           excluded_grasps=proof['excluded'] + extra_excluded,
                           source_pose_cache=source_pose_cache, allow_measured_buffer=False)
                targets = np.asarray(HOME * 2); task.update(targets)
                if not task.active:
                    row = dict(status=task.status, message=task.message, search=task.search_log, new_physics_steps=0)
                    put(branch / 'planning-rejection.json', row); results.append(row)
                    continue
                observer = TableObserver(model)
                try:
                    remaining_time(); result = whole.execute(model, data, layout, task, targets, branch / 'glass-clearance', observer)
                finally:
                    observer.close()
                if result['status'] != 'succeeded':
                    extra_excluded.append((task.side, task.chosen.name, task.chosen_roll))
                with np.load(branch / 'glass-clearance/states.npz') as archive:
                    controls = np.concatenate([prefix['ctrl'], archive['ctrl']]); positions = np.concatenate([prefix['qpos'], archive['qpos'][1:]])
                    velocities = np.concatenate([prefix['qvel'], archive['qvel'][1:]]); stages = np.concatenate([prefix['stage'], archive['stage'][1:]])
                arrays(branch / 'continuous-states.npz', initial_integration=prefix['initial_integration'], ctrl=controls,
                       qpos=positions, qvel=velocities, stage=stages)
                replay = mujoco.MjData(model); mujoco.mj_setState(model, replay, prefix['initial_integration'], spec); mujoco.mj_forward(model, replay)
                error = 0.
                for index, ctrl in enumerate(controls):
                    replay.ctrl[:] = ctrl; mujoco.mj_step(model, replay)
                    error = max(error, float(np.max(np.abs(replay.qpos - positions[index + 1]))), float(np.max(np.abs(replay.qvel - velocities[index + 1]))))
                row = dict(attempt=attempt + 1, result=result, continuous_frames=len(positions),
                           independent_continuous_replay_exact=error == 0., replay_max_error=error,
                           proof=proof, source_grasp=task.chosen.name, wrist_roll=task.chosen_roll)
                put(branch / 'report.json', row); results.append(row)
                if result['status'] == 'succeeded':
                    # Preserve the actual full motor state for same-branch
                    # continuation; no subsequent object/robot pose injection.
                    state = np.empty(mujoco.mj_stateSize(model, spec)); mujoco.mj_getState(model, data, state, spec)
                    arrays(branch / 'actual-cleared-integration.npz', integration=state)
                    put(out / 'selected-glass-branch.json', dict(branch=branch.relative_to(out).as_posix(),
                        meaning='Physical glass clearance passed; mug continuation must replay this whole motor prefix, not restore the hypothetical buffer pose.'))
                    break
            print(dict(attempt=attempt + 1, result=results[-1]), flush=True)
        put(out / 'report.json', dict(protocol=protocol, glass_results=results,
                                      glass_cleared=any(r.get('result', {}).get('status') == 'succeeded' for r in results),
                                      mug_attempted=False, full_table_success=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--geometry', type=Path); args = parser.parse_args()
    physics(args) if args.geometry else geometry(args)
