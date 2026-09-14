"""Exposed contact-only drawer feasibility diagnosis; never a learned policy.

The previous 15 drawer starts are retained verbatim. A cabinet-aligned teacher
pulls only the remaining stroke. No-op cases are reported separately from
physical openings. Nothing in this diagnostic changes the selected runtime.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
import torch
from scripts.measure_manipulation_coverage import setup, read, sha, put
from simulation_lab.autonomy import OPEN, PlanningError
from simulation_lab.dinner_autonomy import DrawerTask
from simulation_lab.dinner_monitor import DinnerPhysicalMonitor
from simulation_lab.scene import HOME
from simulation_lab.storage import require_space


class ConfigurationDrawerTeacher(DrawerTask):
    """Privileged demonstration teacher with unchanged contact/path guards."""

    def _plan(self, targets):
        if np.max(np.abs(self.data.qpos[:12]-np.array(HOME*2))) > .10 or np.max(np.abs(self.data.qvel[:12])) > .12:
            raise PlanningError('Start with both arms parked.')
        opening = float(self.data.joint('drawer_slide').qpos[0])
        if not -.002 <= opening < .105:
            raise PlanningError('Physical-opening teacher requires a not-yet-open drawer.')
        self._select_item('left', {'id': 'drawer', 'body': 'cutlery_drawer'})
        rotation = self.data.body('cutlery_drawer').xmat.reshape(3, 3)
        self.pull_axis = rotation @ self.model.joint('drawer_slide').axis
        self.ik.x_target = rotation @ np.array([0., 1., 0.])
        self.ik.grasp_point = np.array([-.001, 0., -.096])
        self.grasp = self.data.site('drawer_handle_grasp').xpos.copy()
        self.hover = self.grasp + [0., 0., .04]
        q = self.ik.solve(self.grasp, np.array(HOME[:5]))
        above = self.ik.solve(self.hover, q)
        front = self.hover + self.pull_axis*.05
        qfront = self.ik.solve(front, above)
        approach = np.vstack((self._joint_path(self.data.qpos[:5], qfront, OPEN),
                              self._cartesian(front, self.hover, qfront, OPEN)))
        self._cartesian(self.hover, self.grasp, above, OPEN)
        self.attempts = 1
        targets[:] = HOME*2
        self.metrics.update(starting_opening_m=opening, pull_axis_world=self.pull_axis.tolist(),
                            desired_opening_m=.116, observation='exact_simulator_state')
        self._move('approach', approach, OPEN, 3.)

    def _move_point(self, stage, end, grip, duration):
        if stage == 'align':
            remaining = .116-float(self.data.joint('drawer_slide').qpos[0])
            if not 0. <= remaining <= .119:
                raise PlanningError('Remaining drawer stroke is outside the declared range.')
            end = self.ik.point(self.data) + self.pull_axis*remaining
            self.metrics['commanded_remaining_pull_m'] = remaining
        super()._move_point(stage, end, grip, duration)


def declare(path):
    parent_path = ROOT/'docs/robotics/experiments/manipulation-coverage-v1.json'
    parent = read(parent_path)
    names = ['scripts/diagnose_drawer_configuration.py', 'scripts/measure_manipulation_coverage.py']
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT/'simulation_lab').glob('*.py')]
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT/'simulation_lab/assets/so101').rglob('*') if p.is_file()]
    names += [parent_path.relative_to(ROOT).as_posix()]
    names += [(Path(parent['context'])/n).as_posix() for n in parent['context_sha256']]
    evidence = ROOT/'docs/robotics/evidence/manipulation-coverage-v1'
    cases = [c for c in parent['cases'] if c['skill'] == 'drawer']
    for case in cases:
        for mode in ('teacher', 'learned'):
            names += [(evidence/(case['id']+'-'+mode)/n).relative_to(ROOT).as_posix()
                      for n in ('report.json', 'states.npz')]
    p = {'schema': 'talos.drawer-configuration-diagnostic.v1', 'declared_on': '2026-09-14',
         'parent_protocol': parent_path.relative_to(ROOT).as_posix(),
         'parent_evidence': evidence.relative_to(ROOT).as_posix(), 'cases': cases,
         'raw_root': '.run/drawer-configuration-diagnostic-v1',
         'evidence_package': 'docs/robotics/evidence/drawer-configuration-diagnostic-v1',
         'source_sha256': {n: sha(ROOT/n) for n in names},
         'source_git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
         'maximum_simulated_seconds': 85., 'maximum_wall_seconds': 150.,
         'maximum_workers': 1, 'storage_budget_bytes': 512*1024**2, 'reserve_gib': 10,
         'change': 'Same drawer teacher, path/collision checks and torque cap; rotate gripper/front approach with the cabinet and pull only the remaining distance to 0.116 m. Accept initial openings below 0.105 m. No online IK or object state will be presented as learned action.',
         'starts': 'All 15 exposed drawer starts from the original coverage protocol, exactly matched qpos/qvel before control. No success filtering, new fit, fresh evaluation or automatic selected-runtime change.',
         'scoring': 'Independent unchanged DinnerPhysicalMonitor: at least 0.105 m opening, >=0.1 s both-finger handle contact, released fingers, both arms parked for >=0.5 s, no unexpected collision, unrelated objects <=4 mm, inactive arm <=1 degree. Keep all solver/physical failures in the valid-start denominator.',
         'already_open': 'Initially >=0.105 m is a semantic no-op only: leave motor targets at HOME and verify >=0.5 s open/parked without contact or disturbance. Record already_open separately; it is not counted as a physical-opening success.',
         'stop': 'Retain every case. Stop new launches on a harness error. Require the unchanged anchor to pass before proceeding beyond it. Diagnostic results inform a separately declared demonstration/perception/motor experiment; a failed solver is not proof of unreachability.'}
    if path.exists() or (ROOT/p['raw_root']).exists():
        raise FileExistsError('Preserve earlier declarations and outputs.')
    p['preflight'] = require_space(ROOT/p['raw_root'], p['storage_budget_bytes'])
    put(path, p)
    print({'declared_cases': len(cases), 'protocol_sha256': sha(path)}, flush=True)


def run_case(p, protocol_path, case):
    folder = ROOT/p['raw_root']/case['id']
    if folder.exists():
        raise FileExistsError('Preserve every drawer attempt.')
    preflight = require_space(folder, 24*1024**2)
    folder.mkdir()
    trace = {k: [] for k in ('qpos', 'qvel', 'time', 'stage', 'targets')}
    result = {'case': case, 'protocol_sha256': sha(protocol_path), 'preflight': preflight,
              'controller': 'configuration_teacher', 'observation': 'exact_simulator_state',
              'status': 'harness_error'}
    task = monitor = None
    begun = time.perf_counter()
    try:
        model, data, layout, state = setup(read(ROOT/p['parent_protocol']), case, folder)
        with np.load(ROOT/p['parent_evidence']/(case['id']+'-teacher')/'states.npz', allow_pickle=False) as z:
            identical = np.array_equal(data.qpos, z['qpos'][0]) and np.array_equal(data.qvel, z['qvel'][0])
        if not identical:
            raise ValueError('Original paired initial state was not reproduced exactly.')
        state['identical_to_original_qpos_qvel'] = True
        result.update(layout=layout, setup=state)
        targets = data.ctrl.copy()
        initial_open = float(data.joint('drawer_slide').qpos[0])
        noop = initial_open >= .105
        stable_noop = 0.
        monitor = DinnerPhysicalMonitor(model, data, layout, 'drawer', 'left')
        if state['valid'] and not noop:
            task = ConfigurationDrawerTeacher(model, data, layout)
            task.start()
        result['status'] = 'failed' if state['valid'] else 'invalid_start'
        terminal_at = None
        for tick in range(int(p['maximum_simulated_seconds']/model.opt.timestep)+1):
            q, v = data.qpos.copy(), data.qvel.copy()
            stage = 'invalid_start' if not state['valid'] else 'already_open' if noop else task.stage
            if state['valid'] and not noop and task.active:
                task.update(targets)
            if not np.array_equal(q, data.qpos) or not np.array_equal(v, data.qvel):
                raise ValueError('Controller wrote authoritative state.')
            if model.neq or np.any(data.xfrc_applied) or np.any(data.qfrc_applied):
                raise ValueError('Hidden force or constraint.')
            failure = monitor.update() if state['valid'] else None
            if noop and state['valid']:
                okay = (float(data.joint('drawer_slide').qpos[0]) >= .105
                        and monitor.metrics['both_arms_parked'] and not failure
                        and monitor.metrics['verified_handle_contact_s'] == 0.)
                stable_noop = stable_noop + model.opt.timestep if okay else 0.
            terminal = (not state['valid'] or bool(failure) or (noop and stable_noop >= .5)
                        or (not noop and task is not None and task.status == 'failed')
                        or (not noop and task is not None and task.status == 'succeeded' and monitor.succeeded))
            if not noop and task is not None and not task.active:
                terminal_at = float(data.time) if terminal_at is None else terminal_at
                terminal |= float(data.time)-terminal_at >= 1.
            if tick % 10 == 0 or terminal:
                if tick % 2000 == 0:
                    require_space(folder, 24*1024**2)
                for key, value in zip(trace, (q, v, float(data.time), stage, targets.copy())):
                    trace[key].append(value)
            if terminal:
                break
            if time.perf_counter()-begun > p['maximum_wall_seconds']:
                raise TimeoutError('Declared drawer wall-time budget exceeded.')
            data.ctrl[:] = task.apply_gripper_limit(targets) if task is not None else targets
            mujoco.mj_step(model, data)
        if state['valid']:
            if noop:
                result.update(status='already_open' if stable_noop >= .5 else 'failed',
                              noop_stable_seconds=stable_noop, no_op=True,
                              message='Already-open semantic verification; no physical opening claimed.')
            else:
                result.update(status='succeeded' if task.status == 'succeeded' and monitor.succeeded else 'failed',
                              task=task.snapshot(), no_op=False,
                              message=monitor.failure or task.message)
        result.update(monitor=monitor.report(), physics_state_writes_during_control=0,
                      hidden_forces=0, equality_constraints=0)
    except Exception as exc:
        result.update(status='harness_error', error_type=type(exc).__name__, message=str(exc))
        if task is not None:
            result['task'] = task.snapshot()
    finally:
        result.update(wall_seconds=time.perf_counter()-begun, trace_frames=len(trace['time']))
        require_space(folder/'states.npz', 24*1024**2)
        with (folder/'states.npz').open('xb') as stream:
            np.savez_compressed(stream, **{k: np.asarray(v) for k, v in trace.items()})
        put(folder/'report.json', result)
    row = {'case': case['id'], 'status': result['status'], 'message': result.get('message'),
           'valid': result.get('setup', {}).get('valid'), 'report_sha256': sha(folder/'report.json')}
    print(row, flush=True)
    return row


def run(path):
    p = read(path)
    raw = ROOT/p['raw_root']
    if raw.exists():
        raise FileExistsError('Preserve every diagnostic run.')
    for name, digest in p['source_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Declared source or comparison changed: '+name)
    require_space(raw, p['storage_budget_bytes'])
    raw.mkdir(parents=True)
    put(raw/'protocol.json', p)
    for name in p['source_sha256']:
        if name.endswith('.py'):
            target = raw/'frozen-source'/name
            require_space(target, (ROOT/name).stat().st_size+1024)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write((ROOT/name).read_bytes())
    torch.set_num_threads(1)
    rows = []
    for case in p['cases']:
        row = run_case(p, path, case)
        rows.append(row)
        if row['status'] == 'harness_error' or (case['kind'] == 'anchor' and row['status'] != 'succeeded'):
            put(raw/'stopped.json', {'rows': rows, 'reason': 'Harness error or anchor failure; preserve all outcomes.'})
            return 1
    put(raw/'summary.json', {'schema': p['schema'], 'protocol_sha256': sha(path), 'rows': rows,
                            'all_15_cases_retained': len(rows) == 15,
                            'physical_successes': sum(r['status'] == 'succeeded' for r in rows),
                            'already_open': sum(r['status'] == 'already_open' for r in rows),
                            'promotion': False})
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declare', type=Path)
    parser.add_argument('--protocol', type=Path)
    args = parser.parse_args()
    if args.declare:
        declare(args.declare)
    elif args.protocol:
        raise SystemExit(run(args.protocol))
    else:
        parser.error('Use --declare or --protocol.')
