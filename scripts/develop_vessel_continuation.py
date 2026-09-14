"""Bounded alternate-IK continuation after exact accepted joint-scene actions."""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np

from scripts.develop_whole_table import execute, put
from simulation_lab.horizontal_glass_grasp_candidates import horizontal_glass_ik_seeds
from simulation_lab.scene import build_scene
from simulation_lab.storage import require_space
from simulation_lab.table_observation import TableObserver
from simulation_lab.table_teacher import TableTeacher


class AlternateVesselTeacher(TableTeacher):
    def _configure(self, candidate):
        super()._configure(candidate)
        if candidate.name.startswith(('outside_body_band_', 'body_band_', 'handle_axis_')):
            self.ik.fallback_initials = horizontal_glass_ik_seeds()


def run(args):
    out = args.output
    log = out.with_suffix('.log')
    if out.exists() or log.exists():
        raise FileExistsError('Preserve output and log.')
    preflight = require_space(out, 1024**3)
    out.mkdir(parents=True)
    with log.open('x', encoding='utf8') as stream, contextlib.redirect_stdout(stream):
        started = time.perf_counter()
        manifest = {}
        for name in ('scripts/develop_vessel_continuation.py', 'scripts/develop_whole_table.py',
                     'simulation_lab/table_teacher.py', 'simulation_lab/table_buffers.py',
                     'simulation_lab/autonomy.py', 'simulation_lab/dinner_autonomy.py',
                     'simulation_lab/scene.py', 'simulation_lab/dinner.py',
                     'simulation_lab/horizontal_glass_grasp_candidates.py',
                     'simulation_lab/glass_grasp_candidates.py', 'simulation_lab/side_plate_grasp_candidates.py'):
            payload = (ROOT/name).read_bytes()
            put(out/'source'/name, payload)
            manifest[name] = hashlib.sha256(payload).hexdigest()
        put(out/'source-manifest.json', manifest)
        recipe = json.loads((args.workflow/'reset-recipe.json').read_text())
        xml, layout = build_scene(seed=recipe['seed'], scenario='dinner', dinner_preset='task')
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        prefix = []
        for index in range(args.prefix_actions):
            source = json.loads((args.workflow/f'accepted-action-{index:02d}.json').read_text())
            if not source['continuous_main_scene_replay_exact']:
                raise ValueError('An exact accepted source action is required.')
            folder = args.workflow/source['path']
            with np.load(folder/'states.npz') as archive:
                saved = {k: archive[k] for k in ('initial_integration', 'ctrl', 'qpos', 'qvel')}
            if index == 0:
                mujoco.mj_setState(model, data, saved['initial_integration'], mujoco.mjtState.mjSTATE_INTEGRATION)
                mujoco.mj_forward(model, data)
            assert np.array_equal(data.qpos, saved['qpos'][0]) and np.array_equal(data.qvel, saved['qvel'][0])
            for i, ctrl in enumerate(saved['ctrl']):
                data.ctrl[:] = ctrl
                mujoco.mj_step(model, data)
                assert np.array_equal(data.qpos, saved['qpos'][i+1]) and np.array_equal(data.qvel, saved['qvel'][i+1])
            prefix.append({'source': folder.as_posix(), 'item': source['item'], 'role': source['role'],
                           'states_sha256': hashlib.sha256((folder/'states.npz').read_bytes()).hexdigest(),
                           'motor_steps': len(saved['ctrl']), 'exact': True})
        initial = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(model, data, initial, mujoco.mjtState.mjSTATE_INTEGRATION)
        report = {'scope': 'Exposed privileged vessel continuation; no learned control or complete table claim.',
                  'preflight': preflight, 'item': args.item, 'prefix': prefix,
                  'prefix_motor_steps': sum(p['motor_steps'] for p in prefix), 'attempts': [],
                  'whole_table_complete': False}
        put(out/'prefix.json', prefix)
        observer = TableObserver(model)
        excluded = []
        cache = {}
        try:
            for attempt in range(args.attempts):
                trial = mujoco.MjData(model)
                mujoco.mj_setState(model, trial, initial, mujoco.mjtState.mjSTATE_INTEGRATION)
                mujoco.mj_forward(model, trial)
                task = AlternateVesselTeacher(model, trial, layout)
                task.start(object_id=args.item, excluded_grasps=excluded, source_pose_cache=cache)
                targets = trial.ctrl.copy()
                task.update(targets)
                if not task.active:
                    row = {'status': task.status, 'message': task.message, 'search': task.search_log,
                           'new_motor_steps': 0}
                    put(out/f'planning-{attempt:02d}.json', row)
                    report['attempts'].append(row)
                    break
                row = execute(model, trial, layout, task, targets, out/f'continuation-{attempt:02d}', observer)
                report['attempts'].append(row)
                if row['demonstration_eligible']:
                    break
                excluded.append((task.side, task.chosen.name, task.chosen_roll))
        finally:
            observer.close()
            report['wall_s'] = time.perf_counter()-started
            put(out/'report.json', report)
            print({'attempts': [{k: r.get(k) for k in ('status', 'message', 'frames', 'replay_exact')} for r in report['attempts']],
                   'prefix_motor_steps': report['prefix_motor_steps'], 'wall_s': report['wall_s']}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workflow', type=Path, required=True)
    parser.add_argument('--prefix-actions', type=int, required=True)
    parser.add_argument('--item', choices=('bottle', 'mug', 'glass'), required=True)
    parser.add_argument('--attempts', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
