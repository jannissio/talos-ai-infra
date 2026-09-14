"""Measure unchanged baseline table contact; never change its success criterion."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np

from scripts.develop_whole_table import arrays, put
from scripts.evaluate_dinner_scene import load
from simulation_lab.dinner import OBJECTS
from simulation_lab.dinner_autonomy import DinnerSequence
from simulation_lab.scene import HOME
from simulation_lab.storage import require_space


def run(folder):
    if folder.exists():
        raise FileExistsError('Preserve prior contact measurements.')
    preflight = require_space(folder, 128*1024**2)
    folder.mkdir(parents=True)
    sources = {}
    for name in ('scripts/diagnose_table_support.py', 'scripts/evaluate_dinner_scene.py',
                 'simulation_lab/scene.py', 'simulation_lab/dinner.py',
                 'simulation_lab/autonomy.py', 'simulation_lab/dinner_autonomy.py'):
        raw = (ROOT/name).read_bytes()
        put(folder/'source'/name, raw)
        sources[name] = hashlib.sha256(raw).hexdigest()
    m, d, layout = load(42)
    for _ in range(200):
        mujoco.mj_step(m, d)
    d.time = 0.
    initial = np.empty(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(m, d, initial, mujoco.mjtState.mjSTATE_INTEGRATION)
    task = DinnerSequence(m, d, layout)
    task.start(kind='set_table')
    target = np.array(HOME*2)
    table = m.geom('table').id
    names = list(OBJECTS)
    body_columns = {m.body(name).id: i for i, name in enumerate(names)}
    positions, velocities, controls, phases, measurements = [d.qpos.copy()], [d.qvel.copy()], [], [], []
    peaks = {}
    while task.active and len(controls) < 120000:
        before = d.qpos.copy(), d.qvel.copy()
        phase = task.steps[len(task.results)]+'/'+task.child.stage
        task.update(target)
        assert np.array_equal(before[0], d.qpos) and np.array_equal(before[1], d.qvel)
        if not task.active:
            break
        d.ctrl[:] = task.apply_gripper_limit(target)
        assert m.neq == 0 and not np.any(d.xfrc_applied) and not np.any(d.qfrc_applied)
        controls.append(d.ctrl.copy())
        mujoco.mj_step(m, d)
        positions.append(d.qpos.copy()); velocities.append(d.qvel.copy()); phases.append(phase)
        record = np.zeros((len(names), 2))
        for ci, contact in enumerate(d.contact):
            a, b = int(contact.geom1), int(contact.geom2)
            other = b if a == table else a if b == table else None
            if other is None or int(m.geom_bodyid[other]) not in body_columns:
                continue
            column = body_columns[int(m.geom_bodyid[other])]
            record[column, 0] = max(record[column, 0], -float(contact.dist)*1000)
            force = np.empty(6); mujoco.mj_contactForce(m, d, ci, force)
            record[column, 1] += max(0., float(force[0]))
        measurements.append(record)
        active = task.steps[min(len(task.results), len(task.steps)-1)]
        if active in names:
            row = record[names.index(active)]
            old = peaks.setdefault(phase, {'penetration_mm': 0., 'force_n': 0.})
            old['penetration_mm'] = max(old['penetration_mm'], float(row[0]))
            old['force_n'] = max(old['force_n'], float(row[1]))
    arrays(folder/'states.npz', initial_integration=initial, qpos=np.asarray(positions),
           qvel=np.asarray(velocities), ctrl=np.asarray(controls), phase=np.asarray(phases),
           table_contact=np.asarray(measurements))
    result = {'schema': 'talos.table-support-calibration.v1', 'preflight': preflight,
              'scope': 'Unchanged exposed seed-42 physical teacher reference; not whole-table coverage.',
              'source_sha256': sources, 'status': task.status, 'message': task.message,
              'frames': len(positions), 'simulation_s': float(d.time), 'peaks_by_phase': peaks,
              'contact_columns': ['penetration_mm', 'total_normal_force_n'], 'items': names,
              'sequence': task.snapshot(), 'mujoco': mujoco.__version__}
    put(folder/'result.json', result)
    print(json.dumps({'status': task.status, 'frames': len(positions), 'peaks': peaks}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
