"""Exposed drawer failures with a declared plate-clearance reset intervention."""
import argparse
from copy import deepcopy
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
import torch
from scripts import collect_drawer_configurations as collector
from scripts.measure_manipulation_coverage import read, sha, put
from simulation_lab.storage import require_space


def declare(path):
    original = ROOT/'docs/robotics/experiments/drawer-demonstrations-v1.json'
    old = read(original)
    evidence = ROOT/'training/drawer_configurations_v1'
    gate_path = ROOT/'docs/robotics/evidence/drawer-demonstrations-v1/gate.json'
    gate = read(gate_path)
    assert not gate['passed'] and gate['eligible'] == {'training': 31, 'development': 8}
    anchor = next(r for r in gate['rows'] if r['training_eligible'])
    rows = [anchor]+[r for r in gate['rows'] if not r['training_eligible']]
    cases = []
    names = list(old['source_sha256'])+[original.relative_to(ROOT).as_posix(),
        gate_path.relative_to(ROOT).as_posix(), 'scripts/diagnose_drawer_clearance.py']
    for row in rows:
        folder = evidence/row['split']/str(row['seed'])
        r = read(folder/'report.json')
        failure = r['teacher_physical'].get('failure')
        group = 'anchor' if row['training_eligible'] else 'plate_disturbance' if failure == 'Another object was disturbed.' else 'planning'
        cases.append({'seed': row['seed'], 'split': row['split'], 'group': group,
                      'original_status': row['status'], 'original_failure': failure})
        names += [(folder/name).relative_to(ROOT).as_posix() for name in ('report.json', 'teacher-states.npz')]
    assert len(cases) == 42 and sum(c['group'] == 'plate_disturbance' for c in cases) == 29
    p = deepcopy(old)
    p.update(schema='talos.drawer-clearance-diagnostic.v1', raw_root='.run/drawer-clearance-diagnostic-v1',
        cases=cases, original_inputs=evidence.relative_to(ROOT).as_posix(),
        source_sha256={name: sha(ROOT/name) for name in names},
        clearance_plate_xy_m=[-.035, -.155], clearance_settle_steps=300,
        diagnostic_gate={'minimum_recovered_plate_failures': 24, 'require_anchor': True,
                         'maximum_invalid_starts': 0, 'maximum_harness_errors': 0},
        change='After reproducing each original initial qpos/qvel exactly, move only the plate at RESET to [-0.035,-0.155], zero only its six free-joint velocities and settle 300 steps. Keep the cabinet configuration, teacher, paths, torque cap, physical guards and saved-action replay unchanged.',
        scope='One exposed passing anchor and all 41 failed prior configurations. This diagnoses the plate obstruction under a declared clear-access scene; it is not a learned skill, autonomous plate relocation, fresh generalization or a repair of the stopped V1 input gate.',
        next_gate='Retain all 42 outcomes; require anchor replay, no invalid/harness cases and at least 24/29 plate-related failures recovered before collecting separately declared fresh clear-access demonstrations. Planning failures remain in the total denominator. No fitting or promotion in this diagnostic.')
    p.pop('input_gate')
    if path.exists() or (ROOT/p['raw_root']).exists():
        raise FileExistsError('Preserve every drawer-clearance declaration and output.')
    p['preflight'] = require_space(ROOT/p['raw_root'], p['storage_budget_bytes'])
    put(path, p)
    print({'cases': len(cases), 'protocol_sha256': sha(path)}, flush=True)


def run(path):
    p = read(path); raw = ROOT/p['raw_root']
    if raw.exists():
        raise FileExistsError('Preserve the previous clearance diagnostic.')
    for name, digest in p['source_sha256'].items():
        assert sha(ROOT/name) == digest, name
    require_space(raw, p['storage_budget_bytes']); raw.mkdir(parents=True)
    put(raw/'protocol.json', p)
    for name in p['source_sha256']:
        if name.endswith('.py'):
            target = raw/'frozen-source'/name
            require_space(target, (ROOT/name).stat().st_size+1024)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:stream.write((ROOT/name).read_bytes())
    original_setup = collector.setup
    active_case = {}

    def cleared_setup(parent, case, folder):
        model, data, layout, valid = original_setup(parent, case, folder)
        reference = ROOT/p['original_inputs']/active_case['split']/str(active_case['seed'])
        with np.load(reference/'teacher-states.npz', allow_pickle=False) as z:
            assert np.array_equal(data.qpos, z['qpos'][0]) and np.array_equal(data.qvel, z['qvel'][0])
        before_q, before_v = data.qpos.copy(), data.qvel.copy()
        before = np.empty(mujoco.mj_stateSize(model, collector.STATE))
        mujoco.mj_getState(model, data, before, collector.STATE)
        collector.arrays(folder/'original-initial-state.npz', integration=before)
        address = int(model.joint('plate_free').qposadr[0])
        dof = int(model.joint('plate_free').dofadr[0])
        data.qpos[address:address+2] = p['clearance_plate_xy_m']
        data.qvel[dof:dof+6] = 0.
        mask_q, mask_v = np.ones(model.nq, bool), np.ones(model.nv, bool)
        mask_q[address:address+2] = False; mask_v[dof:dof+6] = False
        assert np.array_equal(data.qpos[mask_q], before_q[mask_q])
        assert np.array_equal(data.qvel[mask_v], before_v[mask_v])
        mujoco.mj_forward(model, data)
        penetration = max((-float(c.dist) for c in data.contact
            if model.geom_bodyid[c.geom1] != model.geom_bodyid[c.geom2]), default=0.)
        for _ in range(p['clearance_settle_steps']):mujoco.mj_step(model, data)
        data.time = 0.; mujoco.mj_forward(model, data)
        valid = deepcopy(valid)
        speed = float(abs(data.joint('drawer_slide').qvel[0]))
        valid.update(valid=bool(valid['valid'] and penetration <= .001 and speed < .005),
            clearance_penetration_m=penetration, original_initial_qpos_qvel_identical=True,
            reset_only_plate_clearance_xy_m=p['clearance_plate_xy_m'])
        if not valid['valid']:valid['reason'] = 'Invalid original or clearance reset.'
        put(folder/'reset-intervention.json', {'original_case': active_case,
            'original_report_sha256': sha(reference/'report.json'),
            'original_initial_qpos_qvel_identical': True,
            'only_plate_xy_and_its_velocity_changed_before_settling': True,
            'plate_before_xy_m': before_q[address:address+2].tolist(),
            'plate_after_xy_m': data.qpos[address:address+2].tolist(),
            'declared_xy_m': p['clearance_plate_xy_m'], 'settle_steps': p['clearance_settle_steps'],
            'initial_penetration_m': penetration,
            'scope': 'Reset intervention, not an autonomous robot action.'})
        return model, data, layout, valid

    torch.set_num_threads(1)
    rows = []
    with patch.object(collector, 'setup', cleared_setup):
        for case in p['cases']:
            active_case = case
            row = collector.episode(p, path, case['split'], case['seed'])
            row['group'] = case['group']; rows.append(row)
            if row['status'] == 'harness_error' or (case['group'] == 'anchor' and not row['training_eligible']):
                put(raw/'stopped.json', {'reason': 'Stop on harness error or failed anchor; preserve attempted outcomes.', 'rows': rows})
                return 1
    recovered = sum(r['group'] == 'plate_disturbance' and r['training_eligible'] for r in rows)
    invalid = sum(r['status'] == 'invalid_start' for r in rows)
    passed = len(rows) == len(p['cases']) and recovered >= p['diagnostic_gate']['minimum_recovered_plate_failures'] and invalid == 0
    put(raw/'gate.json', {'schema': p['schema'], 'protocol_sha256': sha(path), 'passed': passed,
        'all_42_outcomes_retained': len(rows) == 42, 'recovered_plate_failures': recovered,
        'planning_recoveries': sum(r['group'] == 'planning' and r['training_eligible'] for r in rows),
        'invalid_starts': invalid, 'rows': rows, 'fit_performed': False, 'promotion': False})
    print({'diagnostic_gate_passed': passed, 'recovered_plate_failures': recovered, 'invalid_starts': invalid}, flush=True)
    return int(not passed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declare', type=Path); parser.add_argument('--protocol', type=Path)
    args = parser.parse_args()
    if args.declare:declare(args.declare)
    elif args.protocol:raise SystemExit(run(args.protocol))
    else:parser.error('Declare or run a protocol.')
