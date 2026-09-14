"""Declared coverage diagnosis with paired learned and exact-state controllers.

Scene intervention is confined to reset. A physical teacher pass is evidence of
feasibility; its failure is never a proof of unreachability. No data fitting or
controller selection occurs in this diagnostic grid.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
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
import torch
from simulation_lab.dinner_autonomy import DinnerTask, DrawerTask
from simulation_lab.learned_dinner import LearnedDinnerTask
from simulation_lab.mug_visual_control import VisualMugTask
from simulation_lab.mug_visual_profile import load_profile
from simulation_lab.scene import HOME, build_scene
from simulation_lab.storage import require_space

CONTEXT = ROOT/'docs/robotics/evidence/visual-mug-deployment-v1/attempt-2/engine-upright'
SKILLS = ['bottle', 'plate', 'mug', 'drawer', 'fork', 'spoon']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path, value):
    payload = (json.dumps(value, indent=2)+'\n').encode()
    require_space(path, len(payload)+1024)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(payload)


def declare(path):
    if path.exists():
        raise FileExistsError('Preserve the declared protocol.')
    profile, _ = load_profile()
    cases = []
    def add(skill, kind, **values):
        cases.append(dict(id=f'{len(cases):03d}-{skill}-{kind}', skill=skill, kind=kind, **values))
    for skill in SKILLS:
        add(skill, 'anchor')
    regions = {
        'bottle': {'x': [-.14, .16], 'y': [-.18, -.06], 'yaw_rad': [-.6, .6]},
        'plate': {'x': [-.18, -.04], 'y': [-.05, .09], 'yaw_rad': [-.6, .6]},
        'mug': {'x': [.17, .31], 'y': [-.14, .01], 'yaw_rad': [-.6, .6]},
        'fork': {'drawer_local_x': [-.038, -.008], 'drawer_local_y': [-.04, .01], 'yaw_rad': [-.35, .35]},
        'spoon': {'drawer_local_x': [.008, .038], 'drawer_local_y': [-.04, .01], 'yaw_rad': [-.35, .35]},
        'drawer': {'closed_origin_x': [-.32, -.24], 'closed_origin_y': [.14, .22],
                   'yaw_rad': [-.26, .26], 'opening_m': [0, .04, .08, .12]},
    }
    goals = {'bottle': [[.10, -.115], [.16, -.06], [.20, .05]],
             'plate': [[-.06, -.025], [-.10, -.025], [-.06, -.075]],
             'mug': [[.265, .005], [.215, .005], [.265, .055]],
             'fork': [[-.16, -.045], [-.20, -.045], [-.16, -.095]],
             'spoon': [[-.02, -.155], [-.06, -.155], [-.02, -.105]]}
    for skill in ('bottle', 'plate', 'mug', 'fork', 'spoon'):
        spec = regions[skill]
        if skill in ('fork', 'spoon'):
            xs = np.linspace(*spec['drawer_local_x'], 3)
            ys = np.linspace(*spec['drawer_local_y'], 3)
            coordinate = 'drawer_local_xy'
        else:
            xs = np.linspace(*spec['x'], 6 if skill == 'bottle' else 3)
            ys = np.linspace(*spec['y'], 4 if skill == 'bottle' else 3)
            coordinate = 'source_xy'
        for x in xs:
            for y in ys:
                add(skill, 'position', **{coordinate: [float(x), float(y)]}, yaw_rad=0., destination_xy=goals[skill][0])
        for yaw in spec['yaw_rad']:
            add(skill, 'orientation', yaw_rad=yaw, destination_xy=goals[skill][0])
        for destination in goals[skill][1:]:
            add(skill, 'destination', destination_xy=destination)
    for x in np.linspace(*regions['drawer']['closed_origin_x'], 3):
        for y in np.linspace(*regions['drawer']['closed_origin_y'], 3):
            add('drawer', 'cabinet_position', cabinet_xy=[float(x), float(y)], cabinet_yaw_rad=0., drawer_open_m=0.)
    for yaw in regions['drawer']['yaw_rad']:
        add('drawer', 'cabinet_orientation', cabinet_yaw_rad=yaw)
    for opening in (.04, .08, .12):
        add('drawer', 'opening', drawer_open_m=opening)
    source_names = ['scripts/measure_manipulation_coverage.py'] + [p.relative_to(ROOT).as_posix() for p in (ROOT/'simulation_lab').glob('*.py')]
    bound_names = {**profile['runtime_sources'], **profile['model_files']}
    bound_names.update({name: sha(ROOT/name) for name in source_names})
    context = {name: sha(CONTEXT/name) for name in ('scene.xml', 'states.npz', 'report.json')}
    with np.load(CONTEXT/'states.npz', allow_pickle=False) as trace:
        frames = {skill: int(np.flatnonzero(trace['stage'] == skill)[0]) for skill in SKILLS}
    put(path, {'schema': 'talos.manipulation-coverage.v1', 'declared_on': '2026-09-14',
        'purpose': 'Measure broader supported-object and drawer coverage before choosing a substantive improvement.',
        'context': CONTEXT.relative_to(ROOT).as_posix(), 'context_sha256': context,
        'skill_context_frames': frames, 'source_seed': 42,
        'source_sha256': bound_names, 'source_git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'regions_m': regions, 'destinations_xy_m': goals, 'cases': cases,
        'controllers': ['learned', 'teacher'], 'maximum_workers': 2, 'maximum_wall_seconds_per_case': 120,
        'maximum_simulated_seconds_per_case': 100, 'settle_steps': 300,
        'storage_budget_bytes': 2*1024**3, 'reserve_gib': 10,
        'setup': 'Restore preceding skills from one exposed recorded context, park the arms, change only declared reset coordinates, and settle before control. This isolates each skill; it is not end-to-end or independent random-scene coverage.',
        'learned': 'Unchanged selected neural skills; mug uses its bound live late-placement controller in an isolated diagnostic harness. Public language routing is not bypassed or claimed supported by this experiment.',
        'teacher': 'Separate exact-state controller supplies only feasibility comparison, never actions or object pose to the learned rollout. Teacher failure means not demonstrated, not unreachable.',
        'validity': 'Retain all planned cases. Label initial inter-body penetration >1 mm, object tipping or off-table/unstable reset as invalid; no resampling. Table/drawer supporting contacts remain physical. Solver or perception refusals stay in the valid-case denominator.',
        'exclusions': ['Overlapping/colliding or unsupported reset arrangements', 'Unknown object geometry, liquids, side plates and glasses as manipulated objects', 'Airborne exchange, tipping/sideways objects, obstacle rearrangement and arbitrary destination yaw'],
        'interpretation': 'The boxes are candidate coverage targets, not asserted reachable sets. Report demonstrated physical feasibility and learned success separately for every point. Orientation/configuration/destination rows do not prove their full Cartesian product.',
        'next_gate': 'Require all six learned anchor regressions before broad rollout. Select one improvement from observed failures under a separately frozen training/development/fresh-evaluation protocol. No final-seed tuning here.'})
    print(json.dumps({'declared_cases': len(cases), 'paired_rollouts': 2*len(cases), 'protocol_sha256': sha(path)}), flush=True)


def setup(protocol, case, folder):
    original_xml, layout = build_scene(seed=protocol['source_seed'], scenario='dinner', dinner_preset='task')
    scene = ET.fromstring(original_xml)
    origin = np.asarray(layout['drawer']['closed_origin_m'], dtype=float)
    new_origin = origin.copy(); new_origin[:2] = case.get('cabinet_xy', origin[:2])
    yaw = case.get('cabinet_yaw_rad', 0.)
    rotation = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    for name in ('cutlery_cabinet', 'cutlery_drawer'):
        node = scene.find(f"worldbody/body[@name='{name}']")
        node.set('pos', ' '.join(map(str, new_origin)))
        node.set('euler', f'0 0 {yaw}')
    destination = case.get('destination_xy')
    if destination:
        target = next((t for t in layout['targets'] if t['object_id'] == case['skill']), None)
        if target:
            target['position_m'][:2] = destination
            marker = scene.find(f"worldbody/body[@name='{target['id']}']")
            if marker is None:
                marker = scene.find(f"worldbody/geom[@name='{target['id']}']")
            if marker is not None:
                marker.set('pos', f'{destination[0]} {destination[1]} {layout["table_z"]+.0006}')
        else:
            layout['targets'].append({'id': case['skill']+'_coverage_goal', 'object_id': case['skill'], 'position_m': [*destination, layout['table_z']]})
    scene.find('compiler').set('meshdir', str(ROOT/'simulation_lab/assets/so101/assets'))
    model = mujoco.MjModel.from_xml_string(ET.tostring(scene, encoding='unicode'))
    model.vis.quality.offsamples = 0
    data = mujoco.MjData(model)
    with np.load(ROOT/protocol['context']/'states.npz', allow_pickle=False) as trace:
        index = protocol['skill_context_frames'][case['skill']]
        data.qpos[:] = trace['qpos'][index]; data.qvel[:] = trace['qvel'][index]
    data.qpos[:12] = HOME*2; data.qvel[:12] = 0.; data.ctrl[:] = HOME*2
    before_open = float(data.joint('drawer_slide').qpos[0])
    opening = case.get('drawer_open_m', before_open)
    data.joint('drawer_slide').qpos[0] = opening
    if case['skill'] == 'drawer':
        for name in ('fork', 'spoon'):
            address = model.joint(name+'_free').qposadr[0]
            xyz = data.qpos[address:address+3].copy()
            xyz[1] -= opening-before_open
            data.qpos[address:address+3] = new_origin+rotation@(xyz-origin)
            old_q = data.qpos[address+3:address+7].copy()
            new_q = np.array([math.cos(yaw/2), 0, 0, math.sin(yaw/2)])
            mujoco.mju_mulQuat(data.qpos[address+3:address+7], new_q, old_q)
    else:
        address = model.joint(case['skill']+'_free').qposadr[0]
        if 'source_xy' in case:
            data.qpos[address:address+2] = case['source_xy']
        if 'drawer_local_xy' in case:
            data.qpos[address:address+2] = origin[:2]+np.array([0., -opening])+case['drawer_local_xy']
        if 'yaw_rad' in case:
            angle = case['yaw_rad']/2
            data.qpos[address+3:address+7] = [math.cos(angle), 0, 0, math.sin(angle)]
    layout['drawer']['closed_origin_m'] = new_origin.tolist()
    layout['drawer']['initial_open_m'] = opening
    layout['drawer_open'] = opening >= .10
    mujoco.mj_forward(model, data)
    penetration = max((-float(c.dist) for c in data.contact
        if model.geom_bodyid[c.geom1] != model.geom_bodyid[c.geom2]), default=0.)
    for _ in range(protocol['settle_steps']):
        mujoco.mj_step(model, data)
    data.time = 0.
    mujoco.mj_forward(model, data)
    skill_body = 'cutlery_drawer' if case['skill'] == 'drawer' else case['skill']
    body = data.body(skill_body)
    dof = model.joint('drawer_slide' if case['skill'] == 'drawer' else case['skill']+'_free').dofadr[0]
    speed = float(np.linalg.norm(data.qvel[dof:dof+(1 if case['skill'] == 'drawer' else 3)]))
    valid = penetration <= .001 and float(body.xmat[8]) > .98 and body.xpos[2] >= layout['table_z']-.004 and speed < .005
    state = {'valid': bool(valid), 'initial_max_penetration_m': penetration, 'settled_source_xyz_m': body.xpos.tolist(),
        'settled_upright_z': float(body.xmat[8]), 'settled_speed_m_s': speed, 'drawer_open_m': float(data.joint('drawer_slide').qpos[0]),
        'reason': None if valid else 'Initial inter-body overlap, tipped/unsupported object or unsettled reset.'}
    scene.find('compiler').set('meshdir', os.path.relpath(ROOT/'simulation_lab/assets/so101/assets', folder).replace('\\', '/'))
    require_space(folder/'scene.xml', 1024**2)
    with (folder/'scene.xml').open('xb') as stream:
        stream.write(ET.tostring(scene, encoding='utf-8'))
    return model, data, layout, state


def run_case(args):
    protocol = read(args.protocol)
    case = next(c for c in protocol['cases'] if c['id'] == args.case)
    output = args.output/(args.case+'-'+args.controller)
    if output.exists():
        raise FileExistsError('Preserve every coverage attempt.')
    for name, digest in protocol['source_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Declared source/model changed: '+name)
    for name, digest in protocol['context_sha256'].items():
        if sha(ROOT/protocol['context']/name) != digest:
            raise ValueError('Coverage context changed.')
    preflight = require_space(output, 24*1024**2)
    output.mkdir()
    torch.set_num_threads(2)
    task = None
    trace = {key: [] for key in ('qpos', 'qvel', 'time', 'stage', 'targets')}
    result = {'case': case, 'controller': args.controller, 'protocol_sha256': sha(args.protocol), 'preflight': preflight}
    start = time.perf_counter()
    try:
        model, data, layout, state = setup(protocol, case, output)
        result.update(setup=state, layout=layout)
        targets = data.ctrl.copy()
        if state['valid']:
            suite_path = ROOT/'models/dinner_suite/suite.json'; suite = read(suite_path)
            if args.controller == 'learned':
                if case['skill'] == 'mug':
                    _, mug_protocol = load_profile()
                    task = VisualMugTask(model, data, layout, suite_path.parent/suite['mug'], mug_protocol, 'live')
                else:
                    task = LearnedDinnerTask(model, data, layout, suite_path.parent/suite[case['skill']], case['skill'])
            else:
                task = (DrawerTask if case['skill'] == 'drawer' else DinnerTask)(model, data, layout)
                task.start(side='auto', object_id=case['skill'])
            for tick in range(int(protocol['maximum_simulated_seconds_per_case']/model.opt.timestep)+1):
                q, v = data.qpos.copy(), data.qvel.copy()
                task.update(targets)
                if not np.array_equal(q, data.qpos) or not np.array_equal(v, data.qvel):
                    raise ValueError('Controller wrote authoritative state.')
                if model.neq or np.any(data.xfrc_applied) or np.any(data.qfrc_applied):
                    raise ValueError('Hidden forces or constraints.')
                if tick % 10 == 0 or not task.active:
                    if tick % 2000 == 0:
                        require_space(output, 24*1024**2)
                    for key, value in zip(trace, (q, v, float(data.time), task.stage, targets.copy())):
                        trace[key].append(value)
                if not task.active:
                    break
                if time.perf_counter()-start > protocol['maximum_wall_seconds_per_case']:
                    raise TimeoutError('Declared coverage wall limit.')
                data.ctrl[:] = task.apply_gripper_limit(targets)
                mujoco.mj_step(model, data)
            if task.active:
                task.cancel(targets)
            result.update(status=task.status, task=task.snapshot(), physics_state_writes_during_control=0, hidden_forces=0, equality_constraints=0)
        else:
            result.update(status='invalid_start')
            for key, value in zip(trace, (data.qpos.copy(), data.qvel.copy(), float(data.time), 'invalid_start', targets.copy())):
                trace[key].append(value)
    except Exception as exc:
        result.update(status='harness_error', error_type=type(exc).__name__, message=str(exc))
        if task is not None:
            result['task'] = task.snapshot()
    finally:
        if task is not None and hasattr(task, 'close'):
            task.close()
        result.update(wall_seconds=time.perf_counter()-start, trace_frames=len(trace['time']))
        require_space(output/'states.npz', 24*1024**2)
        with (output/'states.npz').open('xb') as stream:
            np.savez_compressed(stream, **{key: np.asarray(value) for key, value in trace.items()})
        put(output/'report.json', result)
    print(json.dumps({k: result[k] for k in ('case', 'controller', 'status', 'wall_seconds', 'trace_frames')}), flush=True)
    return int(result['status'] == 'harness_error')


def run_batch(args):
    protocol = read(args.protocol)
    if args.output.exists():
        raise FileExistsError('Preserve prior coverage batches.')
    require_space(args.output, protocol['storage_budget_bytes'])
    args.output.mkdir(parents=True)
    put(args.output/'protocol.json', protocol)
    for name in protocol['source_sha256']:
        if name.endswith('.py'):
            payload = (ROOT/name).read_bytes(); target = args.output/'frozen-source'/name
            require_space(target, len(payload)+1024); target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(payload)
    def launch(item):
        case, controller = item
        log = args.output/(case['id']+'-'+controller+'.log')
        if log.exists() or (args.output/(case['id']+'-'+controller)).exists():
            raise FileExistsError('Preserve coverage folder and log.')
        require_space(log, 24*1024**2)
        with log.open('xb') as stream:
            code = subprocess.run([sys.executable, '-I', str(Path(__file__)), '--protocol', str(args.protocol.resolve()),
                '--output', str(args.output.resolve()), '--case', case['id'], '--controller', controller],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode
        result_path = args.output/(case['id']+'-'+controller)/'report.json'
        report = read(result_path) if result_path.exists() else {'status': 'harness_error'}
        row = {'case': case['id'], 'skill': case['skill'], 'kind': case['kind'], 'controller': controller,
               'exit_code': code, 'status': report['status'], 'message': report.get('task', {}).get('message', report.get('message')),
               'report_sha256': sha(result_path) if result_path.exists() else None}
        print(json.dumps(row), flush=True)
        return row
    with ThreadPoolExecutor(max_workers=protocol['maximum_workers']) as pool:
        anchors = list(pool.map(launch, [(c, 'learned') for c in protocol['cases'] if c['kind'] == 'anchor']))
        put(args.output/'anchor-gate.json', {'passed': all(r['status'] == 'succeeded' for r in anchors), 'rows': anchors})
        if not all(r['status'] == 'succeeded' for r in anchors):
            print('Stopped at the declared anchor gate; broad cases remain unexposed.', flush=True)
            return 1
        items = [(c, mode) for c in protocol['cases'] for mode in protocol['controllers'] if not (c['kind'] == 'anchor' and mode == 'learned')]
        rows = anchors+list(pool.map(launch, items))
    put(args.output/'summary.json', {'schema': protocol['schema'], 'protocol_sha256': sha(args.protocol), 'rows': rows,
        'all_declared_rollouts_completed': len(rows) == len(protocol['cases'])*2,
        'harness_errors': sum(r['status'] == 'harness_error' for r in rows)})
    return int(any(r['status'] == 'harness_error' for r in rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declare', type=Path)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--case')
    parser.add_argument('--controller', choices=['learned', 'teacher'])
    args = parser.parse_args()
    if args.declare:
        declare(args.declare)
    else:
        if not args.protocol or not args.output:
            parser.error('Protocol and unused output are required.')
        raise SystemExit(run_case(args) if args.case else run_batch(args))
