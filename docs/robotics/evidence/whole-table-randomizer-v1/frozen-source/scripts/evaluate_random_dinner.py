"""Measure whole-table joint reset validity without filtering policy failures."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from PIL import Image
from simulation_lab.dinner import OBJECTS
from simulation_lab.random_dinner import FAMILIES, assess, draw
from simulation_lab.scene import build_scene
from simulation_lab.storage import require_space


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def put(path, value):
    payload = (json.dumps(value, indent=2)+'\n').encode() if not isinstance(value, bytes) else value
    require_space(path, len(payload)+1024)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:stream.write(payload)


def arrays(path, **values):
    require_space(path, sum(np.asarray(v).nbytes for v in values.values())+1024**2)
    with path.open('xb') as stream:np.savez_compressed(stream, **values)


def declare(path):
    names = ['scripts/evaluate_random_dinner.py', 'simulation_lab/random_dinner.py',
             'simulation_lab/scene.py', 'simulation_lab/dinner.py', 'simulation_lab/storage.py']
    names += [f.relative_to(ROOT).as_posix() for f in (ROOT/'simulation_lab/assets/so101').rglob('*') if f.is_file()]
    p = {'schema': 'talos.whole-table-randomizer.v1', 'declared_on': '2026-09-14',
        'raw_root': '.run/whole-table-randomizer-v1', 'source_sha256': {n: sha(ROOT/n) for n in names},
        'development_seeds': list(range(2026114001, 2026114017)),
        'evaluation_seeds': list(range(2026114101, 2026114165)),
        'orientation_families': FAMILIES, 'yaw_rad': [-float(np.pi), float(np.pi)],
        'maximum_proposals_per_item': 256, 'settle_steps': 600, 'stable_window_steps': 100,
        'storage_budget_bytes': 1024**3, 'reserve_gib': 10,
        'development_gate': {'minimum_valid_joint_scenes': 12},
        'evaluation_gate': {'minimum_valid_joint_scenes': 48, 'minimum_each_requested_family': 4,
                            'minimum_valid_item_x_span_m': .30, 'minimum_valid_item_y_span_m': .20},
        'sampling': 'Shuffle all seven item identities; independently select a listed orientation family and full yaw (plus full axial roll for sideways vessels). Draw XY over the shared tabletop, bounded only by oriented collision geometry/table edges and a conservative kinematic chain-length upper bound for either robot. Retain every rejected proposal. Resample only geometric proposal failures before settling, never a policy or grasp failure.',
        'validity': 'After 3 seconds of unforced gravity/contact settling, require every item supported on the tabletop, inter-body penetration <=1 mm, linear speed <3 mm/s and angular speed <0.08 rad/s throughout the final 0.5 seconds. No upright requirement. Report changed realized orientations and every invalid full arrangement without resampling.',
        'scope': 'This establishes a whole-table scene and observation interface, not manipulation success. Robot-length bounds are necessary geometry, not grasp/path feasibility. All seven displayed loose items include tabletop fork/spoon, side plate and glass; drawer fixture remains closed and fixed. The next dependency is physically verified grasp/route demonstrations and a shared spatial action policy on these joint scenes.',
        'stop': 'Stop after a failed development gate or harness error, preserve attempted outcomes and keep later seeds unexposed. Never change this declared sampler or limits to rescue this protocol.'}
    if path.exists() or (ROOT/p['raw_root']).exists():raise FileExistsError('Preserve the randomizer experiment.')
    p['preflight'] = require_space(ROOT/p['raw_root'], p['storage_budget_bytes'])
    put(path, p); print({'protocol_sha256': sha(path), 'items': list(OBJECTS)}, flush=True)


def episode(p, path, stage, seed):
    folder = ROOT/p['raw_root']/stage/str(seed)
    if folder.exists():raise FileExistsError('Preserve every joint scene.')
    preflight = require_space(folder, 16*1024**2); folder.mkdir(parents=True)
    xml, layout = build_scene(seed=seed, scenario='dinner', dinner_preset='task')
    model = mujoco.MjModel.from_xml_string(xml); model.vis.quality.offsamples = 0
    data, recipe = draw(model, seed, maximum_proposals_per_item=p['maximum_proposals_per_item'])
    tree = ET.fromstring(xml)
    tree.find('compiler').set('meshdir', os.path.relpath(ROOT/'simulation_lab/assets/so101/assets', folder).replace('\\', '/'))
    put(folder/'scene.xml', ET.tostring(tree, encoding='utf-8'))
    put(folder/'recipe.json', recipe)
    initial = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(model, data, initial, mujoco.mjtState.mjSTATE_INTEGRATION)
    trace = {'qpos': [], 'qvel': [], 'time': []}; stability = []
    if recipe['generated']:
        for tick in range(p['settle_steps']+1):
            if tick % 10 == 0:
                for key, value in zip(trace, (data.qpos.copy(), data.qvel.copy(), float(data.time))):trace[key].append(value)
            if tick >= p['settle_steps']-p['stable_window_steps']:
                stability.append(assess(model, data))
            if tick == p['settle_steps']:break
            assert model.neq == 0 and not np.any(data.xfrc_applied) and not np.any(data.qfrc_applied)
            mujoco.mj_step(model, data)
        final = stability[-1]
        valid = all(s['valid'] for s in stability)
    else:final = None; valid = False
    arrays(folder/'states.npz', initial_integration=initial, **{k: np.asarray(v) for k, v in trace.items()})
    put(folder/'stability-window.json', stability)
    images = {}
    if recipe['generated']:
        renderer = mujoco.Renderer(model, width=480, height=360)
        try:
            for camera in ('overhead', 'opposite', 'overview'):
                renderer.update_scene(data, camera=camera); rgb = renderer.render().copy()
                if rgb.mean() < 10:raise ValueError('Camera image is black.')
                target = folder/(camera+'.png'); require_space(target, 1024**2)
                with target.open('xb') as stream:Image.fromarray(rgb).save(stream, format='PNG')
                images[camera] = sha(target)
        finally:renderer.close()
    report = {'schema': p['schema'], 'seed': seed, 'stage': stage, 'protocol_sha256': sha(path),
        'generated': recipe['generated'], 'valid_joint_scene': bool(valid), 'final_geometry': final,
        'proposals': len(recipe['proposals']), 'rejected_proposals': dict(Counter(x['rejection'] for x in recipe['proposals'] if x['rejection'])),
        'stability_samples': len(stability), 'trace_frames': len(trace['time']), 'images_sha256': images,
        'preflight': preflight, 'control_policy_used': False, 'grasp_or_route_feasibility_claimed': False}
    put(folder/'report.json', report)
    row = {'seed': seed, 'valid_joint_scene': bool(valid), 'generated': recipe['generated'],
           'report_sha256': sha(folder/'report.json'), 'proposals': report['proposals']}
    print(row, flush=True); return row


def run(path, stage):
    p = read(path); raw = ROOT/p['raw_root']; folder = raw/stage
    for name, digest in p['source_sha256'].items():assert sha(ROOT/name) == digest, name
    if folder.exists():raise FileExistsError('Preserve this stage.')
    if stage == 'evaluation':
        previous = read(raw/'development/gate.json')
        assert previous['passed'] and previous['protocol_sha256'] == sha(path)
    require_space(folder, p['storage_budget_bytes']); folder.mkdir(parents=True)
    if stage == 'development':
        put(raw/'protocol.json', p)
        for name in p['source_sha256']:
            if name.endswith('.py'):put(raw/'frozen-source'/name, (ROOT/name).read_bytes())
    rows = []
    try:
        for seed in p[stage+'_seeds']:rows.append(episode(p, path, stage, seed))
    except Exception as exc:
        put(folder/'stopped.json', {'error_type': type(exc).__name__, 'message': str(exc), 'rows': rows})
        raise
    valid = sum(r['valid_joint_scene'] for r in rows)
    passed = valid >= p[stage+'_gate']['minimum_valid_joint_scenes']
    coverage = {}
    for name in OBJECTS:
        recipes = [read(folder/str(r['seed'])/'recipe.json')['objects'].get(name) for r in rows]
        families = Counter(x['family'] for x in recipes if x)
        quadrants = Counter(int((x['yaw_rad']+np.pi)/(np.pi/2)) % 4 for x in recipes if x)
        positions = [read(folder/str(r['seed'])/'report.json')['final_geometry']['objects'][name]['position_m'][:2]
                     for r in rows if r['valid_joint_scene']]
        span = np.ptp(positions, axis=0).tolist() if positions else [0., 0.]
        coverage[name] = {'requested_families': dict(families), 'requested_yaw_quadrants': dict(quadrants), 'valid_joint_scene_xy_span_m': span}
        if stage == 'evaluation':
            g = p['evaluation_gate']
            passed &= all(families[f] >= g['minimum_each_requested_family'] for f in FAMILIES[name])
            passed &= len(quadrants) == 4 and span[0] >= g['minimum_valid_item_x_span_m'] and span[1] >= g['minimum_valid_item_y_span_m']
    put(folder/'gate.json', {'schema': p['schema'], 'protocol_sha256': sha(path), 'passed': bool(passed),
        'valid_joint_scenes': valid, 'planned': len(rows), 'all_outcomes_retained': len(rows) == len(p[stage+'_seeds']),
        'coverage': coverage, 'rows': rows, 'manipulation_success_claimed': False})
    print({'stage': stage, 'passed': bool(passed), 'valid_joint_scenes': valid, 'planned': len(rows)}, flush=True)
    return int(not passed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declare', type=Path); parser.add_argument('--protocol', type=Path)
    parser.add_argument('--stage', choices=['development', 'evaluation']); args = parser.parse_args()
    if args.declare:declare(args.declare)
    elif args.protocol and args.stage:raise SystemExit(run(args.protocol, args.stage))
    else:parser.error('Declare or select a stage.')
