"""Preserve all joint scenes and exactly replay their gravity/contact settling."""
from collections import Counter
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from scripts.evaluate_random_dinner import read, sha, put
from simulation_lab.random_dinner import assess, draw
from simulation_lab.storage import require_space


def run():
    protocol = ROOT/'docs/robotics/experiments/whole-table-randomizer-v1.json'
    p = read(protocol); raw = ROOT/p['raw_root']
    output = ROOT/'docs/robotics/evidence/whole-table-randomizer-v1'
    if output.exists():raise FileExistsError('Preserve the joint-scene archive.')
    gates = {s: read(raw/s/'gate.json') for s in ('development', 'evaluation')}
    assert gates['development']['passed'] and gates['evaluation']['passed']
    preflight = require_space(output, sum(f.stat().st_size for f in raw.rglob('*') if f.is_file())+16*1024**2)
    copied = {}
    for source in raw.rglob('*'):
        if not source.is_file():continue
        rel = source.relative_to(raw)
        if source.name == 'scene.xml':rel = rel.parent/'source-scene.xml'
        target = output/rel; put(target, source.read_bytes())
        copied[rel.as_posix()] = {'source': source.relative_to(ROOT).as_posix(), 'sha256': sha(source)}
    for source in output.rglob('source-scene.xml'):
        tree = ET.parse(source)
        tree.find('compiler').set('meshdir', os.path.relpath(ROOT/'simulation_lab/assets/so101/assets', source.parent).replace('\\', '/'))
        put(source.parent/'scene.xml', ET.tostring(tree.getroot(), encoding='utf-8'))
    for stage in gates:
        source = ROOT/'.run/final-goal'/f'whole-table-randomizer-v1-{stage}.log'
        put(output/'console'/source.name, source.read_bytes())
    for name, digest in p['source_sha256'].items():
        snapshot = output/'frozen-source'/name
        assert sha(snapshot if snapshot.exists() else ROOT/name) == digest, name
    records = []; frames = proposals = 0; reasons = Counter()
    for stage, gate in gates.items():
        assert gate['protocol_sha256'] == sha(protocol) and gate['all_outcomes_retained']
        assert {r['seed'] for r in gate['rows']} == set(p[stage+'_seeds'])
        valid_count = 0
        for row in gate['rows']:
            folder = output/stage/str(row['seed']); r = read(folder/'report.json')
            assert sha(folder/'report.json') == row['report_sha256'] and r['generated']
            recipe = read(folder/'recipe.json'); proposals += len(recipe['proposals'])
            model = mujoco.MjModel.from_xml_path(str(folder/'scene.xml'))
            data, regenerated = draw(model, row['seed'], maximum_proposals_per_item=p['maximum_proposals_per_item'])
            assert regenerated == recipe
            with np.load(folder/'states.npz', allow_pickle=False) as z:state = {k: z[k] for k in z.files}
            initial = np.empty_like(state['initial_integration'])
            mujoco.mj_getState(model, data, initial, mujoco.mjtState.mjSTATE_INTEGRATION)
            assert np.array_equal(initial, state['initial_integration'])
            assert len(state['time']) == r['trace_frames'] == 61
            original_window = read(folder/'stability-window.json'); window = []; cursor = 0
            for tick in range(p['settle_steps']+1):
                if tick % 10 == 0:
                    assert np.array_equal(data.qpos, state['qpos'][cursor])
                    assert np.array_equal(data.qvel, state['qvel'][cursor])
                    assert data.time == state['time'][cursor]; cursor += 1
                if tick >= p['settle_steps']-p['stable_window_steps']:window.append(assess(model, data))
                if tick < p['settle_steps']:mujoco.mj_step(model, data)
            assert window == original_window and window[-1] == r['final_geometry']
            valid = all(w['valid'] for w in window)
            assert valid == row['valid_joint_scene'] == r['valid_joint_scene']
            if not valid:
                reasons.update({(name, why) for sample in window for name, obj in sample['objects'].items() for why in obj['reasons']})
            valid_count += int(valid); frames += cursor
            for camera, digest in r['images_sha256'].items():assert sha(folder/(camera+'.png')) == digest
            records.append({**row, 'exact_settling_replay': True, 'trace_frames': cursor})
        assert valid_count == gate['valid_joint_scenes']
    audit = {'schema': p['schema'], 'audit_passed': True, 'all_80_outcomes_retained': len(records) == 80,
        'state_frames': frames, 'recorded_proposals': proposals,
        'stable_joint_scenes': {s: g['valid_joint_scenes'] for s, g in gates.items()},
        'invalid_scene_reasons': [{'object_id': k[0], 'reason': k[1], 'scenes': n} for k, n in reasons.items()],
        'rows': records, 'source_copies': copied, 'preflight': preflight,
        'verification': 'Every recipe and initial integration state regenerates; all 80 physical settling replays exactly match every saved qpos/qvel/time value and every final-window geometry/stability sample. All original RGB hashes and source hashes match.',
        'scope': 'Whole-table geometry and observation dataset, not manipulation demonstrations or grasp/path certification. Full yaw and alternative orientations are explicit. All seven items are loose tabletop objects; the drawer remains a fixed closed fixture.'}
    put(output/'audit.json', audit); put(output/'audit-source.py', Path(__file__).read_bytes())
    put(output/'README.md', f'''# Joint whole-table scene generation

All **80** declared scenes and **{proposals}** geometric proposals are retained. Stable complete arrangements: **14/16 development**, **63/64 final**. All seven items share the tabletop sampling domain: bottle, both plates, mug, glass, fork and spoon. Requested yaw spans the full circle. Vessels include upright/inverted/sideways starts; plates and cutlery include both faces. Physics determines the realized resting orientation.

Every saved reset and all **{frames:,}** settling frames independently replay exactly. Invalid stability cases remain in the planned denominator. The sampler conditions only on table/object geometry, initial non-intersection and a conservative robot-chain length bound. That bound does not certify a grasp or route, and successful scene generation is not successful table setting.

The next dependency is physical grasp/route demonstrations and a shared spatial action policy evaluated on jointly randomized complete workflows. Side plate, glass, broad orientations and tabletop cutlery remain manipulation work. See the single authoritative goal in `docs/PC_HANDOFF.md`. The running demonstration and private fallback are unchanged.
'''.encode())
    put(output/'manifest.json', {'schema': p['schema'], 'files': {f.relative_to(output).as_posix(): sha(f) for f in output.rglob('*') if f.is_file()}})
    print({k: audit[k] for k in ('audit_passed', 'state_frames', 'recorded_proposals', 'stable_joint_scenes', 'invalid_scene_reasons')}, flush=True)


if __name__ == '__main__':run()
