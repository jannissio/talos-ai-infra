"""Archive the six exact wider-bottle task-wrapper implementation regressions."""
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.bottle_refinement_experiment import read, sha, space, write


def run():
    protocol = ROOT/'docs/robotics/experiments/bottle-wide-deployment-v1.json'
    d = read(protocol); p = read(ROOT/d['physical_protocol'])
    raw = ROOT/d['raw_root']/'parity'
    output = ROOT/'docs/robotics/evidence/bottle-wide-deployment-v1/parity'
    gate = read(raw/'gate.json')
    assert gate['passed'] and gate['all_six_outcomes_retained'] and len(gate['rows']) == 6
    if output.exists():
        raise FileExistsError('Preserve the wrapper evidence archive.')
    preflight = space(p, output, sum(f.stat().st_size for f in raw.rglob('*') if f.is_file())+16*1024**2)
    copied = {}
    def copy(source, target):
        space(p, target, source.stat().st_size+1024**2)
        write(target, source.read_bytes())
        copied[target.relative_to(output).as_posix()] = {'source': source.relative_to(ROOT).as_posix(), 'sha256': sha(source)}
    for source in raw.rglob('*'):
        if source.is_file():
            relative = source.relative_to(raw)
            if source.name == 'scene.xml':
                relative = relative.parent/'source-scene.xml'
            copy(source, output/relative)
    for source in output.rglob('source-scene.xml'):
        tree = ET.parse(source)
        tree.find('compiler').set('meshdir', os.path.relpath(ROOT/'simulation_lab/assets/so101/assets', source.parent).replace('\\', '/'))
        write(source.parent/'scene.xml', ET.tostring(tree.getroot(), encoding='utf-8'))
    for name, digest in d['source_sha256'].items():
        assert sha(ROOT/name) == digest, name
        copy(ROOT/name, output/'source'/name)
    for suffix in ('prepare', 'parity'):
        source = ROOT/'.run/final-goal'/f'bottle-wide-deployment-v1-{suffix}.log'
        copy(source, output/'console'/source.name)
    rows = []; frames = 0
    for row in gate['rows']:
        folder = output/str(row['seed']); report = read(folder/'report.json')
        assert sha(folder/'report.json') == row['report_sha256']
        assert report['passed'] and report['status'] == 'succeeded' and report['physical']['passed']
        reference = ROOT/p['evidence_package']/'physical/evaluation'/f"{row['seed']}-live"/'states.npz'
        with np.load(folder/'states.npz', allow_pickle=False) as actual, np.load(reference, allow_pickle=False) as expected:
            count = len(actual['time']); frames += count
            assert count == report['trace_frames']
            for key in ('qpos', 'qvel', 'time', 'targets'):
                assert np.array_equal(actual[key], expected[key]), (row['seed'], key)
        rows.append({**row, 'all_physics_and_motor_arrays_identical': True, 'trace_frames': count})
    audit = {'schema': d['schema'], 'passed': True, 'all_six_outcomes_retained': True, 'rows': rows,
             'trace_frames': frames, 'preflight': preflight, 'source_copies': copied,
             'scope': 'Six exposed implementation regressions, with every qpos/qvel/time/target value identical to the original frozen evaluator. They add no fresh generalization trials. Browser entry points and complete workflows still require separately declared checks.'}
    write(output/'audit.json', audit)
    write(output/'audit-source.py', Path(__file__).read_bytes())
    write(output/'README.md', f'''# Wider bottle task-wrapper parity

All **6/6** exposed implementation regressions pass. Every value of every recorded joint-position, velocity, time and motor-target array is **bit-for-bit identical** to the frozen physical evaluator. The package retains **{frames:,}** state frames, all observations, original outcomes, portable scenes and exact wrapper/build sources.

The copied observer and neural motor are unchanged. The wrapper adds requested-destination and source-region guards, cancellation/status reporting and optional dinner sequencing. These six tests verify only the standalone bottle wrapper; full dinner chaining and browser/hosted entry points remain unverified here. No new fresh-generalization sample is claimed, and no browser deployment follows merely from this package.
'''.encode())
    write(output/'manifest.json', {'schema': d['schema'], 'files': {f.relative_to(output).as_posix(): sha(f) for f in output.rglob('*') if f.is_file()}})
    print({'exact_regressions': 6, 'state_frames': frames}, flush=True)


if __name__ == '__main__':
    run()
