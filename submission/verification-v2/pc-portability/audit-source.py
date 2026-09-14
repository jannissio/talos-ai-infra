"""Package the isolated two-workflow portability check without private paths."""
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import zipfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from simulation_lab.storage import require_space
from scripts.package_spoon_release import physical_criteria


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def run():
    archive = ROOT/'submission/verification-v2/talos-intel-live-mug-verification.zip'
    extracted = ROOT/'.run/intel-verification-kit-v2/extracted'
    raw = extracted/'.run/pc-portability-v1'
    output = ROOT/'submission/verification-v2/pc-portability'
    if output.exists():
        raise FileExistsError('Preserve prior portability evidence.')
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read('kit-manifest.json'))
        assert z.testzip() is None
        assert set(z.namelist()) == set(manifest['files']) | {'kit-manifest.json'}
        for name, digest in manifest['files'].items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest and sha(extracted/name) == digest, name
        members = len(z.namelist())
    verification = read(raw/'verification.json')
    assert verification['physical_passed'] and not verification['strict_hardware_and_physics_passed']
    expected = ['bottle', 'plate', 'mug', 'drawer', 'fork', 'spoon']
    frame_count = 0
    for preset in ('upright', 'wide_left'):
        report = read(raw/('engine-'+preset)/'report.json')
        assert report['passed'] and report['correction_queries'] > 0
        steps = expected if preset == 'upright' else ['reverse_bottle_right', 'reverse_bottle_left', *expected[1:]]
        assert report['task']['completed_steps'] == steps
        for row in report['task']['results']:
            physical_criteria(row)
    mapping = {p: p.relative_to(raw).as_posix() for p in raw.rglob('*') if p.is_file()}
    mapping[extracted/'kit-manifest.json'] = 'kit-manifest.json'
    for name in ('intel-verification-kit-v2-extraction.json', 'intel-verification-kit-v2-extraction-harness-incident.json', 'intel-verification-kit-v2-physical.log', 'intel-verification-kit-v2-build.log'):
        mapping[ROOT/'.run/final-goal'/name] = 'preparation/'+name
    preflight = require_space(output, sum(p.stat().st_size for p in mapping)+8*1024**2)
    output.mkdir(parents=True)
    for source, name in mapping.items():
        if name.endswith('/scene.xml'):
            name = name[:-len('scene.xml')]+'source-scene.xml'
        target = output/name
        require_space(target, source.stat().st_size+1024)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(source.read_bytes())
    for preset in ('upright', 'wide_left'):
        folder = output/('engine-'+preset)
        tree = ET.parse(folder/'source-scene.xml')
        tree.find('compiler').set('meshdir', os.path.relpath(ROOT/'simulation_lab/assets/so101/assets', folder).replace('\\', '/'))
        with (folder/'scene.xml').open('xb') as stream:
            stream.write(ET.tostring(tree.getroot(), encoding='utf-8'))
        model = mujoco.MjModel.from_xml_path(str(folder/'scene.xml'))
        data = mujoco.MjData(model)
        with np.load(folder/'states.npz', allow_pickle=False) as trace:
            assert trace['qpos'].shape[1] == model.nq and np.isfinite(trace['qpos']).all()
            for index in (0, len(trace['time'])-1):
                data.qpos[:] = trace['qpos'][index]; data.qvel[:] = trace['qvel'][index]
                mujoco.mj_forward(model, data)
            frame_count += len(trace['time'])
    result = {'schema': 'talos.visual-mug-kit-portability.v1', 'archive_sha256': sha(archive),
        'archive_bytes': archive.stat().st_size, 'members': members, 'source_git_revision': manifest['source_git_revision'],
        'all_archive_extracted_hashes_match': True, 'isolated_python': True, 'physical_passed': True,
        'cases': verification['cases'], 'trace_frames': frame_count, 'strict_hardware_and_physics_passed': False,
        'hardware': verification['hardware'], 'storage_preflight': preflight,
        'scope': 'Two exposed entry-point checks from an isolated kit on AMD/NVIDIA. Actual legacy Intel baseline is separately verified; the changed live-mug components still require their scoped laptop run. No wider coverage claim.'}
    with (output/'audit.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2)
    with (output/'audit-source.py').open('xb') as stream:
        stream.write(Path(__file__).read_bytes())
    records = {p.relative_to(output).as_posix(): sha(p) for p in output.rglob('*') if p.is_file()}
    with (output/'manifest.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump({'schema': result['schema'], 'files': records}, stream, indent=2)
    print(json.dumps({k: result[k] for k in ('archive_bytes', 'members', 'physical_passed', 'trace_frames')}), flush=True)


if __name__ == '__main__':
    run()
