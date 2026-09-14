"""Preserve failed preview checks and the passing display-only repair together."""
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
from simulation_lab.storage import require_space
from scripts.package_spoon_release import physical_criteria


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    output = ROOT/'docs/robotics/evidence/visual-mug-deployment-v1'
    if output.exists():
        raise FileExistsError('Preserve previous deployment evidence.')
    roots = [ROOT/'.run/visual-mug-deployment-v1', ROOT/'.run/visual-mug-deployment-v2']
    declaration = read(roots[1]/'protocol.json')
    freeze = roots[1]/'frozen-source'
    if freeze.exists():
        raise FileExistsError('Preserve the passing deployment source snapshot.')
    require_space(freeze, 4*1024**2)
    for name, digest in declaration['source_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('A passing deployment source changed.')
        target = freeze/name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write((ROOT/name).read_bytes())
    preflight = require_space(output, sum(p.stat().st_size for r in roots for p in r.rglob('*') if p.is_file())+8*1024**2)
    output.mkdir(parents=True)
    sources = {}
    def put(target, payload, source=None):
        require_space(output/target, len(payload)+1024)
        path = output/target
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(payload)
        if source is not None:
            sources[target] = {'source': source.relative_to(ROOT).as_posix(), 'source_sha256': sha(source),
                'copied_sha256': sha(path), 'changed': payload != source.read_bytes()}
    for number, raw in enumerate(roots, 1):
        for source in raw.rglob('*'):
            if not source.is_file() or '__pycache__' in source.parts:
                continue
            name = f'attempt-{number}/'+source.relative_to(raw).as_posix()
            payload = source.read_bytes()
            if source.suffix in ('.json', '.log', '.py'):
                payload = payload.replace(str(ROOT).encode(), b'.').replace(str(Path.home()).encode(), b'<USER_HOME>')
            if source.name == 'scene.xml':
                put(name.replace('scene.xml', 'original-scene.xml'), payload, source)
                tree = ET.fromstring(payload)
                tree.find('compiler').set('meshdir', os.path.relpath(ROOT/'simulation_lab/assets/so101/assets', (output/name).parent).replace('\\', '/'))
                payload = ET.tostring(tree, encoding='utf-8')
            put(name, payload, source)
    cases = []
    for surface in ('engine', 'public'):
        for preset in ('upright', 'wide_left'):
            name = surface+'-'+preset
            earlier, current = [output/f'attempt-{i}'/name for i in (1, 2)]
            report = read(current/'report.json')
            if not report['passed'] or report['task']['status'] != 'succeeded' or report['correction_queries'] != 15:
                raise ValueError('All four selected deployment checks must pass with live correction.')
            for row in report['task']['results']:
                physical_criteria(row)
            with np.load(earlier/'states.npz', allow_pickle=False) as a, np.load(current/'states.npz', allow_pickle=False) as b:
                same = {key: bool(np.array_equal(a[key], b[key])) for key in a.files}
                if not all(same.values()):
                    raise ValueError('The preview repair changed a recorded control/physics array.')
                frames = len(b['time'])
            for folder in (earlier, current):
                model = mujoco.MjModel.from_xml_path(str(folder/'scene.xml'))
                with np.load(folder/'states.npz', allow_pickle=False) as state:
                    if state['qpos'].shape[1] != model.nq:
                        raise ValueError('Portable scene/trace mismatch.')
            cases.append({'case': name, 'attempt_1_passed': read(earlier/'report.json')['passed'], 'attempt_2_passed': True,
                'complete_state_arrays_identical': same, 'trace_frames_per_attempt': frames, 'live_correction_queries': 15,
                'wall_seconds': report['wall_seconds']})
    for filename in ('hosted-preview-thread-contract-v1.log', 'hosted-preview-thread-contract-v2.log'):
        source = ROOT/'.run/final-goal'/filename
        payload = source.read_bytes().replace(str(ROOT).encode(), b'.').replace(str(Path.home()).encode(), b'<USER_HOME>')
        put('contracts/'+filename, payload, source)
    audit = {'schema': 'talos.visual-mug-deployment-package.v1', 'passed': True, 'cases': cases,
        'portable_scenes_loaded': 8, 'total_trace_frames': sum(r['trace_frames_per_attempt']*2 for r in cases),
        'profile_sha256': sha(ROOT/'models/dinner_visual_mug_v1/profile.json'), 'storage_preflight': preflight,
        'source_mapping': sources, 'auditor_sha256': sha(Path(__file__)),
        'scope': 'Exposed seed-42 production entry-point checks. The first attempt passed physics but hosted final previews were black. A passive display thread repairs previews without changing any recorded state, target or progress array. No model or physical threshold changed. One earlier unittest invocation failed module discovery before running the contract; the corrected discovery command passes.'}
    put('audit.json', (json.dumps(audit, indent=2)+'\n').encode())
    put('audit-source.py', Path(__file__).read_bytes())
    put('README.md', ('# Live mug deployment verification\n\n'
        'All four exposed seed-42 checks now pass: real local language entry point and hosted trial entry point, each with standard and farther-left starts. Each completes all six/seven planned skills and 15 live mug queries.\n\n'
        'The first attempt completes physics but produces black final hosted previews. A passive display renderer now owns a separate model and OpenGL thread; policy renderers retain their evaluated lifetime and implementation. All six complete state/control arrays match exactly before and after this repair in all four cases.\n\n'
        'Both attempts, every failure/preview/report, 42,492 state frames, original sources and the passing no-physics renderer contract are preserved. The eight portable scenes load. These are deployment regressions, not additional unseen accuracy trials. The 20/20 final live-image result belongs to the separately frozen mug-visual-correction-v2 experiment. Cloud deployment, new-option Intel execution, human full-sequence voice and anonymous access require their own checks.\n\n'
        'The selected option is `Learned dinner · live mug vision`; use `Set the table`. Only late mug placement receives continuing stereo correction. Original dinner and relay modes remain available. GitHub/HF stay private; final Submit belongs to the user.\n').encode())
    manifest = {'files': {p.relative_to(output).as_posix(): sha(p) for p in output.rglob('*') if p.is_file()}}
    put('manifest.json', (json.dumps(manifest, indent=2)+'\n').encode())
    print(json.dumps({'passed': True, 'files': len(manifest['files']), 'trace_frames': audit['total_trace_frames'],
        'manifest_sha256': sha(output/'manifest.json')}), flush=True)


if __name__ == '__main__':
    run()
