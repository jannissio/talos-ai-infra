"""Preserve the entire spoon input gate and portable initial-state reproduction."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from scripts.prepare_bottle_data import STATE
from simulation_lab.storage import GIB, require_space


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(folder):
    protocol, info, replay = read(folder/'protocol.json'), read(folder/'retrieval.json'), read(folder/'input-replay.json')
    transform = read(folder/'transformation.json')
    if (sha(folder/'retrieval.npz') != replay['source_sha256'] or transform['candidate_sha256'] != replay['source_sha256']
            or sha(folder/'protocol.json') != transform['protocol_sha256']
            or info['parent_training_input_sha256'] != protocol['source_sha256']
            or sha(ROOT/protocol['source']/'retrieval.npz') != protocol['source_sha256']):
        raise ValueError('Input, gate or transformation provenance changed.')
    names = info['episodes']
    rows = replay['episodes']
    if len(names) != 39 or [r['episode'] for r in rows] != names or not replay['complete']:
        raise ValueError('The complete 39-episode denominator is required.')
    if replay['passed'] != all(r['passed'] for r in rows):
        raise ValueError('The summary contradicts a preserved outcome.')
    inputs = read(folder/'reproduction-inputs.json')
    for name,digest in inputs['runtime_source_sha256'].items():
        if sha(folder/'source'/name) != digest:
            raise ValueError('A captured runtime source changed.')
    for name,digest in inputs['repository_asset_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('A required simulation asset changed.')
    with np.load(folder/'retrieval.npz',allow_pickle=False) as packed:
        current = {name:packed[name] for name in packed.files}
    with np.load(ROOT/protocol['source']/'retrieval.npz',allow_pickle=False) as packed:
        previous = {name:packed[name] for name in packed.files}
    if any(not np.array_equal(current[name],previous[name]) for name in current if name != 'actions'):
        raise ValueError('An undeclared input array changed.')
    changes = np.max(abs(current['actions'][:,:5]-previous['actions'][:,:5]))
    if changes > protocol['maximum_active_joint_label_change_rad'] or not np.array_equal(current['actions'][:,6:],previous['actions'][:,6:]):
        raise ValueError('Action changes exceed the declared scope.')
    models_loaded = 0
    for row in rows:
        base = folder/'episodes'/row['episode']
        manifest = base/'spoon/manifest.json'
        expected = next(r['manifest_sha256'] for r in info['lineage'] if r['episode']==row['episode'])
        if sha(manifest) != expected:
            raise ValueError('An original collection manifest changed.')
        model = mujoco.MjModel.from_xml_path(str((base/'scene.xml').resolve()))
        data = mujoco.MjData(model)
        state = np.load(base/'spoon/initial-integration-state.npy',allow_pickle=False)
        if len(state) != mujoco.mj_stateSize(model,STATE) or not np.isfinite(state).all():
            raise ValueError('An initial integration state is incomplete or nonfinite.')
        mujoco.mj_setState(model,data,state,STATE)
        mujoco.mj_forward(model,data)
        if not np.isfinite(data.qpos).all():
            raise ValueError('A reconstructed initial scene is nonfinite.')
        m = row['metrics']
        if row['passed'] and (row['failure'] is not None or m['placement_error_mm'] >= 8
                             or m['placement_z_error_mm'] >= 4 or m['hold_verified_s'] < 1.49
                             or not m['both_arms_parked'] or m['stable_release_and_park_s'] < .5-1e-8
                             or m['unexpected_collisions'] != 0 or m['other_object_max_displacement_m'] > .004
                             or m['parked_arm_max_motion_deg'] > 1 or m['longest_unsupported_gap_s'] > .18):
            raise ValueError('A successful replay violates unchanged physical criteria.')
        models_loaded += 1
    return dict(episodes=len(rows), passed=sum(row['passed'] for row in rows), gate_passed=replay['passed'],
                portable_initial_scenes_verified=models_loaded, maximum_active_joint_label_change_rad=float(changes),
                failures=[{'episode':r['episode'],'failure':r['failure'],'metrics':r['metrics']} for r in rows if not r['passed']],
                training_started=False, scope='Physical saved-action replay only; no learned-policy evaluation or promotion.')


def package(args):
    if args.output.exists():
        raise FileExistsError('Preserve the existing input package.')
    protocol = read(args.protocol)
    source = args.source
    replay = read(source/'input-replay.json')
    if not replay['complete']:
        raise ValueError('Wait for the full replay batch before packaging.')
    used = sum(p.stat().st_size for base in (ROOT/'.run/spoon-release-v1', ROOT/'training/spoon_release_v1',
              ROOT/'models/spoon_release_v1',ROOT/'docs/robotics/evidence/spoon-release-v1') if base.exists() for p in base.rglob('*') if p.is_file())
    if used+64*1024**2 > protocol['budget']['maximum_data_gib_including_packages']*GIB:
        raise ValueError('Input packaging exceeds the declared cumulative data budget.')
    require_space(args.output,64*1024**2)
    args.output.mkdir(parents=True)
    entries = []

    def write(name,payload,transformation='byte-identical copy'):
        target = args.output/name
        require_space(target,len(payload)+1024**2)
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:
            stream.write(payload)
        entries.append({'file':name,'sha256':sha(target),'bytes':len(payload),'transformation':transformation})

    for name in ('retrieval.npz','retrieval.json','input-replay.json','transformation.json'):
        write(name,(source/name).read_bytes())
    write('protocol.json',args.protocol.read_bytes())
    info = read(source/'retrieval.json')
    assets = {}
    for episode in info['episodes']:
        require_space(args.output,4*1024**2)
        expected = next(r['manifest_sha256'] for r in info['lineage'] if r['episode']==episode)
        matches = [ROOT/base/episode for base in protocol['raw_episode_sources'] if (ROOT/base/episode/'spoon/manifest.json').is_file()
                   and sha(ROOT/base/episode/'spoon/manifest.json')==expected]
        if len(matches) != 1:
            raise ValueError('Need the exact original source episode: '+episode)
        raw = matches[0]
        for name in ('manifest.json','initial-integration-state.npy'):
            write('episodes/'+episode+'/spoon/'+name,(raw/'spoon'/name).read_bytes())
        tree = ET.fromstring((raw/'scene.xml').read_bytes())
        compiler = tree.find('compiler')
        meshdir = (raw/compiler.get('meshdir')).resolve()
        if not meshdir.is_relative_to(ROOT/'simulation_lab/assets'):
            raise ValueError('A mesh directory is outside repository assets.')
        for element in tree.findall('./asset/mesh'):
            path = meshdir/element.get('file')
            assets[path.relative_to(ROOT).as_posix()] = sha(path)
        target_parent = args.output/'episodes'/episode
        compiler.set('meshdir',os.path.relpath(meshdir,target_parent).replace('\\','/'))
        write('episodes/'+episode+'/scene.xml',ET.tostring(tree,encoding='utf-8'),'Only meshdir remapped to the same repository assets.')
    names = [p.relative_to(ROOT).as_posix() for p in (ROOT/'simulation_lab').rglob('*.py') if '__pycache__' not in p.parts]
    names += ['scripts/prepare_spoon_release.py','scripts/verify_dinner_learning_inputs.py','scripts/prepare_bottle_data.py','scripts/package_spoon_release_inputs.py']
    hashes = {}
    for name in names:
        write('source/'+name,(ROOT/name).read_bytes())
        hashes[name] = sha(ROOT/name)
    write('reproduction-inputs.json',(json.dumps({'runtime_source_sha256':hashes,'repository_asset_sha256':assets},indent=2)+'\n').encode(),'Explicit reproducibility manifest')
    result = audit(args.output)
    write('audit.json',(json.dumps(result,indent=2)+'\n').encode(),'Independent package audit')
    require_space(args.output/'manifest.json',1024**2)
    (args.output/'manifest.json').write_bytes((json.dumps({'files':entries,'originals_modified':False,'all_episodes_retained':True},indent=2)+'\n').encode())
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'docs/robotics/experiments/spoon-release-v1.json')
    parser.add_argument('--source',type=Path,default=ROOT/'.run/spoon-release-v1/input')
    parser.add_argument('--output',type=Path,default=ROOT/'training/spoon_release_v1')
    parser.add_argument('--verify-only',action='store_true')
    args = parser.parse_args()
    if args.verify_only:
        for entry in read(args.output/'manifest.json')['files']:
            if sha(args.output/entry['file']) != entry['sha256']:
                raise ValueError('A packaged file changed: '+entry['file'])
        result = audit(args.output)
        if result != read(args.output/'audit.json'):
            raise ValueError('Saved audit changed.')
    else:
        result = package(args)
    print(json.dumps(result))
