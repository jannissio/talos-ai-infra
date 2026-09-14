"""Preserve all release-label inputs and physical replay outcomes portably."""
import argparse
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
from scripts.release_experiment import read,repository_path,sha,space,write_json


def audit(folder):
    p,info,gate,transformation = (read(folder/name) for name in ('protocol.json','retrieval.json','input-replay.json','transformation.json'))
    original = repository_path(p['source'])
    if (sha(folder/'retrieval.npz')!=gate['source_sha256'] or transformation['candidate_sha256']!=gate['source_sha256']
            or sha(folder/'protocol.json')!=transformation['protocol_sha256']
            or sha(original/'retrieval.npz')!=p['source_sha256'] or info['parent_training_input_sha256']!=p['source_sha256']):
        raise ValueError('The input, transformation or replay provenance changed.')
    rows = gate['episodes']
    if (not gate['complete'] or len(rows)!=p['budget']['modified_training_episodes']
            or [r['episode'] for r in rows]!=info['episodes'] or gate['passed']!=all(r['passed'] for r in rows)):
        raise ValueError('The complete declared input denominator is required.')
    with np.load(folder/'retrieval.npz',allow_pickle=False) as x,np.load(original/'retrieval.npz',allow_pickle=False) as y:
        if any(not np.array_equal(x[k],y[k]) for k in x.files if k!='actions'):
            raise ValueError('Undeclared input arrays changed.')
        offset = p['arm_offset']
        active,inactive = slice(offset,offset+5),slice(6,12) if offset==0 else slice(0,6)
        change = float(np.max(abs(x['actions'][:,active]-y['actions'][:,active])))
        if change>p['maximum_active_joint_label_change_rad'] or not np.array_equal(x['actions'][:,inactive],y['actions'][:,inactive]):
            raise ValueError('Action changes exceed the declared scope.')
        stage_bounds,cursor = {},0
        for name,duration in info['durations_s'].items():
            count = round(duration*20);stage_bounds[name]=(cursor,cursor+count);cursor+=count
        keep = np.ones(cursor,dtype=bool)
        for name in ('lower','release','retract'):
            a,b=stage_bounds[name];keep[a:b]=False
        for begin,end in x['bounds']:
            if end-begin!=cursor or not np.array_equal(x['actions'][begin:end][keep],y['actions'][begin:end][keep]):
                raise ValueError('Actions outside the declared stages changed.')
    provenance = read(folder/'reproduction-inputs.json')
    for name,digest in provenance['runtime_source_sha256'].items():
        if sha(folder/'source'/name)!=digest:raise ValueError('A source snapshot changed.')
    for name,digest in provenance['repository_asset_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('A simulation asset changed.')
    for row in rows:
        base = folder/'episodes'/row['episode']
        manifest = base/p['skill']/'manifest.json'
        lineage = next(v for v in info['lineage'] if v['episode']==row['episode'])
        if sha(manifest)!=lineage['manifest_sha256']:raise ValueError('An original collection manifest changed.')
        model = mujoco.MjModel.from_xml_path(str((base/'scene.xml').resolve()))
        data = mujoco.MjData(model)
        state = np.load(base/p['skill']/'initial-integration-state.npy',allow_pickle=False)
        if len(state)!=mujoco.mj_stateSize(model,STATE) or not np.isfinite(state).all():
            raise ValueError('An initial integration state is incompatible or nonfinite.')
        mujoco.mj_setState(model,data,state,STATE);mujoco.mj_forward(model,data)
        m = row['metrics']
        if row['passed'] and (row['failure'] is not None or m['placement_error_mm']>=8 or m['placement_z_error_mm']>=4
                or m['hold_verified_s']<1.49 or not m['both_arms_parked'] or m['stable_release_and_park_s']<.5-1e-8
                or m['unexpected_collisions'] or m['other_object_max_displacement_m']>.004
                or m['parked_arm_max_motion_deg']>1 or m['longest_unsupported_gap_s']>.18):
            raise ValueError('A passing action replay contradicts the unchanged physical criteria.')
    return {'episodes':len(rows),'passed':sum(r['passed'] for r in rows),'gate_passed':gate['passed'],
            'portable_initial_scenes_verified':len(rows),'maximum_active_joint_label_change_rad':change,
            'failures':[r for r in rows if not r['passed']],
            'scope':'Physical saved-action replay only; no learned-policy evaluation or promotion.'}


def package(args):
    p = read(args.protocol)
    source,output = repository_path(p['raw_root'])/'input',repository_path(p['training_package'])
    if output.exists():raise FileExistsError('Preserve the existing input package.')
    if not read(source/'input-replay.json')['complete']:raise ValueError('Wait for every input replay.')
    space(p,output,64*1024**2);output.mkdir(parents=True)
    entries = []

    def put(name,payload,transformation='byte-identical copy'):
        path = output/name
        space(p,path,len(payload)+1024**2);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:stream.write(payload)
        entries.append({'file':name,'sha256':sha(path),'bytes':len(payload),'transformation':transformation})

    for name in ('retrieval.npz','retrieval.json','input-replay.json','transformation.json'):
        put(name,(source/name).read_bytes())
    put('protocol.json',args.protocol.read_bytes())
    info = read(source/'retrieval.json')
    assets = {}
    for name in info['episodes']:
        space(p,output,4*1024**2)
        lineage = next(v for v in info['lineage'] if v['episode']==name)
        matches = [repository_path(parent)/name for parent in p['raw_episode_sources']
                   if (repository_path(parent)/name/p['skill']/'manifest.json').is_file()
                   and sha(repository_path(parent)/name/p['skill']/'manifest.json')==lineage['manifest_sha256']]
        if len(matches)!=1:raise ValueError('Require the exact original episode source: '+name)
        raw = matches[0]
        for leaf in ('manifest.json','initial-integration-state.npy'):
            put('episodes/'+name+'/'+p['skill']+'/'+leaf,(raw/p['skill']/leaf).read_bytes())
        tree = ET.fromstring((raw/'scene.xml').read_bytes())
        compiler = tree.find('compiler');meshdir = (raw/compiler.get('meshdir')).resolve()
        if not meshdir.is_relative_to(ROOT/'simulation_lab/assets'):raise ValueError('Unexpected mesh asset path.')
        for node in tree.findall('./asset/mesh'):
            path=meshdir/node.get('file');assets[path.relative_to(ROOT).as_posix()]=sha(path)
        compiler.set('meshdir',os.path.relpath(meshdir,output/'episodes'/name).replace('\\','/'))
        put('episodes/'+name+'/scene.xml',ET.tostring(tree,encoding='utf-8'),'Only meshdir remapped to the same repository assets.')
    sources = [v.relative_to(ROOT).as_posix() for v in (ROOT/'simulation_lab').glob('*.py')]
    sources += ['scripts/prepare_release_labels.py','scripts/release_experiment.py','scripts/verify_dinner_learning_inputs.py',
                'scripts/prepare_bottle_data.py','scripts/package_release_inputs.py']
    hashes = {}
    for name in sources:
        put('source/'+name,(ROOT/name).read_bytes());hashes[name]=sha(ROOT/name)
    put('reproduction-inputs.json',(json.dumps({'runtime_source_sha256':hashes,'repository_asset_sha256':assets},indent=2)+'\n').encode(),'Explicit reproduction manifest.')
    result = audit(output)
    put('audit.json',(json.dumps(result,indent=2)+'\n').encode(),'Independent provenance/action/state/physical-criterion audit.')
    write_json(output/'manifest.json',{'files':entries,'originals_modified':False,'all_episodes_retained':True})
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--verify-only',action='store_true')
    args = parser.parse_args()
    if args.verify_only:
        folder=repository_path(read(args.protocol)['training_package'])
        for entry in read(folder/'manifest.json')['files']:
            if sha(folder/entry['file'])!=entry['sha256']:raise ValueError('A packaged file changed.')
        result=audit(folder)
        if result!=read(folder/'audit.json'):raise ValueError('The independent audit changed.')
    else:result=package(args)
    print(json.dumps(result))
