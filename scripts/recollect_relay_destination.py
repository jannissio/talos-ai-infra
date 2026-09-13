"""Re-demonstrate a relay leg from a saved physical start at a declared fixed goal.

Old demonstrations remain unchanged. The new teacher and exact endpoint replay
must both succeed; changing a label on the old trajectory is not sufficient.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from simulation_lab.storage import require_space
from scripts.collect_dinner_learning import collect_episode,save_json
from scripts.prepare_bottle_data import STATE


def run(a):
    protocol=json.loads(a.protocol.read_text(encoding='utf-8'))
    seeds=protocol['training_seeds'];skill=protocol['skill'];destination=protocol['destination_xy']
    if skill!='reverse_bottle_left' or len(seeds)!=len(set(seeds)) or not 1<=len(seeds)<=32:
        raise ValueError('Expected unique declared reverse-left training seeds.')
    if np.shape(destination)!=(2,) or not np.isfinite(destination).all() or np.any(np.abs(destination)>.3):
        raise ValueError('Invalid declared destination.')
    if a.output.exists():raise FileExistsError(a.output)
    require_space(a.output,len(seeds)*96*1024**2);a.output.mkdir(parents=True)
    save_json(a.output/'protocol.json',protocol)
    rows=[]
    for seed in seeds:
        matches=[Path(root)/f'seed-{seed}' for root in protocol['source_roots'] if (Path(root)/f'seed-{seed}'/skill/'initial-integration-state.npy').is_file()]
        if len(matches)!=1:raise ValueError('Expected one physical starting episode for seed '+str(seed))
        source=matches[0];summary=json.loads((source/'summary.json').read_text())
        if not summary['legs'][0]['training_eligible']:raise ValueError('Preceding right-arm release was not verified.')
        folder=a.output/f'seed-{seed}';require_space(folder,96*1024**2);folder.mkdir()
        model=mujoco.MjModel.from_xml_path(str(source/'scene.xml'));model.vis.quality.offsamples=0;data=mujoco.MjData(model)
        mujoco.mj_setState(model,data,np.load(source/skill/'initial-integration-state.npy',allow_pickle=False),STATE)
        mujoco.mj_forward(model,data)
        manifest=json.loads((source/skill/'manifest.json').read_text());layout=deepcopy(manifest['layout'])
        layout['targets']=[t for t in layout['targets'] if t['object_id']!='bottle']
        layout['targets'].append({'id':skill+'_destination','object_id':'bottle','position_m':[*destination,layout['table_z']],'radius_m':.012})
        scene=ET.fromstring((source/'scene.xml').read_text());compiler=scene.find('compiler')
        meshdir=(source/Path(compiler.get('meshdir'))).resolve()
        compiler.set('meshdir',os.path.relpath(meshdir,folder).replace('\\','/'))
        (folder/'scene.xml').write_text(ET.tostring(scene,encoding='unicode'),encoding='utf-8')
        result=collect_episode(model,data,layout,'bottle',folder/skill,side='left')
        provenance={'source_initial_state_sha256':hashlib.sha256((source/skill/'initial-integration-state.npy').read_bytes()).hexdigest(),
                    'new_destination_xy':destination,'result':result,'seed':seed}
        save_json(folder/'summary.json',provenance);rows.append(provenance)
        save_json(a.output/'summary.json',{'attempts':rows,'passed':sum(r['result']['training_eligible'] for r in rows)})
        print(json.dumps({'seed':seed,'teacher':result['teacher_status'],'replay':result['training_eligible'],'message':result['message']}),flush=True)
    return int(not all(r['result']['training_eligible'] for r in rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    raise SystemExit(run(p.parse_args()))
