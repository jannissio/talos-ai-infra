"""Small, disclosed neighborhood probes; never replace the reserved pose test set."""
import argparse,json,shutil,subprocess,sys,xml.etree.ElementTree as ET
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco,numpy as np
from scripts.prepare_bottle_data import STATE

ROOT=Path(__file__).resolve().parents[1]
PROBES=[('x-plus-2mm',.002,0,1.),('x-minus-2mm',-.002,0,1.),
        ('y-plus-2mm',0,.002,1.),('y-minus-2mm',0,-.002,1.),('light-90pct',0,0,.9)]

def main(a):
    source=ROOT/'.run/bottle-matched-nvidia/upright-01';out=Path(a.output).resolve()
    if out.exists():raise FileExistsError(out)
    if out.parent!=ROOT/'.run':raise ValueError('Keep the same depth for relative mesh paths.')
    out.mkdir(parents=True);results=[]
    for name,dx,dy,light in PROBES:
        folder=out/name;folder.mkdir();tree=ET.parse(source/'scene.xml')
        if light!=1:
            for node in list(tree.iter('light'))+list(tree.iter('headlight')):
                if node.get('diffuse'):node.set('diffuse',' '.join(str(float(x)*light) for x in node.get('diffuse').split()))
        tree.write(folder/'scene.xml',encoding='unicode')
        model=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'));data=mujoco.MjData(model)
        mujoco.mj_setState(model,data,np.load(source/'initial-integration-state.npy',allow_pickle=False),STATE)
        address=model.joint('bottle_free').qposadr[0];data.qpos[address:address+2]+=[dx,dy]
        mujoco.mj_forward(model,data)
        state=np.empty(mujoco.mj_stateSize(model,STATE));mujoco.mj_getState(model,data,state,STATE)
        np.save(folder/'initial-integration-state.npy',state,allow_pickle=False)
        manifest=json.loads((source/'manifest.json').read_text());manifest['training_eligible']=False
        manifest['probe']={'source':'upright-01','initial_bottle_xy_offset_m':[dx,dy],'diffuse_light_scale':light}
        manifest['images']={'status':'not_captured'};manifest.pop('outcome',None)
        (folder/'manifest.json').write_text(json.dumps(manifest,indent=2))
        with (folder/'evaluation.log').open('w') as log:
            subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_act.py'),'--checkpoint',str(Path(a.checkpoint).resolve()),
                '--episode',str(folder),'--output',str(folder/'result.json'),'--device','cpu'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
        result=json.loads((folder/'result.json').read_text());results.append(result)
        print(name,result['success'],result['reason'],flush=True)
    (out/'summary.json').write_text(json.dumps({'scope':'Five small deterministic development probes near upright-01; not reserved test poses','successes':sum(r['success'] for r in results),'trials':len(results),'results':results},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);main(p.parse_args())
