"""Build compact initial-image imitation inputs without copying image sequences."""
import argparse,gzip,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from simulation_lab.primitive_policy import primitive_image_vector
from simulation_lab.storage import require_space

CAMERAS=['overhead','left_wrist_cam','right_wrist_cam']

def run(a):
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    require_space(out,128*1024**2)
    rows=[];vectors=[];actions=[];seconds=[];bounds=[];cursor=0
    folders=sorted(Path(a.dataset).glob('train-*'))
    folders += [Path('.run/bottle-matched-nvidia')/n for n in ['upright-01','upright-04','sideways-01','sideways-03','sideways-11']]
    for folder in folders:
        manifest=json.loads((folder/'manifest.json').read_text())
        if not manifest.get('training_eligible'):continue
        compact=(folder/'trajectory.npz').is_file()
        paths=[folder/(c+'.png') if compact else folder/'images'/('000000_'+c+'.png') for c in CAMERAS]
        vectors.append(primitive_image_vector([np.array(Image.open(p).convert('RGB')) for p in paths],a.preprocess))
        if compact:
            with np.load(folder/'trajectory.npz',allow_pickle=False) as z:target=z['actions20'];t=z['action_indices']/200
        else:
            with gzip.open(folder/'actions.jsonl.gz','rt') as f:raw=np.array([json.loads(line)['target'] for line in f])
            indices=np.unique(np.r_[np.arange(0,len(raw),10),len(raw)-1]);target=raw[indices];t=indices/200
        actions.append(target);seconds.append(t);bounds.append([cursor,cursor+len(t)]);cursor+=len(t)
        rows.append({'source':folder.name,'folder':folder.as_posix(),'frames':len(t),
                     'manifest_sha256':hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest()})
    if len(vectors)<33:raise ValueError('Need at least 33 verified initial views for this 32-component encoder.')
    pixels=np.array(vectors,dtype='float32');mean=pixels.mean(0)
    _,_,vt=np.linalg.svd(pixels-mean,full_matrices=False);components=vt[:32].astype('float32');scale=np.ones(32,dtype='float32')
    projected=(pixels-mean)@components.T
    visual=np.concatenate([np.repeat(v[None],len(t),axis=0) for v,t in zip(projected,seconds)])
    errors=np.mean(((pixels-mean)-projected@components)**2,axis=1)
    out.mkdir(parents=True)
    np.savez_compressed(out/'retrieval.npz',mean=mean,components=components,scale=scale,visual=visual,
                        actions=np.concatenate(actions).astype('float32'),seconds=np.concatenate(seconds).astype('float32'),bounds=np.array(bounds))
    meta={'episodes':[r['source'] for r in rows],'lineage':rows,'visual_preprocess':a.preprocess,
          'reconstruction_check':'initial_only','reconstruction_limit':max(.002,float(errors.max())*4),
          'scope':'Initial-image-conditioned neural trajectory training; this NPZ is not an inference retrieval model.',
          'training_reconstruction_mse_max':float(errors.max()),'images_stored':len(vectors)*3,'action_endpoints':cursor}
    (out/'retrieval.json').write_text(json.dumps(meta,indent=2));print(json.dumps({'episodes':len(rows),'endpoints':cursor,'dataset_bytes':(out/'retrieval.npz').stat().st_size}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',default='.run/bottle-wide-compact-v1');p.add_argument('--output',required=True)
    p.add_argument('--preprocess',choices=['raw','brightness_normalized'],default='brightness_normalized');run(p.parse_args())
