"""Fit a small nonparametric observation/action baseline from existing recordings."""
import argparse,gzip,hashlib,json,os,sys
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from simulation_lab.dataset import load_policy_episode
from simulation_lab.retrieval_policy import image_vector
from simulation_lab.storage import require_space,GIB

def fit(a):
    if not np.isfinite(a.progress_weight) or a.progress_weight<0:raise ValueError('Progress weight must be finite and nonnegative.')
    root=Path(a.source);out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    require_space(out,int(.1*GIB));out.mkdir(parents=True)
    lineage=json.loads(Path(a.lineage).read_text());states=[];actions=[];visual=[];bounds=[];names=[];alignment=[]
    for row in lineage['episodes']:
        name=row['source'];start=len(states);names.append(name)
        folder=Path(row['folder']) if 'folder' in row else root/name
        if row.get('source_manifest_sha256') and hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest()!=row['source_manifest_sha256']:
            raise ValueError('Source manifest changed after lineage creation: '+name)
        samples=load_policy_episode(folder)
        with gzip.open(folder/'actions.jsonl.gz','rt') as stream:
            phases=[json.loads(line)['stage'] for line in stream][::10]
        alignment.append(hashlib.sha256(json.dumps(phases).encode()).hexdigest())
        if len(samples)!=row['frames']:raise ValueError('Lineage frame count differs from episode: '+name)
        for sample in samples:
            states.append(np.r_[sample['joint_position'],sample['joint_velocity']]);actions.append(sample['action'])
            images=[]
            for path in sample['images'].values():
                with Image.open(path) as im:images.append(np.asarray(im.convert('RGB')))
            visual.append(image_vector(images))
        bounds.append([start,len(states)]);print(name,len(states)-start,flush=True)
    pixels=np.asarray(visual);mean=pixels.mean(0);pixels-=mean
    # Deterministic randomized PCA, implemented with NumPy to avoid a new package.
    rng=np.random.default_rng(0);omega=rng.standard_normal((pixels.shape[1],48)).astype(np.float32)
    q=np.linalg.qr(pixels@omega,mode='reduced')[0]
    for _ in range(2):q=np.linalg.qr(pixels@(pixels.T@q),mode='reduced')[0]
    _,singular,vt=np.linalg.svd(q.T@pixels,full_matrices=False)
    components=vt[:32];features=pixels@components.T
    scale=np.maximum(singular[:32]/np.sqrt(len(pixels)-1),.001)
    residual=np.mean((pixels-features@components)**2,axis=1)
    reconstruction_limit=max(.002,float(np.quantile(residual,.999))*4)
    np.savez_compressed(out/'retrieval.npz',states=np.asarray(states),actions=np.asarray(actions),
        visual=features/scale,mean=mean,components=components,scale=scale,bounds=np.asarray(bounds),alignment=np.asarray(alignment))
    (out/'retrieval.json').write_text(json.dumps({'episodes':names,'features':'PCA of three 32x24 RGB views, joint positions and velocities','reconstruction_limit':reconstruction_limit,'training_reconstruction_p999':float(np.quantile(residual,.999)),'temporal_prior':'bounded search with weak four-frame tie-breaker; no simulator clock','scope':'familiar-start nonparametric baseline, not ACT or a VLA'},indent=2))
    (out/'run.json').write_text(json.dumps({'dataset':lineage,'policy_type':'retrieval'},indent=2))
    metadata=json.loads((out/'retrieval.json').read_text());metadata['progress_weight']=a.progress_weight
    metadata['blend_neighbors']=a.blend_neighbors
    (out/'retrieval.json').write_text(json.dumps(metadata,indent=2))
    (out/'talos_normalization.json').write_text(json.dumps({'observation.state':{'mean':[0.]*24,'std':[1.]*24},'action':{'mean':[0.]*12,'std':[1.]*12},'images':{'mean':[0.]*3,'std':[1.]*3}}))
    print('Saved',out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='.run/bottle-matched-nvidia');p.add_argument('--lineage',default='.run/act-matched-nvidia-fit5/talos_lineage.json');p.add_argument('--output',required=True)
    p.add_argument('--progress-weight',type=float,default=1e-8)
    p.add_argument('--blend-neighbors',type=int,choices=[1,2],default=1);fit(p.parse_args())
