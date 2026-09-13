"""Fit train-only initial RGB encoders and aligned per-skill imitation inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from PIL import Image
from simulation_lab.primitive_policy import primitive_image_vector
from simulation_lab.dinner_autonomy import SKILLS
from simulation_lab.storage import require_space
from scripts.collect_dinner_learning import save_json
from scripts.prepare_bottle_data import STATE
from simulation_lab.policy_cameras import camera_argument
from simulation_lab.dinner_vision import dinner_features
from simulation_lab.bottle_vision import bottle_features

CAMERAS = ['overhead', 'left_wrist_cam', 'right_wrist_cam']
ORDER = ['approach', 'descend', 'close', 'lift', 'hold', 'clearance', 'transit',
         'rotate', 'align', 'lower', 'release', 'retract', 'park', 'verify']


def initial_images(folder,cameras=CAMERAS):
    model = mujoco.MjModel.from_xml_path(str(folder.parent/'scene.xml'))
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, np.load(folder/'initial-integration-state.npy', allow_pickle=False), STATE)
    mujoco.mj_forward(model, data)
    model.vis.quality.offsamples = 0
    renderer = mujoco.Renderer(model, height=240, width=320)
    option = mujoco.MjvOption(); option.geomgroup[3:] = 0
    images = []
    try:
        for camera in cameras:
            renderer.update_scene(data, camera=camera_argument(camera), scene_option=option)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            images.append(renderer.render().copy())
    finally: renderer.close()
    return images


def prepare(dataset, output, skill, geometry=False, additional_datasets=(), sustain_grip=False):
    require_space(output, 256*1024**2)
    if output.exists(): raise FileExistsError(output)
    rows, excluded = [], []
    folders=[folder for root in (dataset,*additional_datasets) for folder in sorted(root.glob('seed-*/'+skill))]
    names=[folder.parent.name for folder in folders]
    if len(names)!=len(set(names)):raise ValueError('Duplicate seed across training datasets.')
    for folder in folders:
        manifest = json.loads((folder/'manifest.json').read_text())
        if not manifest.get('training_eligible') or not manifest.get('replayed_saved_float32_endpoints'):
            excluded.append({'episode': folder.parent.name, 'reason': 'No passing exact saved-endpoint replay'})
            continue
        with np.load(folder/'trajectory.npz', allow_pickle=False) as z:
            trajectory = {k: z[k] for k in z.files}
        if set(trajectory['stages'])-set(ORDER):
            excluded.append({'episode': folder.parent.name, 'reason': 'Different stage family (e.g. retry) requires a separate model'})
            continue
        cameras=['overhead','table_left','table_right'] if geometry else CAMERAS
        images = initial_images(folder,cameras)
        rows.append((folder, manifest, trajectory, images))
    if not rows: raise ValueError('No eligible demonstrations for '+skill)
    kinds={m['skill'] for _,m,_,_ in rows}
    if len(kinds)!=1:raise ValueError('Do not mix object types in one primitive.')
    kind=next(iter(kinds))
    arms = {m['arm'] for _, m, _, _ in rows}
    if len(arms) != 1: raise ValueError('Separate models required for different arms.')
    stages = [s for s in ORDER if any(s in t['stages'] for _, _, t, _ in rows)]
    if any(set(t['stages']) != set(stages) for _, _, t, _ in rows):
        raise ValueError('Do not align incompatible stage families.')
    durations = {s: max(int(np.sum(t['stages'] == s)) for _, _, t, _ in rows) for s in stages}
    pixels = np.array([(bottle_features(images) if kind=='bottle' else dinner_features(images,kind)) if geometry else primitive_image_vector(images, 'brightness_normalized') for _, _, _, images in rows])
    mean = pixels.mean(0)
    # SVD is train-only. Padding keeps the existing small OpenVINO network shape.
    _, singular, vt = np.linalg.svd(pixels-mean, full_matrices=False)
    n = min(16, max(0, len(rows)-1))
    components = np.zeros((32, pixels.shape[1]), dtype=np.float32)
    components[:n] = vt[:n]
    if geometry:
        mean=np.zeros(32,dtype=np.float32);components=np.eye(32,dtype=np.float32);n=4 if kind in ('fork','spoon') else 2
    visual = (pixels-mean)@components.T
    actions, seconds, visuals, bounds = [], [], [], []
    cursor = 0
    for row_index, (_, _, t, _) in enumerate(rows):
        parts, times, offset = [], [], 0.
        for stage in stages:
            ix = np.flatnonzero(t['stages'] == stage)
            count = durations[stage]
            sample = np.linspace(0, len(ix), count, endpoint=False)
            ix = np.r_[ix, min(ix[-1]+1, len(t['actions20'])-1)]
            part=np.column_stack([np.interp(sample, np.arange(len(ix)), t['actions20'][ix, j]) for j in range(12)])
            if sustain_grip and stage in ('lift','hold','clearance','transit','rotate','align','lower'):
                # Offline label repair: maintain the demonstrated closed target
                # through carry transitions. It must pass physical input replay
                # before fitting; these stage labels never reach inference.
                part[:,5 if next(iter(arms))=='left' else 11]=-.17
            parts.append(part)
            times.append(offset+np.arange(count)/20)
            offset += count/20
        act, time = np.concatenate(parts), np.concatenate(times)
        actions.append(act); seconds.append(time)
        visuals.append(np.repeat(visual[row_index:row_index+1], len(time), axis=0))
        bounds.append([cursor, cursor+len(time)]); cursor += len(time)
    output.mkdir(parents=True)
    require_space(output, 256*1024**2)
    np.savez_compressed(output/'retrieval.npz', mean=mean, components=components, scale=np.ones(32, dtype=np.float32),
                        visual=np.concatenate(visuals).astype('float32'), actions=np.concatenate(actions).astype('float32'),
                        seconds=np.concatenate(seconds).astype('float32'), bounds=np.array(bounds))
    residual = np.mean(((pixels-mean)-visual@components)**2, axis=1)
    meta = {'skill': kind, 'logical_skill':skill, 'episodes': [f.parent.name for f, _, _, _ in rows], 'excluded': excluded,
            'arm_offset': 0 if next(iter(arms)) == 'left' else 6,
            'gripper_cap_nm': rows[0][1]['gripper_cap_nm'],
            'visual_encoder': ('bottle_rgb_geometry' if kind=='bottle' else 'dinner_rgb_geometry') if geometry else 'camera_pca', 'visual_preprocess': 'brightness_normalized',
            'cameras':cameras,
            'reconstruction_check': 'initial_only', 'reconstruction_limit': max(.002, float(residual.max())*4),
            'visual_components': n, 'durations_s': {s: n/20 for s, n in durations.items()},
            'lineage': [{'episode': f.parent.name, 'manifest_sha256': hashlib.sha256((f/'manifest.json').read_bytes()).hexdigest()} for f, _, _, _ in rows],
            'scope': 'Initial three-camera RGB-conditioned neural skill. Offline stage alignment; no runtime object pose or stage input.'}
    if geometry:
        margin=np.zeros(32,dtype=np.float32);margin[:2]=[1/320,1/240]
        if n==4:margin[2:4]=.05
        meta['feature_support']={'min':(pixels.min(0)-margin).tolist(),'max':(pixels.max(0)+margin).tolist(),
                'description':'Train-only feature range with one pixel centroid margin; not a workspace-coverage guarantee.'}
    if sustain_grip:
        meta['action_preprocessing']='Offline closed-gripper target (-0.17 rad) sustained during carry stages; physical aligned-input replay required before training.'
    if skill.startswith(('relay_','reverse_')):
        destinations=[]
        for _,m,_,_ in rows:
            target=next((t['position_m'] for t in m['layout']['targets'] if t['object_id']==kind),None)
            if target is None:
                original=next(o['initial_position_m'] for o in m['layout']['objects'] if o['id']==kind)
                target=(np.array(original)+[.070,-.040,0]).tolist()
            destinations.append(target)
        if not np.allclose(destinations,destinations[0],atol=1e-9):raise ValueError('Relay leg requires one fixed trained destination.')
        meta['trained_destination_m']=destinations[0]
    save_json(output/'retrieval.json', meta)
    print(json.dumps({'skill': skill, 'episodes': len(rows), 'excluded': excluded, 'endpoints': cursor}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--additional-dataset', type=Path, action='append', default=[])
    p.add_argument('--skills', default=','.join(SKILLS))
    p.add_argument('--geometry',action='store_true',help='Use explicit RGB geometry and fixed workspace cameras for plate/mug/cutlery.')
    p.add_argument('--sustain-grip',action='store_true',help='Repair carry-transition gripper labels offline; requires separate physical aligned-input verification.')
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    for skill in args.skills.split(','):
        if skill not in (*SKILLS,'relay_bottle_left','relay_bottle_right','reverse_bottle_left','reverse_bottle_right'): raise ValueError('Unknown skill')
        prepare(args.dataset, args.output/skill, skill,args.geometry,args.additional_dataset,args.sustain_grip)
