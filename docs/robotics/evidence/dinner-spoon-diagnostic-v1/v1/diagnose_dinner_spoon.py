"""Compare the preserved spoon fit and exposed physical grasp/placement offsets."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from PIL import Image, ImageDraw
import torch
from safetensors.torch import load_file
from simulation_lab.dinner_vision import dinner_features
from simulation_lab.policy_cameras import camera_argument
from simulation_lab.primitive_policy import PrimitiveNet
from simulation_lab.rgb_servo_motor import RobotGeometry
from simulation_lab.scene import build_scene
from simulation_lab.storage import require_space


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_file(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def quantiles(values):
    return {'median':float(np.median(values)), 'p95':float(np.quantile(values,.95)), 'maximum':float(np.max(values))}


def run(args):
    if args.output.exists():
        raise FileExistsError('Preserve the existing diagnosis.')
    protocol = read(args.protocol)
    require_space(args.output, 64*1024**2)
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    torch.set_num_threads(2)
    source = ROOT/protocol['training_input']
    inputs, info = array_file(source/'retrieval.npz'), read(source/'retrieval.json')
    if sha(source/'retrieval.npz') != protocol['training_input_sha256']:
        raise ValueError('The declared training input changed.')
    gate = read(source/'input-replay.json')
    if not gate['passed'] or gate['source_sha256'] != sha(source/'retrieval.npz'):
        raise ValueError('Training input differs from its preserved physical replay.')
    xml, _ = build_scene(seed=42, scenario='dinner', dinner_preset='task')
    model = mujoco.MjModel.from_xml_string(xml)
    geometry = RobotGeometry(model)
    truth_tool = np.asarray([geometry.pose('left',q[:5])[0] for q in inputs['actions']])
    stage_ends = np.cumsum(list(info['durations_s'].values()))
    stage_names = list(info['durations_s'])
    stages = np.asarray(stage_names)[np.minimum(np.searchsorted(stage_ends,inputs['seconds'],side='right'),len(stage_names)-1)]
    fit = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for label, path in [('baseline',protocol['baseline']), ('existing_lbfgs_comparison',protocol['comparison'])]:
        folder = ROOT/path
        expected_hash = protocol['baseline_sha256'] if label == 'baseline' else protocol['comparison_sha256']
        if sha(folder/'primitive.safetensors') != expected_hash:
            raise ValueError('A declared checkpoint changed.')
        meta = read(folder/'primitive.json')
        if meta['source_sha256'] != sha(source/'retrieval.npz'):
            raise ValueError('A comparison was trained on different data.')
        network = PrimitiveNet().to(device).eval()
        network.load_state_dict(load_file(str(folder/'primitive.safetensors'),device=device))
        visual = (inputs['visual']-meta['visual_mean'])/meta['visual_std']
        parts = []
        with torch.inference_mode():
            for begin in range(0,len(visual),2048):
                predictions = network(torch.tensor(visual[begin:begin+2048],dtype=torch.float32,device=device),
                                      torch.tensor(inputs['seconds'][begin:begin+2048],dtype=torch.float32,device=device)).cpu().numpy()
                parts.append(predictions*np.asarray(meta['action_std'])+np.asarray(meta['action_mean']))
        predicted = np.concatenate(parts)
        joints = np.max(np.abs(predicted[:,:5]-inputs['actions'][:,:5]),axis=1)
        tools = np.asarray([geometry.pose('left',q[:5])[0] for q in predicted])
        errors = np.linalg.norm(tools-truth_tool,axis=1)*1000
        episodes = []
        for index,(begin,end) in enumerate(inputs['bounds']):
            episodes.append({'episode':info['episodes'][index], 'stages':{
                stage:{'tool_error_mm':quantiles(errors[begin:end][stages[begin:end]==stage]),
                       'max_active_joint_error_rad':quantiles(joints[begin:end][stages[begin:end]==stage])} for stage in stage_names}})
        fit[label] = {'checkpoint_sha256':sha(folder/'primitive.safetensors'), 'source_sha256':meta['source_sha256'],
                      'endpoints':len(visual), 'tool_error_mm':quantiles(errors), 'max_active_joint_error_rad':quantiles(joints),
                      'stages':{stage:{'tool_error_mm':quantiles(errors[stages==stage]),
                                       'max_active_joint_error_rad':quantiles(joints[stages==stage])} for stage in stage_names},
                      'episodes':episodes}
    baseline = read(ROOT/protocol['baseline']/'primitive.json')
    raw_visual = inputs['visual'][inputs['bounds'][:,0]]
    traces = []
    for seed in protocol['exposed_sequence_seeds']:
        if time.perf_counter()-started > protocol['budget']['maximum_wall_minutes']*60:
            raise TimeoutError('Diagnostic time budget reached.')
        folder = ROOT/protocol['exposed_sequence_evidence']/str(seed)
        retained = read(folder/'report.json')
        for name,digest in retained['frozen_inputs']['source_sha256'].items():
            if sha(ROOT/name) != digest:
                raise ValueError('An evaluated runtime source changed: '+name)
        states = array_file(folder/'states.npz')
        indices = np.flatnonzero(states['stage']=='spoon')
        model = mujoco.MjModel.from_xml_path(str((folder/'scene.xml').resolve()))
        model.vis.quality.offsamples = 0
        data = mujoco.MjData(model)
        origin = float(states['time'][indices[0]])

        def load(index):
            data.qpos[:] = states['qpos'][index]
            data.qvel[:] = states['qvel'][index]
            data.ctrl[:] = states['targets'][index]
            data.time = states['time'][index]
            mujoco.mj_forward(model,data)

        load(indices[0])
        renderer = mujoco.Renderer(model,width=320,height=240)
        option = mujoco.MjvOption()
        option.geomgroup[3:] = 0
        images = []
        try:
            for name in baseline['cameras']:
                renderer.update_scene(data,camera=camera_argument(name),scene_option=option)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                images.append(renderer.render().copy())
        finally:
            renderer.close()
        features = dinner_features(images,'spoon')
        normalized = (features-np.asarray(baseline['visual_mean']))/np.asarray(baseline['visual_std'])
        training_distance = np.linalg.norm((raw_visual-features)/np.asarray(baseline['visual_std']),axis=1)
        canvas = Image.new('RGB',(960,275),'#0d1c2b')
        for j,pixels in enumerate(images):
            canvas.paste(Image.fromarray(pixels),(320*j,0))
        ImageDraw.Draw(canvas).text((10,247),f'{seed}: initial spoon frame; exposed diagnostic only',fill='white')
        require_space(args.output,1024**2)
        canvas.save(args.output/f'{seed}-initial.png')
        spoon = next(row for row in retained['task']['results'] if row['skill']=='spoon')
        waits = spoon['policy_details']['tracking_guard_wait_calls']
        if waits:
            raise ValueError('Stage-time diagnostic requires the declared zero-wait traces.')
        probes = []
        for stage in ('close','hold','lower','release','park'):
            end_time = stage_ends[stage_names.index(stage)]-.1
            index = int(indices[np.argmin(abs(states['time'][indices]-origin-end_time))])
            load(index)
            hand = data.body('left_gripper')
            rotation = hand.xmat.reshape(3,3)
            tool = hand.xpos+rotation@np.array([.003,0.,-.092])
            body = data.body('spoon')
            object_rotation = body.xmat.reshape(3,3)
            relative = object_rotation.T@(tool-body.xpos)
            probes.append({'stage_time_reference':stage, 'frame_index':index, 'skill_seconds':float(data.time-origin),
                           'tool_world_m':tool.tolist(), 'object_world_m':body.xpos.tolist(),
                           'tool_in_object_frame_mm':(relative*1000).tolist(),
                           'object_yaw_rad':float(np.arctan2(object_rotation[1,0],object_rotation[0,0]))})
        traces.append({'seed':seed, 'status':spoon['status'], 'metrics':spoon['metrics'],
                       'source_hashes':{name:sha(folder/name) for name in ('report.json','states.npz','scene.xml')},
                       'initial_rgb_features':features[:4].tolist(), 'normalized_features':normalized[:4].tolist(),
                       'nearest_training_episode':info['episodes'][int(training_distance.argmin())],
                       'nearest_standardized_feature_distance':float(training_distance.min()),
                       'probes':probes, 'tracking_guard_wait_calls':waits})
    report = {'schema':protocol['schema'], 'protocol_sha256':sha(args.protocol), 'script_sha256':sha(Path(__file__)),
              'parent_git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'training_input_sha256':sha(source/'retrieval.npz'), 'fit':fit, 'exposed_traces':traces,
              'wall_seconds':time.perf_counter()-started, 'training_steps':0, 'new_physical_trials':0,
              'scope':'Training fit and previously exposed state diagnosis only. No candidate selection or new generalization evidence.'}
    require_space(args.output,8*1024**2)
    (args.output/'diagnostic.json').write_bytes((json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps({'fit':{name:{k:v for k,v in value.items() if k!='episodes'} for name,value in fit.items()},
                      'traces':[{k:v for k,v in row.items() if k not in ('source_hashes','metrics')} for row in traces],
                      'wall_seconds':report['wall_seconds']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'docs/robotics/experiments/dinner-spoon-diagnostic-v1.json')
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
