"""Compare the preserved mug fit with every exposed paired-baseline mug trace."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
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
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arrays(path):
    with np.load(path,allow_pickle=False) as archive:
        return {name:archive[name] for name in archive.files}


def stats(values):
    return {'median':float(np.median(values)),'p95':float(np.quantile(values,.95)),'maximum':float(np.max(values))}


def run(args):
    if args.output.exists():
        raise FileExistsError('Preserve every existing diagnostic.')
    p = read(args.protocol)
    source,checkpoint,evidence = (ROOT/p[name] for name in ('training_input','baseline','evidence'))
    if sha(source/'retrieval.npz')!=p['training_input_sha256'] or sha(checkpoint/'primitive.safetensors')!=p['baseline_sha256']:
        raise ValueError('The declared training input or baseline changed.')
    frozen = read(evidence/'frozen-inputs.json')
    if sha(evidence/'frozen-inputs.json')!=p['evidence_freeze_sha256']:
        raise ValueError('The exposed evidence freeze changed.')
    for name,digest in frozen['inputs']['source_sha256'].items():
        if sha(ROOT/name)!=digest:
            raise ValueError('An exposed physical runtime source changed.')
    require_space(args.output,32*1024**2)
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    torch.set_num_threads(2)
    x,info,meta = arrays(source/'retrieval.npz'),read(source/'retrieval.json'),read(checkpoint/'primitive.json')
    gate = read(source/'input-replay.json')
    if not gate['passed'] or gate['source_sha256']!=p['training_input_sha256'] or meta['source_sha256']!=p['training_input_sha256']:
        raise ValueError('The baseline input lacks its complete preserved physical gate.')
    offset,point = meta['arm_offset'],np.asarray(p['grasp_point_local_m'])
    if offset!=6 or p['arm']!='right':
        raise ValueError('The declared mug arm changed.')
    xml,_ = build_scene(seed=42,scenario='dinner',dinner_preset='task')
    geometry = RobotGeometry(mujoco.MjModel.from_xml_string(xml))

    def tool(q):
        _,rotation = geometry.pose(p['arm'],q[offset:offset+5])
        return geometry.scratch.body(p['arm']+'_gripper').xpos+rotation@point

    truth = np.asarray([tool(q) for q in x['actions']])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    network = PrimitiveNet().to(device).eval()
    network.load_state_dict(load_file(str(checkpoint/'primitive.safetensors'),device=device))
    visual = (x['visual']-meta['visual_mean'])/meta['visual_std']
    predictions = []
    with torch.inference_mode():
        for begin in range(0,len(visual),2048):
            predicted = network(torch.tensor(visual[begin:begin+2048],dtype=torch.float32,device=device),
                                torch.tensor(x['seconds'][begin:begin+2048],dtype=torch.float32,device=device)).cpu().numpy()
            predictions.append(predicted*np.asarray(meta['action_std'])+np.asarray(meta['action_mean']))
    predicted = np.concatenate(predictions)
    errors = np.linalg.norm(np.asarray([tool(q) for q in predicted])-truth,axis=1)*1000
    joints = np.max(abs(predicted[:,offset:offset+5]-x['actions'][:,offset:offset+5]),axis=1)
    names = list(info['durations_s'])
    ends = np.cumsum(list(info['durations_s'].values()))
    stages = np.asarray(names)[np.minimum(np.searchsorted(ends,x['seconds'],side='right'),len(names)-1)]
    fit = {'endpoints':len(predicted),'tool_error_mm':stats(errors),'joint_error_rad':stats(joints),
           'stages':{stage:{'tool_error_mm':stats(errors[stages==stage]),'joint_error_rad':stats(joints[stages==stage])} for stage in names},
           'episodes':[{'episode':episode,'tool_error_mm':stats(errors[begin:end]),
                        'stages':{stage:stats(errors[begin:end][stages[begin:end]==stage]) for stage in names}}
                       for episode,(begin,end) in zip(info['episodes'],x['bounds'])]}
    training_visual = x['visual'][x['bounds'][:,0]]
    traces = []
    summary = read(evidence/'summary.json')
    for seed in p['exposed_seeds']:
        for preset in p['presets']:
            if time.perf_counter()-started > p['budget']['maximum_wall_minutes']*60:
                raise TimeoutError('Declared diagnostic time limit reached.')
            require_space(args.output,4*1024**2)
            name = f'{seed}-{preset}-{p["controller"]}'
            folder = evidence/name
            retained = read(folder/'report.json')
            entry = next(r for r in summary['rows'] if r['seed']==seed and r['preset']==preset and r['controller']==p['controller'])
            if sha(folder/'report.json')!=entry['report_sha256']:
                raise ValueError('An original physical report changed.')
            mug = next(r for r in retained['task']['results'] if r['skill']=='mug')
            if mug['policy_details']['tracking_guard_wait_calls']!=0:
                raise ValueError('The declared stage-time method requires zero guard waits.')
            frames = arrays(folder/'states.npz')
            indices = np.flatnonzero(frames['stage']=='mug')
            model = mujoco.MjModel.from_xml_path(str((folder/'scene.xml').resolve()))
            model.vis.quality.offsamples = 0
            data = mujoco.MjData(model)
            origin = float(frames['time'][indices[0]])

            def load(index):
                data.qpos[:] = frames['qpos'][index]
                data.qvel[:] = frames['qvel'][index]
                data.ctrl[:] = frames['targets'][index]
                data.time = frames['time'][index]
                mujoco.mj_forward(model,data)

            load(indices[0])
            renderer = mujoco.Renderer(model,width=320,height=240)
            option = mujoco.MjvOption();option.geomgroup[3:]=0
            images = []
            try:
                for camera in meta['cameras']:
                    renderer.update_scene(data,camera=camera_argument(camera),scene_option=option)
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
                    images.append(renderer.render().copy())
            finally:
                renderer.close()
            features = dinner_features(images,'mug')
            distance = np.linalg.norm((training_visual-features)/np.asarray(meta['visual_std']),axis=1)
            canvas = Image.new('RGB',(960,275),'#0d1c2b')
            for i,pixels in enumerate(images):canvas.paste(Image.fromarray(pixels),(320*i,0))
            ImageDraw.Draw(canvas).text((10,247),name+': exposed mug initial frame',fill='white')
            canvas.save(args.output/(name+'-initial.png'))
            probes = []
            for stage in p['stage_probes']:
                seconds = ends[names.index(stage)]-.1
                index = int(indices[np.argmin(abs(frames['time'][indices]-origin-seconds))])
                load(index)
                hand,obj = data.body('right_gripper'),data.body('mug')
                rotation,object_rotation = hand.xmat.reshape(3,3),obj.xmat.reshape(3,3)
                actual_tool = hand.xpos+rotation@point
                probes.append({'stage':stage,'frame_index':index,'skill_seconds':float(data.time-origin),
                    'tool_world_m':actual_tool.tolist(),'object_world_m':obj.xpos.tolist(),
                    'tool_in_object_frame_mm':(object_rotation.T@(actual_tool-obj.xpos)*1000).tolist(),
                    'placement_error_mm':float(np.linalg.norm(obj.xpos[:2]-data.geom('mug_place').xpos[:2])*1000),
                    'object_yaw_rad':float(np.arctan2(object_rotation[1,0],object_rotation[0,0]))})
            traces.append({'seed':seed,'preset':preset,'status':mug['status'],'metrics':mug['metrics'],
                'source_hashes':{n:sha(folder/n) for n in ('report.json','states.npz','scene.xml')},
                'initial_features':features.tolist(),'nearest_training_episode':info['episodes'][int(distance.argmin())],
                'nearest_standardized_feature_distance':float(distance.min()),'probes':probes})
    result = {'schema':p['schema'],'protocol_sha256':sha(args.protocol),'script_sha256':sha(Path(__file__)),
              'parent_git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'checkpoint_sha256':p['baseline_sha256'],'training_input_sha256':p['training_input_sha256'],
              'grasp_point_local_m':point.tolist(),'fit':fit,'exposed_traces':traces,
              'wall_seconds':time.perf_counter()-started,'training_steps':0,'new_physical_trials':0,'scope':p['scope']}
    require_space(args.output,8*1024**2)
    with (args.output/'diagnostic.json').open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'fit_stages':fit['stages'],'traces':[{'seed':r['seed'],'preset':r['preset'],'status':r['status'],
                     'nearest_training_distance':r['nearest_standardized_feature_distance'],
                     'placement_mm_by_stage':{v['stage']:v['placement_error_mm'] for v in r['probes']}}
                    for r in traces],'wall_seconds':result['wall_seconds']}))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'docs/robotics/experiments/dinner-mug-diagnostic-v1.json')
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
