"""Build a separately declared spoon release-label candidate for physical replay."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from simulation_lab.scene import build_scene
from simulation_lab.storage import require_space


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def smooth(values):
    values = np.clip(values,0.,1.)
    return values*values*(3.-2.*values)


def run(args):
    if args.output.exists():
        raise FileExistsError('Preserve previous label candidates.')
    protocol = json.loads(args.protocol.read_text())
    source = ROOT/protocol['source']
    if sha(source/'retrieval.npz') != protocol['source_sha256']:
        raise ValueError('Declared original training input changed.')
    if sha(ROOT/protocol['baseline_spoon']/'primitive.safetensors') != protocol['baseline_spoon_sha256']:
        raise ValueError('Declared baseline model changed.')
    require_space(args.output,64*1024**2)
    args.output.mkdir(parents=True)
    with np.load(source/'retrieval.npz',allow_pickle=False) as archive:
        arrays = {key:archive[key] for key in archive.files}
    original = arrays['actions'].copy()
    info = json.loads((source/'retrieval.json').read_text())
    if len(info['episodes']) != protocol['budget']['modified_training_episodes']:
        raise ValueError('The declared episode count changed.')
    limits, cursor = {}, 0
    for stage,duration in info['durations_s'].items():
        count = round(duration*20)
        limits[stage] = (cursor,cursor+count)
        cursor += count
    xml,_ = build_scene(seed=42,scenario='dinner',dinner_preset='task')
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    point = np.array([-.003,0.,-.100])

    def tool(q):
        data.qpos[:5] = q
        mujoco.mj_kinematics(model,data)
        hand = data.body('left_gripper')
        return hand.xpos+hand.xmat.reshape(3,3)@point

    records = []
    for episode,(begin,end) in zip(info['episodes'],arrays['bounds']):
        require_space(args.output,8*1024**2)
        if end-begin != cursor:
            raise ValueError('Episode is not aligned to the declared stage durations.')
        actions = arrays['actions'][begin:end]
        old = original[begin:end]
        lo,hi = limits['lower']
        z = np.array([tool(q[:5])[2] for q in old[lo:hi]])
        target_z = z[-1]+protocol['clearance_m']
        intersections = [i for i in range(len(z)-1) if z[i] >= target_z >= z[i+1] and z[i] != z[i+1]]
        if not intersections:
            raise ValueError('The lowering path cannot supply declared clearance: '+episode)
        index = intersections[0]
        cut = index+(z[index]-target_z)/(z[index]-z[index+1])
        sample = np.minimum(np.arange(hi-lo),cut)
        actions[lo:hi,:5] = np.column_stack([np.interp(sample,np.arange(hi-lo),old[lo:hi,j]) for j in range(5)])
        raised = actions[hi-1,:5].copy()
        release_a,release_b = limits['release']
        delta = raised.astype(float)-old[release_a,:5]
        seconds = np.arange(release_b-release_a)/20
        duration = seconds[-1]
        hold = protocol['stationary_open_seconds']
        sample = np.clip((seconds-hold)/(duration-hold),0.,1.)*(len(seconds)-1)
        actions[release_a:release_b,:5] = np.column_stack([np.interp(sample,np.arange(len(seconds)),old[release_a:release_b,j]) for j in range(5)])+delta
        opening = smooth(seconds/protocol['opening_seconds'])
        closed,opened = old[hi-1,5], old[release_a:release_b,5].max()
        actions[release_a:release_b,5] = closed+(opened-closed)*opening
        retract_a,retract_b = limits['retract']
        taper = 1-smooth(np.linspace(0.,1.,retract_b-retract_a))
        actions[retract_a:retract_b,:5] = old[retract_a:retract_b,:5]+taper[:,None]*delta
        changed = np.max(np.abs(actions[:,:5]-old[:,:5]))
        if changed > protocol['maximum_active_joint_label_change_rad']:
            raise ValueError('The declared motor-label change limit is exceeded: '+episode)
        outside = np.ones(len(actions),dtype=bool)
        outside[lo:hi] = False
        outside[release_a:release_b] = False
        outside[retract_a:retract_b] = False
        if not np.array_equal(actions[outside],old[outside]) or not np.array_equal(actions[:,6:],old[:,6:]):
            raise ValueError('The transformation changed undeclared stages or the inactive arm.')
        records.append({'episode':episode,'lower_cut_endpoint':float(cut),'target_clearance_m':protocol['clearance_m'],
                        'actual_fk_clearance_m':float(tool(raised)[2]-z[-1]),'maximum_active_joint_change_rad':float(changed),
                        'joint_offset_rad':delta.tolist(),'opening_target_rad':float(opened)})
    require_space(args.output,64*1024**2)
    with (args.output/'retrieval.npz').open('xb') as stream:
        np.savez_compressed(stream,**arrays)
    info['action_preprocessing'] += ' Separately declared spoon-release-v1: 3 mm lowering clearance, stationary opening before lateral retreat, tapered retract offset; requires a fresh complete physical replay gate.'
    info['release_protocol_sha256'] = sha(args.protocol)
    info['parent_training_input_sha256'] = protocol['source_sha256']
    for name,value in [('retrieval.json',info),('transformation.json',{'schema':protocol['schema'],'protocol_sha256':sha(args.protocol),
                       'source_sha256':protocol['source_sha256'],'candidate_sha256':sha(args.output/'retrieval.npz'),
                       'script_sha256':sha(Path(__file__)),'records':records,'physical_replay_complete':False,
                       'training_started':False,'originals_modified':False})]:
        require_space(args.output,1024**2)
        (args.output/name).write_bytes((json.dumps(value,indent=2)+'\n').encode())
    print(json.dumps({'episodes':len(records),'source_sha256':protocol['source_sha256'],'candidate_sha256':sha(args.output/'retrieval.npz'),
                      'maximum_active_joint_change_rad':max(r['maximum_active_joint_change_rad'] for r in records),
                      'fk_clearance_range_m':[min(r['actual_fk_clearance_m'] for r in records),max(r['actual_fk_clearance_m'] for r in records)],
                      'next_gate':'All modified physical action replays must pass before training.'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'docs/robotics/experiments/spoon-release-v1.json')
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
