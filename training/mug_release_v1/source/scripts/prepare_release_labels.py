"""Prepare a protocol-defined release-label intervention without changing originals."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from simulation_lab.scene import build_scene
from scripts.release_experiment import read,repository_path,sha,space,write_json


def smooth(value):
    value = np.clip(value,0.,1.)
    return value*value*(3.-2.*value)


def run(args):
    p = read(args.protocol)
    if args.output.exists():
        raise FileExistsError('Preserve the prior release-label candidate.')
    source = repository_path(p['source'])
    baseline = repository_path(p['baseline'])
    if sha(source/'retrieval.npz')!=p['source_sha256'] or sha(baseline/'primitive.safetensors')!=p['baseline_sha256']:
        raise ValueError('The declared original inputs changed.')
    info,meta = read(source/'retrieval.json'),read(baseline/'primitive.json')
    if (info['skill']!=p['skill'] or info['arm_offset']!=p['arm_offset'] or meta['arm_offset']!=p['arm_offset']
            or len(info['episodes'])!=p['budget']['modified_training_episodes']):
        raise ValueError('The declared skill, arm or episode count changed.')
    space(p,args.output,64*1024**2)
    args.output.mkdir(parents=True)
    with np.load(source/'retrieval.npz',allow_pickle=False) as saved:
        values = {name:saved[name] for name in saved.files}
    original = values['actions'].copy()
    offset = p['arm_offset']
    active,inactive = slice(offset,offset+5),slice(6,12) if offset==0 else slice(0,6)
    limits,cursor = {},0
    for stage,duration in info['durations_s'].items():
        count = round(duration*20)
        limits[stage] = cursor,cursor+count
        cursor += count
    xml,_ = build_scene(seed=42,scenario='dinner',dinner_preset='task')
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    point = np.asarray(p['grasp_point_local_m'])
    side = 'left' if offset==0 else 'right'

    def tool(q):
        data.qpos[active] = q
        mujoco.mj_kinematics(model,data)
        hand = data.body(side+'_gripper')
        return hand.xpos+hand.xmat.reshape(3,3)@point

    records = []
    for episode,(begin,end) in zip(info['episodes'],values['bounds']):
        space(p,args.output,8*1024**2)
        if end-begin!=cursor:
            raise ValueError('Episode is not aligned to the declared stage durations.')
        actions,old = values['actions'][begin:end],original[begin:end]
        lower_a,lower_b = limits['lower']
        z = np.asarray([tool(q[active])[2] for q in old[lower_a:lower_b]])
        target = z[-1]+p['clearance_m']
        intersections = [i for i in range(len(z)-1) if z[i]>=target>=z[i+1] and z[i]!=z[i+1]]
        if not intersections:
            raise ValueError('The existing lowering path cannot supply the declared clearance: '+episode)
        index = intersections[0]
        cut = index+(z[index]-target)/(z[index]-z[index+1])
        sample = np.minimum(np.arange(lower_b-lower_a),cut)
        actions[lower_a:lower_b,active] = np.column_stack([np.interp(sample,np.arange(lower_b-lower_a),old[lower_a:lower_b,j]) for j in range(offset,offset+5)])
        raised = actions[lower_b-1,active].copy()
        release_a,release_b = limits['release']
        delta = raised.astype(float)-old[release_a,active]
        seconds = np.arange(release_b-release_a)/20
        hold,duration = p['stationary_open_seconds'],seconds[-1]
        if not 0<p['opening_seconds']<=hold<duration:
            raise ValueError('The opening and hold durations must fit the original release stage.')
        sample = np.clip((seconds-hold)/(duration-hold),0.,1.)*(len(seconds)-1)
        actions[release_a:release_b,active] = np.column_stack([np.interp(sample,np.arange(len(seconds)),old[release_a:release_b,j]) for j in range(offset,offset+5)])+delta
        closed,opened = old[lower_b-1,offset+5],old[release_a:release_b,offset+5].max()
        actions[release_a:release_b,offset+5] = closed+(opened-closed)*smooth(seconds/p['opening_seconds'])
        retract_a,retract_b = limits['retract']
        taper = 1-smooth(np.linspace(0.,1.,retract_b-retract_a))
        actions[retract_a:retract_b,active] = old[retract_a:retract_b,active]+taper[:,None]*delta
        changed = float(np.max(abs(actions[:,active]-old[:,active])))
        if changed>p['maximum_active_joint_label_change_rad']:
            raise ValueError('Active motor-label changes exceed the declared limit.')
        outside = np.ones(len(actions),dtype=bool)
        for stage in ('lower','release','retract'):
            a,b = limits[stage];outside[a:b]=False
        if not np.array_equal(actions[outside],old[outside]) or not np.array_equal(actions[:,inactive],old[:,inactive]):
            raise ValueError('Undeclared stages or inactive-arm targets changed.')
        records.append({'episode':episode,'lower_cut_endpoint':float(cut),'target_clearance_m':p['clearance_m'],
                        'actual_fk_clearance_m':float(tool(raised)[2]-z[-1]),'maximum_active_joint_change_rad':changed,
                        'joint_offset_rad':delta.tolist(),'opening_target_rad':float(opened)})
    space(p,args.output,64*1024**2)
    with (args.output/'retrieval.npz').open('xb') as stream:
        np.savez_compressed(stream,**values)
    info['action_preprocessing'] += ' Separately declared '+p['schema']+': lowering clearance and stationary finger opening before lateral retreat, with a tapered retract offset. Requires a new full physical replay gate.'
    info['release_protocol_sha256'] = sha(args.protocol)
    info['parent_training_input_sha256'] = p['source_sha256']
    write_json(args.output/'retrieval.json',info)
    report = {'schema':p['schema'],'protocol_sha256':sha(args.protocol),'source_sha256':p['source_sha256'],
              'candidate_sha256':sha(args.output/'retrieval.npz'),'script_sha256':sha(Path(__file__)),
              'helper_sha256':sha(ROOT/'scripts/release_experiment.py'),'records':records,
              'physical_replay_complete':False,'training_started':False,'originals_modified':False}
    write_json(args.output/'transformation.json',report)
    print(json.dumps({'episodes':len(records),'candidate_sha256':report['candidate_sha256'],
                      'maximum_active_joint_change_rad':max(r['maximum_active_joint_change_rad'] for r in records),
                      'fk_clearance_range_m':[min(r['actual_fk_clearance_m'] for r in records),max(r['actual_fk_clearance_m'] for r in records)],
                      'next_gate':'Every revised exact float32 trajectory must pass physical replay before training.'}))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
