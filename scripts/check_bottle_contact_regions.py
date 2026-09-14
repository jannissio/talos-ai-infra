"""Reconstruct saved planned/actual contact geometry without any new trial."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,hashlib,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from simulation_lab.storage import require_space
from simulation_lab.scene import build_scene
from simulation_lab.table_teacher import TableTeacher
from simulation_lab.bottle_lift_probe import bottle_contact_region_violations


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=2026114001)
    parser.add_argument('--accepted-run',type=Path)
    args=parser.parse_args()
    out=args.out.resolve();log=out.with_suffix('.log')
    if out.exists() or log.exists():raise FileExistsError('Preserve output and log.')
    preflight=require_space(out,128*1024**2);out.mkdir(parents=True)
    for path in [Path(__file__),ROOT/'simulation_lab/bottle_lift_probe.py',ROOT/'simulation_lab/table_teacher.py',ROOT/'simulation_lab/autonomy.py',ROOT/'simulation_lab/dinner_autonomy.py']:
        p=out/'source'/path.relative_to(ROOT);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(path.read_bytes())
    xml,layout=build_scene(seed=args.seed,scenario='dinner',dinner_preset='task');(out/'scene.xml').write_text(xml)
    model=mujoco.MjModel.from_xml_string(xml)
    sources=[('original',ROOT/'.run/whole-table-development-v23-bottle4001/lookahead-00-010-00-bottle',.04,.85,'left',None),
        ('alternative_1',ROOT/'.run/bottle-com-lift-physical-v1',.05,.85,'left',None),
        ('alternative_2',ROOT/'.run/bottle-com-lift-physical-v2',.05,1.,'left',None)]
    if args.accepted_run:
        sources=[]
        for index in (0,1):
            path=args.accepted_run.resolve()/f'accepted-action-{index:02d}.json'
            action=json.loads(path.read_text())
            assert action['item']=='bottle' and action['status']=='succeeded'
            band=float(action['candidate'].split('_band_')[1].split('_')[0])
            sources.append((f'accepted_{index}',path.parent/action['path'],band,.85,action['arm'],action))
            (out/path.name).write_bytes(path.read_bytes())
    report={'scope':'Saved nominal open-approach and actual-state geometry audit only; zero new physical steps.',
        'preflight':preflight,'contract':{'distal_gripper_local_z_max_m':-.075,'axial_band_half_width_m':.025,'contact_penetration_threshold_m':.00015,
            'meaning':'Conservative grasp-family contact semantics; action rejection does not classify the scene unreachable.'},'sources':[]}
    for label,folder,band,opening,side,accepted in sources:
        require_space(out,32*1024**2)
        with np.load(folder/'states.npz') as a:s={key:a[key] for key in ('qpos','qvel','ctrl','stage')}
        data=mujoco.MjData(model);data.qpos[:]=s['qpos'][0];mujoco.mj_forward(model,data)
        task=TableTeacher(model,data,layout);task._select_item(side,next(x for x in layout['objects'] if x['id']=='bottle'))
        offset=0 if side=='left' else 6
        actual_mask=[];planned_mask=[];actual_records=[];planned_records=[]
        for i,q in enumerate(s['qpos']):
            data.qpos[:]=q;data.qvel[:]=s['qvel'][i];mujoco.mj_forward(model,data)
            violations=bottle_contact_region_violations(model,data,side,band,task.jaw_geoms)
            actual_mask.append(bool(violations))
            if violations and (not actual_records or i%10==0 or i==len(s['qpos'])-1):actual_records.append({'frame':i,'stage':str(s['stage'][i]),'contacts':violations})
            if i<len(s['ctrl']) and str(s['stage'][i+1]) in ('approach','descend'):
                # Arm control equals the recorded planned joint target. Keep the
                # scene/object at its original planning pose and jaws open.
                data.qpos[:]=s['qpos'][0];data.qpos[offset:offset+5]=s['ctrl'][i,offset:offset+5];data.qpos[offset+5]=opening
                data.qvel[:]=0.;mujoco.mj_forward(model,data)
                violations=bottle_contact_region_violations(model,data,side,band,task.jaw_geoms)
                planned_mask.append(bool(violations))
                if violations and (not planned_records or i%10==0):planned_records.append({'control_frame':i,'stage':str(s['stage'][i+1]),'contacts':violations})
        final_actual=actual_records[-1] if actual_mask[-1] else None
        row={'label':label,'source':str(folder),'side':side,'band_m':band,'prior_physical_result':accepted,
            'source_states_sha256':hashlib.sha256((folder/'states.npz').read_bytes()).hexdigest(),
            'nominal_open_route_rejected':any(planned_mask),'nominal_violating_frames':sum(planned_mask),'actual_violating_frames':sum(actual_mask),
            'final_actual_pose_rejected':bool(actual_mask[-1]),'final_actual_violations':final_actual,
            'first_nominal_violation':planned_records[0] if planned_records else None,
            'first_actual_violation':actual_records[0] if actual_records else None}
        report['sources'].append(row)
        with (out/f'{label}-masks.npz').open('xb') as stream:np.savez_compressed(stream,actual=np.array(actual_mask),nominal_open_route=np.array(planned_mask))
        (out/f'{label}-contact-maps.json').write_text(json.dumps({'actual':actual_records,'nominal':planned_records},indent=2)+'\n')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    compact=[{k:r[k] for k in ('label','nominal_open_route_rejected','nominal_violating_frames','actual_violating_frames','final_actual_pose_rejected')} for r in report['sources']]
    log.write_text(json.dumps(compact)+'\n');print(log.read_text())


if __name__=='__main__':main()
