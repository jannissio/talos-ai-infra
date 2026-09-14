"""Retain a bounded geometry gate for physically supported edge regrasp."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,hashlib,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from simulation_lab.scene import build_scene
from simulation_lab.storage import require_space
from simulation_lab.table_edge_regrasp import assess_edge_receiver,edge_candidates,supported_overhang
from simulation_lab.table_regrasp import copied_data


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--support-diagnostic',action='store_true')
    parser.add_argument('--tilted-approaches',action='store_true')
    args=parser.parse_args();out=args.out.resolve();log=out.with_suffix('.log')
    if out.exists() or log.exists():raise FileExistsError('Preserve output and log.')
    preflight=require_space(out,128*1024**2);out.mkdir(parents=True)
    for path in (ROOT/'simulation_lab').glob('*.py'):
        dst=out/'source/simulation_lab'/path.name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(path.read_bytes())
    (out/'source/scripts').mkdir();(out/'source/scripts/develop_table_edge_regrasp.py').write_bytes(Path(__file__).read_bytes())
    xml,layout=build_scene(seed=2026114002,scenario='dinner',dinner_preset='task')
    (out/'scene.xml').write_text(xml);(out/'layout.json').write_text(json.dumps(layout,indent=2))
    model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model)
    source=ROOT/'.run/whole-table-development-v15-seed4002/lookahead-00-015-02-spoon/states.npz'
    with np.load(source) as archive:values={k:archive[k] for k in ('initial_integration','ctrl','qpos','qvel')}
    spec=mujoco.mjtState.mjSTATE_INTEGRATION
    mujoco.mj_setState(model,data,values['initial_integration'],spec);mujoco.mj_forward(model,data)
    for i,ctrl in enumerate(values['ctrl']):
        data.ctrl[:]=ctrl;mujoco.mj_step(model,data)
        assert np.array_equal(data.qpos,values['qpos'][i+1])
        assert np.array_equal(data.qvel,values['qvel'][i+1])
    support=[];rows=[];found=[]
    if args.support_diagnostic:
        poses=[x for x in edge_candidates(model,data) if x['inward_m'] in (.020,.030) and
               min(abs(x['center'][0]-z) for z in (-.2,-.133333333333,0.,.133333333333,.2))<1e-8]
        for index,candidate in enumerate(poses):
            require_space(out,64*1024**2)
            trial=copied_data(model,data)
            # Explicit reset-only stability diagnostic; not a reached workflow state.
            trial.joint('spoon_free').qpos[:3]=candidate['center'];trial.joint('spoon_free').qpos[3:]=candidate['quaternion']
            dof=int(model.joint('spoon_free').dofadr[0]);trial.qvel[dof:dof+6]=0.
            mujoco.mj_forward(model,trial)
            initial=np.empty(mujoco.mj_stateSize(model,spec));mujoco.mj_getState(model,trial,initial,spec)
            qs=[trial.qpos.copy()];vs=[trial.qvel.copy()];controls=[];checks=[]
            unrelated={o['id']:trial.body(o['id']).xpos.copy() for o in layout['objects'] if o['id']!='spoon'}
            initial_check=supported_overhang(model,trial)
            guard_reason='Reset target intersects external geometry beyond 1 mm.' if initial_check['external_penetration_mm']>1. else None
            for tick in range(600 if guard_reason is None else 0):
                assert not np.any(trial.xfrc_applied) and not np.any(trial.qfrc_applied) and model.neq==0
                controls.append(trial.ctrl.copy());mujoco.mj_step(model,trial)
                qs.append(trial.qpos.copy());vs.append(trial.qvel.copy());checks.append(supported_overhang(model,trial))
                disturbance=max(float(np.linalg.norm(trial.body(n).xpos-p)*1000) for n,p in unrelated.items())
                if checks[-1]['external_penetration_mm']>1.:guard_reason='External penetration exceeded 1 mm.'
                if disturbance>4.:guard_reason='Unrelated item moved more than 4 mm.'
                for c in trial.contact:
                    if c.dist<-.0008 and any((model.body(int(model.geom_bodyid[g])).name or '').startswith(('left_','right_')) for g in (c.geom1,c.geom2)):
                        guard_reason='Unexpected robot contact exceeded 0.8 mm.'
                if guard_reason:break
            row={'candidate':{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in candidate.items()},
                'reset_only_not_physical_delivery':True,'final':checks[-1] if checks else initial_check,
                'frames':len(controls),'guard_stop':guard_reason,
                'stable_for_final_half_second':len(checks)>=100 and all(c['supported_stable'] for c in checks[-100:]),
                'maximum_external_penetration_mm':max(c['external_penetration_mm'] for c in [initial_check,*checks]),
                'unrelated_item_max_displacement_mm':max(float(np.linalg.norm(trial.body(n).xpos-p)*1000) for n,p in unrelated.items())}
            with (out/f'support-{index:02d}-states.npz').open('xb') as stream:
                np.savez_compressed(stream,initial_integration=initial,ctrl=np.array(controls),qpos=np.array(qs),qvel=np.array(vs))
            replay=mujoco.MjData(model);mujoco.mj_setState(model,replay,initial,spec);mujoco.mj_forward(model,replay)
            exact=True
            for i,ctrl in enumerate(controls):
                replay.ctrl[:]=ctrl;mujoco.mj_step(model,replay)
                exact &= np.array_equal(replay.qpos,qs[i+1]) and np.array_equal(replay.qvel,vs[i+1])
            row['motor_replay_exact']=bool(exact);support.append(row)
            (out/f'support-{index:02d}.json').write_text(json.dumps({'result':row,'contact_checks':checks},indent=2)+'\n')
            if row['stable_for_final_half_second'] and row['maximum_external_penetration_mm']<=1. and row['unrelated_item_max_displacement_mm']<=4.:
                measured=dict(candidate,center=trial.body('spoon').xpos.copy(),quaternion=trial.joint('spoon_free').qpos[3:].copy(),rotation=trial.body('spoon').xmat.reshape(3,3).copy())
                candidates,attempts=assess_edge_receiver(model,trial,layout,poses=[measured],
                    pitches=(0.,-.35,.35,-.7,.7) if args.tilted_approaches else (0.,));found.extend(candidates);rows.extend(attempts)
    else:found,rows=assess_edge_receiver(model,data,layout)
    report={'scope':'Explicit reset-only overhang settling diagnostic and measured-pose geometry gate; no physical donor placement or release demonstrated.' if args.support_diagnostic else 'Hypothetical supported-overhang planning gate only; zero new physical trial frames.',
        'support_physics_frames':sum(r['frames'] for r in support),'manipulation_trial_frames':0,
        'preflight':preflight,'prefix_replay_exact':True,'prefix_frames':len(values['ctrl']),
        'prefix_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'geometry_candidates':len(found),
        'physical_success':False,'whole_table_complete':False,'checks':rows,'support_diagnostics':support,
        'support_geometry_note':'Inverted spoon handle is 16 mm thick; bowl top is only 7 mm above body origin. Flat inverted overhang rests on the inside portion of its handle, not its bowl. Actual release/stability still requires physical verification.'}
    payload=json.dumps(report,indent=2)+'\n';require_space(out,len(payload)+1024)
    (out/'geometry.json').write_text(payload);log.write_text(json.dumps({'geometry_candidates':len(found),'checks':len(rows),'physical_success':False})+'\n')
    print(log.read_text())


if __name__=='__main__':main()
