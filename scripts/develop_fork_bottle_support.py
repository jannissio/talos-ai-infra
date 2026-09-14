"""Two reset-only bottle-rim support tests after exact V26 bottle prefix."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,hashlib,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from simulation_lab.storage import require_space
from simulation_lab.scene import build_scene
from simulation_lab.table_regrasp import copied_data
from simulation_lab.fork_bottle_support import support_poses,support_metrics,receiver_gate


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);args=p.parse_args();out=args.out.resolve();log=out.with_suffix('.log')
    if out.exists() or log.exists():raise FileExistsError('Preserve output and log.')
    pf=require_space(out,256*1024**2);out.mkdir(parents=True)
    with log.open('x',encoding='utf8') as stream:
        for path in [*sorted((ROOT/'simulation_lab').glob('*.py')),Path(__file__)]:
            dest=out/'source'/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
        protocol=ROOT/'docs/robotics/experiments/fork-bottle-support-development-v1.json';(out/'protocol.json').write_bytes(protocol.read_bytes())
        xml,layout=build_scene(seed=2026122001,scenario='dinner',dinner_preset='task');(out/'scene.xml').write_text(xml)
        model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model);spec=mujoco.mjtState.mjSTATE_INTEGRATION
        source=ROOT/'.run/whole-table-development-v26-direct-contact';action=json.loads((source/'accepted-action-00.json').read_text())
        recipe=json.loads((source/'reset-recipe.json').read_text())
        prefixes=[source/'reset-candidates'/f"{recipe['selected_candidate_index']:03d}"/'states.npz',source/action['path']/'states.npz'];proof=[]
        for part,path in enumerate(prefixes):
            with np.load(path) as a:s={k:a[k] for k in ('initial_integration','qpos','qvel','ctrl')}
            if part==0:mujoco.mj_setState(model,data,s['initial_integration'],spec);mujoco.mj_forward(model,data)
            assert np.array_equal(data.qpos,s['qpos'][0]) and np.array_equal(data.qvel,s['qvel'][0]),'Prefix boundary mismatch.'
            for tick,ctrl in enumerate(s['ctrl']):
                data.ctrl[:]=ctrl;mujoco.mj_step(model,data)
                assert np.array_equal(data.qpos,s['qpos'][tick+1]) and np.array_equal(data.qvel,s['qvel'][tick+1]),'Prefix replay mismatch.'
            proof.append({'segment':part,'steps':len(s['ctrl']),'states_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'replay_exact':True})
        trials=[]
        for index,pose in enumerate(support_poses(model,data)):
            require_space(out,128*1024**2);trial=copied_data(model,data)
            # Explicit reset-only intervention BEFORE this support trial starts.
            # This is not physical delivery and never changes the evolved data.
            trial.joint('fork_free').qpos[:3]=pose['center'];trial.joint('fork_free').qpos[3:]=pose['quaternion']
            dof=int(model.joint('fork_free').dofadr[0]);trial.qvel[dof:dof+6]=0.;mujoco.mj_forward(model,trial)
            initial=np.empty(mujoco.mj_stateSize(model,spec));mujoco.mj_getState(model,trial,initial,spec)
            others={o['id']:trial.body(o['id']).xpos.copy() for o in layout['objects'] if o['id']!='fork'}
            qs=[trial.qpos.copy()];vs=[trial.qvel.copy()];controls=[];checks=[];first=support_metrics(model,trial,others);failure=None
            if max(first['fork_external_penetration_mm'],first['bottle_external_penetration_mm'])>1. or first['robot_penetration_mm']>.8:failure='Invalid support reset collision.'
            for tick in range(600 if failure is None else 0):
                assert model.neq==0 and not np.any(trial.xfrc_applied) and not np.any(trial.qfrc_applied)
                controls.append(trial.ctrl.copy());mujoco.mj_step(model,trial);qs.append(trial.qpos.copy());vs.append(trial.qvel.copy())
                check=support_metrics(model,trial,others);checks.append(check)
                if check['fork_external_penetration_mm']>1.:failure='Fork external penetration exceeded1mm.'
                if check['bottle_external_penetration_mm']>1.:failure='Bottle external penetration exceeded1mm.'
                if check['robot_penetration_mm']>.8:failure='Unexpected robot contact exceeded0.8mm.'
                if max(check['other_displacement_mm'].values())>4.:failure='Bottle/unrelated item moved over4mm.'
                if failure:break
            with (out/f'support-{index:02d}-states.npz').open('xb') as f:np.savez_compressed(f,initial_integration=initial,ctrl=np.asarray(controls).reshape(-1,model.nu),qpos=np.asarray(qs),qvel=np.asarray(vs))
            replay=mujoco.MjData(model);mujoco.mj_setState(model,replay,initial,spec);mujoco.mj_forward(model,replay);exact=True
            for tick,ctrl in enumerate(controls):
                replay.ctrl[:]=ctrl;mujoco.mj_step(model,replay)
                exact &= np.array_equal(replay.qpos,qs[tick+1]) and np.array_equal(replay.qvel,vs[tick+1])
            stable=not failure and len(checks)>=100 and all(c['stable'] for c in checks[-100:])
            row={'pose':{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in pose.items()},'reset_only':True,'steps':len(controls),'saved_frames':len(qs),
                'replay_exact':bool(exact),'failure':failure,'stable_final_half_second':bool(stable),'final':checks[-1] if checks else first,
                'maximum_fork_external_penetration_mm':max(c['fork_external_penetration_mm'] for c in [first,*checks]),
                'maximum_bottle_displacement_mm':max(c['other_displacement_mm']['bottle'] for c in [first,*checks]),'geometry_found':False}
            with (out/f'support-{index:02d}-contacts.json').open('x',encoding='utf8') as f:json.dump({'initial':first,'frames':checks},f)
            if stable:
                found,search=receiver_gate(model,trial,layout,pose['receiver']);row['geometry_found']=bool(found);row['receiver_search']=search
            with (out/f'support-{index:02d}-result.json').open('x',encoding='utf8') as f:json.dump(row,f,indent=2);f.write('\n')
            trials.append(row);brief={k:row[k] for k in ('steps','replay_exact','failure','stable_final_half_second','geometry_found')};stream.write(json.dumps(brief)+'\n');stream.flush();print(brief,flush=True)
        report={'scope':'Reset-only fork support on actual bottle rim, followed by measured-pose geometry checks. No physical delivery/regrasp/placement.',
            'preflight':pf,'prefix':proof,'support_reset_trials':len(trials),'support_physics_steps':sum(r['steps'] for r in trials),'new_manipulation_steps':0,
            'stable_support_resets':sum(r['stable_final_half_second'] for r in trials),'checked_receiver_routes':sum(r['geometry_found'] for r in trials),
            'physical_fork_setting':False,'whole_table_complete':False,'trials':trials}
        with (out/'report.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2);f.write('\n')
        stream.write(json.dumps({k:v for k,v in report.items() if k!='trials'})+'\n');print({k:v for k,v in report.items() if k!='trials'},flush=True)


if __name__=='__main__':main()
