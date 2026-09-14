"""Bounded fork support geometry gate after exact accepted-scene motor prefix."""
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
from simulation_lab.table_regrasp import copied_data
from simulation_lab.fork_supported_regrasp import head_grasp_gate,buffer_poses,side_buffer_gate,final_horizontal_gate


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--buffers',action='store_true');p.add_argument('--side-buffer',action='store_true');p.add_argument('--final-endpoints',action='store_true');args=p.parse_args()
    out=args.out.resolve();log=out.with_suffix('.log')
    if out.exists() or log.exists():raise FileExistsError('Preserve output and log.')
    preflight=require_space(out,256*1024**2);out.mkdir(parents=True)
    for path in [*sorted((ROOT/'simulation_lab').glob('*.py')),Path(__file__)]:
        dest=out/'source'/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
    xml,layout=build_scene(seed=2026114004,scenario='dinner',dinner_preset='task');(out/'scene.xml').write_text(xml)
    model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model);spec=mujoco.mjtState.mjSTATE_INTEGRATION
    prefix=[];source=ROOT/'.run/whole-table-development-v24-seed4004'
    for index in range(4):
        action=json.loads((source/f'accepted-action-{index:02d}.json').read_text());folder=source/action['path']
        with np.load(folder/'states.npz') as archive:s={k:archive[k] for k in ('initial_integration','qpos','qvel','ctrl')}
        if index==0:mujoco.mj_setState(model,data,s['initial_integration'],spec);mujoco.mj_forward(model,data)
        assert np.array_equal(data.qpos,s['qpos'][0]) and np.array_equal(data.qvel,s['qvel'][0]),'Prefix boundary mismatch.'
        for tick,ctrl in enumerate(s['ctrl']):
            data.ctrl[:]=ctrl;mujoco.mj_step(model,data)
            assert np.array_equal(data.qpos,s['qpos'][tick+1]) and np.array_equal(data.qvel,s['qvel'][tick+1]),'Prefix replay mismatch.'
        prefix.append({'action_index':index,'motor_steps':len(s['ctrl']),'replay_exact':True,'states_sha256':hashlib.sha256((folder/'states.npz').read_bytes()).hexdigest()})
    initial=np.empty(mujoco.mj_stateSize(model,spec));mujoco.mj_getState(model,data,initial,spec)
    with (out/'prefix-final-state.npz').open('xb') as f:np.savez_compressed(f,integration=initial,qpos=data.qpos,qvel=data.qvel)
    report={'scope':'Fork-specific hypothetical geometry after exact continuously evolved prefix. No new physical trial.','preflight':preflight,'prefix':prefix,'new_physical_steps':0,'source_fork_pose':data.joint('fork_free').qpos.tolist(),'checks':[]}
    found,rows=(final_horizontal_gate(model,data,layout) if args.final_endpoints else side_buffer_gate(model,data,layout) if args.side_buffer else head_grasp_gate(model,data,layout))
    report['checks'].append({'pose':'hypothetical_final_horizontal' if args.final_endpoints else 'hypothetical_side_buffers' if args.side_buffer else 'actual_current_scene','rows':rows,'planned':bool(found)})
    if not found and args.buffers:
        for pose in buffer_poses(model,data):
            scratch=copied_data(model,data);scratch.joint('fork_free').qpos[:3]=pose['center'];scratch.joint('fork_free').qpos[3:]=pose['quaternion'];mujoco.mj_forward(model,scratch)
            found,rows=head_grasp_gate(model,scratch,layout,full_routes=False)
            row={'pose':{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in pose.items()},'rows':rows,'planned':bool(found)}
            report['checks'].append(row)
            print({'pose':row['pose'],'planned':bool(found)},flush=True)
            if found:break
    report['geometry_candidate_found']=bool(found);report['physical_fork_setting_verified']=False
    payload=json.dumps(report,indent=2)+'\n';require_space(out,len(payload)+1024)
    (out/'report.json').write_text(payload);log.write_text(json.dumps({'geometry_candidate_found':bool(found),'poses':len(report['checks']),'prefix_steps':sum(x['motor_steps'] for x in prefix),'new_physical_steps':0})+'\n');print(log.read_text())


if __name__=='__main__':main()
