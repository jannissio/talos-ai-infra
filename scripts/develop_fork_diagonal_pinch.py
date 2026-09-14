"""New finite diagonal endpoint protocol. All poses are hypothetical geometry."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,hashlib,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from simulation_lab.scene import build_scene
from simulation_lab.storage import require_space
from simulation_lab.fork_diagonal_pinch import diagonal_endpoint_gate


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True,type=Path);args=parser.parse_args()
    out=args.out.resolve();log=out.with_suffix('.log')
    if out.exists() or log.exists():raise FileExistsError('Preserve output and log.')
    pf=require_space(out,128*1024**2);out.mkdir(parents=True)
    with log.open('x',encoding='utf8') as stream:
        protocol=ROOT/'docs/robotics/experiments/fork-diagonal-pinch-development-v1.json'
        (out/'protocol.json').write_bytes(protocol.read_bytes())
        manifest={}
        for p in [*sorted((ROOT/'simulation_lab').glob('*.py')),Path(__file__)]:
            dest=out/'source'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
            manifest[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
        (out/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        source=ROOT/'.run/whole-table-development-v26-direct-contact'
        action=json.loads((source/'accepted-action-00.json').read_text());states=source/action['path']/'states.npz'
        xml,layout=build_scene(seed=2026122001,scenario='dinner',dinner_preset='task');(out/'scene.xml').write_text(xml)
        model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model)
        with np.load(states) as a:
            data.qpos[:]=a['qpos'][-1];data.qvel[:]=a['qvel'][-1];data.ctrl[:]=a['ctrl'][-1]
        # This new MjData exists solely for static geometry reconstruction.
        # The fork at its requested final pose has NOT been physically placed.
        data.joint('fork_free').qpos[:]=[-.16,-.045,.76,1.,0.,0.,0.]
        mujoco.mj_forward(model,data)
        before=data.qpos.copy()
        def record(index,row):
            payload=json.dumps(row,indent=2)+'\n';require_space(out,len(payload)+1024)
            with (out/f'endpoint-{index:03d}.json').open('x',encoding='utf8') as f:f.write(payload)
            compact={'index':index,'candidate':row['candidate'],'arm':row['arm'],'ik_solutions':len(row['unique_solutions'])}
            stream.write(json.dumps(compact)+'\n');stream.flush();print(compact,flush=True)
        rows,clear=diagonal_endpoint_gate(model,data,layout,record)
        assert np.array_equal(before,data.qpos)
        report={'scope':'Hypothetical fork final-pose diagonal-pinch endpoint geometry only. No reached pose, force-support, physical settling or transfer claim.',
            'preflight':pf,'new_physical_steps':0,'prefix_replay_steps':0,'source_state_geometry_only':True,
            'source_action_sha256':hashlib.sha256((source/'accepted-action-00.json').read_bytes()).hexdigest(),
            'source_states_sha256':hashlib.sha256(states.read_bytes()).hexdigest(),
            'endpoint_candidate_count':len(rows),'ik_seed_attempts':sum(len(r['seed_attempts']) for r in rows),
            'ik_solved_seed_attempts':sum(s['solved'] for r in rows for s in r['seed_attempts']),
            'unique_ik_solutions':sum(len(r['unique_solutions']) for r in rows),'clear_endpoints':clear,
            'physical_success':False,'whole_table_complete':False,'demonstration_eligible':False}
        with (out/'report.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2);f.write('\n')
        stream.write(json.dumps(report)+'\n');print(report,flush=True)


if __name__=='__main__':main()
