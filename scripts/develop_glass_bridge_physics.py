"""One motor-only vertical bridge placement, then actual-state horizontal final leg."""
import argparse,contextlib,json,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from simulation_lab.storage import require_space
from scripts.develop_side_plate_grasp import put
from scripts.develop_glass_reorientation import run

def main(args):
    out=args.output.resolve();log=out.with_suffix('.log')
    if out.exists() or log.exists():raise FileExistsError('Preserve output and log.')
    preflight=require_space(out,600*1024**2);out.mkdir(parents=True)
    with log.open('x') as stream,contextlib.redirect_stdout(stream):
        proof=json.loads(args.proof.read_text());put(out/'bridge-proof.json',proof);put(out/'harness.py',Path(__file__).read_bytes())
        base=dict(seed=2026114004,mode='upright',side_axis=False,bearing=0.,post_hold_sideways=False,horizontal_plane=False,roll_bias=0.,align_duration=4.,carry_torque=None,carry_clearance=0.,reorient_corrections=0,torque=.65,tool_x=None,tool_z=None,rim_angle=None,side_wrap=False,refined_family=False,pitch=None,goal_contact=False,goal_contact_height=.072,rim_radius=.022,rim_height=.067,closing_sign=1,branches=24,arm='right',require_full_preflight=False)
        first=SimpleNamespace(**base,output=out/'leg1',teacher_source=str(args.proof.parent/'source/simulation_lab/table_teacher.py'),initial_from='.run/whole-table-development-v18-seed4004/lookahead-01-034-00-glass',replay_prefix=False,grasp='padded_glass_wrap_0.045_0.78540',target=([.14425000014816797,-.020749999937376695,.76] if args.replan_after_hold else proof['vertical']['target_position']),explicit_target_quaternion=([1.,0.,0.,0.] if args.replan_after_hold else proof['vertical']['target_quaternion']),horizontal_proof=None,post_hold_bridge=str(args.proof) if args.replan_after_hold else None)
        first.bridge_park_fallback=args.continue_released is not None;first.continuation_from=args.continue_released
        run(first);first_result=json.loads((out/'leg1/result.json').read_text());results=[dict(leg=1,result=first_result)]
        if first_result['status']=='succeeded' and first_result['replay_exact']:
            base['rim_height']=proof['horizontal']['band'];base['arm']=proof['horizontal']['side'];base['require_full_preflight']=True
            second=SimpleNamespace(**base,output=out/'leg2',teacher_source=str(out/'leg1/source/simulation_lab/table_teacher.py'),initial_from=str(out/'leg1'),replay_prefix=True,grasp='',target=[.075,.10,.76],explicit_target_quaternion=[1.,0.,0.,0.],horizontal_proof='.run/glass-goal-geometry-v3/proof-001.json')
            run(second);results.append(dict(leg=2,result=json.loads((out/'leg2/result.json').read_text())))
        put(out/'result.json',dict(scope='Exposed privileged teacher dependency only, not all-seven success or learned coverage.',preflight=preflight,results=results,completed_two_leg_route=len(results)==2 and all(x['result']['status']=='succeeded' and x['result']['replay_exact'] for x in results)))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--proof',type=Path,default=Path('.run/glass-bridge-geometry-v7/proof-001.json'));p.add_argument('--replan-after-hold',action='store_true');p.add_argument('--continue-released');main(p.parse_args())
