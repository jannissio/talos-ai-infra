"""Separate neural motor refusal from offline kinematic feasibility on a fixed grid."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
import torch
from simulation_lab.autonomy import ArmIK, PlanningError
from simulation_lab.rgb_servo_motor import CartesianMotorPolicy
from simulation_lab.scene import build_scene, HOME
from simulation_lab.storage import require_space


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads(args.protocol.read_text()); probe = protocol['probe']
    require_space(args.output, 16*1024**2); args.output.mkdir(parents=True)
    xml, layout = build_scene(seed=probe['scene_seed'],scenario='dinner',dinner_preset='task')
    model = mujoco.MjModel.from_xml_string(xml); data = mujoco.MjData(model)
    data.qpos[:12] = HOME*2; mujoco.mj_forward(model,data)
    torch.set_num_threads(2); motor = CartesianMotorPolicy(args.motor,model)
    rows = []; calls = 0; began = time.perf_counter()
    points = [('grid',[x,y,layout['table_z']+height]) for x in probe['x_samples_m']
              for y in probe['y_samples_m'] for height in probe['height_above_table_m']]
    points += [('destination',[*xy,layout['table_z']+height]) for xy in protocol['destinations_m']
               for height in probe['destination_height_above_table_m']]
    for kind, point in points:
        for side in ('left','right'):
            require_space(args.output,1024**2)
            if time.perf_counter()-began > probe['max_wall_minutes']*60:
                raise RuntimeError('Declared probe wall budget reached; retain the partial report.')
            row = {'kind':kind,'point_m':point,'side':side,'neural_accepted':False,'solver_accepted':False,'solver_attempts':[]}
            try:
                prediction = motor.predict(side,point); row['neural_accepted'] = True
                row['neural_joints'] = prediction.tolist()
            except ValueError as exc:
                row['neural_refusal'] = str(exc)
            solver = ArmIK(model,data,side)
            starts = [np.asarray(HOME[:5]),(solver.lo+solver.hi)/2]
            for index, start in enumerate(starts):
                if calls >= probe['max_solver_calls']:
                    raise RuntimeError('Declared solver-call budget reached.')
                calls += 1
                try:
                    solution = solver.solve(np.asarray(point),start)
                    actual,rotation = motor.geometry.pose(side,solution)
                    row['solver_attempts'].append({'start':index,'succeeded':True,'joints':solution.tolist(),
                         'position_error_mm':float(np.linalg.norm(actual-point)*1000),
                         'axis_error':float(np.linalg.norm(rotation[:,1]-[0,0,1]))})
                    row['solver_accepted'] = True
                    break
                except PlanningError as exc:
                    row['solver_attempts'].append({'start':index,'succeeded':False,'message':str(exc)})
            rows.append(row)
            result = {'schema':'talos.rgb-servo-motor-probe.v2','protocol_sha256':hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
                      'motor_sha256':hashlib.sha256(args.motor.read_bytes()).hexdigest(),'attempted_arm_points':len(rows),
                      'planned_arm_points':len(points)*2,'solver_calls':calls,'wall_seconds':time.perf_counter()-began,
                      'counts':{'neural_accepted':sum(r['neural_accepted'] for r in rows),
                                'solver_accepted':sum(r['solver_accepted'] for r in rows),
                                'solver_pass_neural_refused':sum(r['solver_accepted'] and not r['neural_accepted'] for r in rows)},
                      'scope':'Kinematic probe only; no collision-free physical manipulation claim.', 'rows':rows}
            (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=Path('docs/robotics/experiments/rgb-servo-bottle-v2.json'))
    parser.add_argument('--motor',type=Path,default=Path('models/bottle_servo_v1/motor.safetensors'))
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
