import json
from pathlib import Path
import mujoco,numpy as np,torch
from simulation_lab.autonomy import ArmIK,PlanningError
from simulation_lab.rgb_servo_motor import CartesianMotorPolicy
from simulation_lab.storage import require_space

root=Path.cwd();folder=root/'.run/rgb-servo-v2-routing-5604-r2';output=root/'.run/rgb-servo-v2-offset-diagnostic'
if output.exists():raise FileExistsError(output)
require_space(output,8*1024**2);output.mkdir()
report=json.loads((folder/'report.json').read_text());routing=json.loads((root/'docs/robotics/experiments/rgb-servo-routing-v2-r2.json').read_text())
offset=np.asarray(report['current_route_details']['visual_carry_updates'][-1]['observed_offset_m'])
model=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'));data=mujoco.MjData(model)
with np.load(folder/'states.npz',allow_pickle=False) as states:data.qpos[:]=states['qpos'][-1]
torch.set_num_threads(2);motor=CartesianMotorPolicy(root/'models/bottle_servo_v1/motor.safetensors',model)
start,_=motor.geometry.pose('left',data.qpos[:5]);solver=ArmIK(model,data,'left');rows=[]
for height in routing['transport_height_above_table_m']:
    end=np.r_[report['destination_m'],report['layout']['table_z']+height]-offset
    for fraction in np.linspace(0,1,13):
        point=start+(end-start)*fraction
        try:motor.predict('left',point)
        except ValueError as exc:
            row={'transport_height':height,'fraction':float(fraction),'point_m':point.tolist(),'neural_refusal':str(exc),'solver_accepted':False}
            try:
                q=solver.solve(point,data.qpos[:5]);actual,_=motor.geometry.pose('left',q)
                row.update(solver_accepted=True,solver_error_mm=float(np.linalg.norm(actual-point)*1000))
            except PlanningError as exc:row['solver_refusal']=str(exc)
            rows.append(row)
result={'source':'Exposed training seed 2026095604, revision 2; last 20 Hz state approximates the align transition.',
        'neural_refused_probes':len(rows),'solver_accepted_of_neural_refusals':sum(r['solver_accepted'] for r in rows),'rows':rows}
(output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
