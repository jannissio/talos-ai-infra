"""Coupled azimuth solve for the unchanged legacy mug wall geometry; no physics.

For radial closing direction equal to tool X, the wall contact equation is
tool_origin + R*local_tool = mug_origin + .041*bodyZ + .023*toolX.
Thus solve tool_origin + R*(local_tool-[.023,0,0]) at the centerline band while
aligning tool Z with body Z. This frees the contact azimuth analytically without
changing the physical contact offset, depth, torque, geometry or IK tolerances.
"""
import contextlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from scripts.develop_mug_wall_grasp import setup
from scripts.develop_glass_depth_two import prepare, put, arrays


def main():
    out = ROOT / '.run/mug-wall-coupled-geometry-v1'
    previous = ROOT / '.run/mug-wall-geometry-v2'
    log = prepare(out)
    with log.open('x') as stream, contextlib.redirect_stdout(stream):
        teacher, whole, module = setup(out, previous)
        put(out / 'coupled-source.py', Path(__file__).read_bytes())
        from simulation_lab.scene import HOME
        report = json.loads((previous / 'report.json').read_text())
        rows, proofs = [], []
        for context in report['rows']:
            folder = previous / context['context']
            xml = (folder / 'scene.xml').read_bytes()
            layout = json.loads((folder / 'layout.json').read_text())
            model = mujoco.MjModel.from_xml_string(xml.decode())
            data = mujoco.MjData(model)
            with np.load(folder / 'actual-source-state.npz') as archive:
                data.qpos[:] = archive['qpos']; data.qvel[:] = archive['qvel']; data.ctrl[:] = archive['ctrl']
            mujoco.mj_forward(model, data)
            actual_pose = data.joint('mug_free').qpos.copy()
            variants = [('privileged_teacher_pose', actual_pose)]
            if context['context'].startswith('V24'):
                estimate = json.loads((previous / 'camera-estimate-before-truth.json').read_text())['estimate']
                variants.append(('camera_derived_grasp_seed_actual_geometry_checks', np.r_[estimate['body_origin_m'], estimate['body_quaternion_wxyz']]))
            for mode, pose in variants:
                rotation = np.empty((3, 3))
                mujoco.mju_quat2Mat(rotation.ravel(), pose[3:] / np.linalg.norm(pose[3:]))
                up = rotation[:, 2]
                target = pose[:3] + .041 * up
                for side in ('left', 'right'):
                    task = teacher.TableTeacher(model, data, layout)
                    task.start(side=side, object_id='mug')
                    task._select_item(side, next(o for o in layout['objects'] if o['id'] == 'mug'))
                    for family in ('home', 'generic_elbow'):
                        for roll in (-2.4, 0., 2.4):
                            initial = np.array(HOME[:5] if family == 'home' else [0., .8, .4, -1.2, 0.])
                            initial[4] = roll
                            row = dict(context=context['context'], mode=mode, side=side, family=family, roll=roll,
                                       initial=initial.tolist(), stage='coupled_source_ik')
                            try:
                                task.ik.grasp_point = np.array([-.029, 0., -.092])
                                task.ik.axis_index = 2; task.ik.axis_target = up.copy(); task.ik.x_target = None
                                task.ik.axis_local = None; task.ik.soft_placement = False; task.ik.placement_solve = False
                                q = task.ik.solve(target, initial)
                                task.ik.data.qpos[task.offset:task.offset + 5] = q
                                mujoco.mj_kinematics(model, task.ik.data)
                                tool = task.ik.data.body(side + '_gripper').xmat.reshape(3, 3)
                                radial = tool[:, 0] - up * float(tool[:, 0] @ up)
                                radial /= np.linalg.norm(radial)
                                candidate = teacher.GraspCandidate('mug_coupled_deep_wall', target + .023 * radial,
                                    np.array([-.006, 0., -.092]), 2, up.copy(), radial, .4, up.copy(), .35)
                                task._configure(candidate)
                                q = task.ik.solve(candidate.point, q)
                                row.update(source_q=q.tolist(), point=candidate.point.tolist(), radial=radial.tolist(),
                                           source_azimuth_rad=float(np.arctan2((rotation.T @ radial)[1], (rotation.T @ radial)[0])))
                                row['stage'] = 'source_collision'
                                task._check_path([q], .4, True)
                                row['stage'] = 'destination_geometry'
                                task._precheck_transfer(q)
                                row['stage'] = 'source_route'
                                task._configure(candidate)
                                hover = candidate.point + .04 * up
                                above = task.ik.solve(hover, q)
                                descent = task._cartesian(hover, candidate.point, above, .4)
                                approach = task._route(data.qpos[task.offset:task.offset + 5], above, .4)
                                row.update(passed=True, approach_samples=len(approach), descent_samples=len(descent))
                                proofs.append(row)
                                put(out / f'proof-{len(proofs):03d}.json', row)
                            except teacher.PlanningError as exc:
                                row.update(passed=False, error=str(exc))
                            rows.append(row)
                            put(out / f'check-{len(rows):03d}.json', row)
                            print({k: v for k, v in row.items() if k not in ('source_q', 'point', 'radial')}, flush=True)
        put(out / 'report.json', dict(scope=__doc__, rows=rows, proofs=proofs, physics_steps=0,
                                      radius_m=.023, height_m=.041, physical_local_tool_point=[-.006, 0., -.092], torque_nm=.35,
                                      result='No checked route' if not proofs else 'Checked geometry only; physical verification required'))


if __name__ == '__main__':
    main()
