import unittest

import mujoco
import numpy as np

from scripts.evaluate_dinner_scene import load
from scripts.develop_whole_table import orientation_progress
from simulation_lab.autonomy import PlanningError
from simulation_lab.scene import TABLE_Z
from simulation_lab.table_teacher import TableIK, TableTeacher


class SharedTablePlanningTests(unittest.TestCase):
    def test_offset_point_and_object_normal_without_live_state_writes(self):
        model, live, _ = load()
        original = live.qpos.copy()
        expected = mujoco.MjData(model)
        expected.qpos[:] = live.qpos
        expected.qpos[:5] = [.7, -.1, .2, 1.35, 2.5]
        mujoco.mj_forward(model, expected)
        ik = TableIK(model, live, 'left')
        ik.soft_placement = True
        ik.grasp_point = np.array([.035, -.012, -.095])
        ik.axis_local = np.array([-.1036, -.0142, .9945])
        ik.axis_local /= np.linalg.norm(ik.axis_local)
        rotation = expected.body('left_gripper').xmat.reshape(3, 3)
        target = expected.body('left_gripper').xpos+rotation@ik.grasp_point
        ik.axis_target = rotation@ik.axis_local
        q = ik.solve(target, expected.qpos[:5]+[.05, -.08, .06, .03, -.2])
        expected.qpos[:5] = q
        mujoco.mj_forward(model, expected)
        actual = expected.body('left_gripper').xpos+expected.body('left_gripper').xmat.reshape(3, 3)@ik.grasp_point
        self.assertLess(np.linalg.norm(actual-target), .00025)
        self.assertGreater(np.dot(expected.body('left_gripper').xmat.reshape(3, 3)@ik.axis_local, ik.axis_target), np.cos(np.deg2rad(3)))
        np.testing.assert_array_equal(live.qpos, original)
        self.assertTrue(np.all(q >= ik.lo) and np.all(q <= ik.hi))

    def test_unsolved_bounds_are_reported_as_planning_failure(self):
        model, live, _ = load()
        ik = TableIK(model, live, 'left')
        ik.soft_placement = True
        with self.assertRaises(PlanningError):
            ik.solve(np.array([10., 0., 1.]), live.qpos[:5])

    def test_plane_constraint_keeps_free_bearing_and_live_state_unchanged(self):
        model, live, _ = load()
        original = live.qpos.copy()
        expected = mujoco.MjData(model); expected.qpos[:] = live.qpos
        expected.qpos[:5] = [.5, -.4, .7, .8, -1.4]
        mujoco.mj_forward(model, expected)
        ik = TableIK(model, live, 'left')
        ik.soft_placement = ik.placement_solve = ik.axis_in_plane = True
        ik.grasp_point = np.array([.030, -.008, -.092])
        rotation = expected.body('left_gripper').xmat.reshape(3, 3)
        ik.axis_local = rotation.T@np.array([.6, .8, 0.])
        ik.axis_target = np.array([0., 0., 1.])
        target = expected.body('left_gripper').xpos+rotation@ik.grasp_point
        q = ik.solve(target, expected.qpos[:5]+[.04, -.05, .06, .02, -.1])
        expected.qpos[:5] = q; mujoco.mj_forward(model, expected)
        actual_rotation = expected.body('left_gripper').xmat.reshape(3, 3)
        actual = expected.body('left_gripper').xpos+actual_rotation@ik.grasp_point
        self.assertLess(np.linalg.norm(actual-target), .00025)
        self.assertLess(abs((actual_rotation@ik.axis_local)[2]), np.sin(np.deg2rad(3)))
        self.assertTrue(np.all(q >= ik.lo) and np.all(q <= ik.hi))
        np.testing.assert_array_equal(live.qpos, original)

    def test_inverted_temporary_target_uses_its_actual_support_height(self):
        model, live, layout = load()
        original = live.qpos.copy()
        task = TableTeacher(model, live, layout)
        task.start(object_id='side_plate', target=[0., 0., TABLE_Z], target_quaternion=[0., 1., 0., 0.])
        task._select_item('left', next(o for o in layout['objects'] if o['id'] == 'side_plate'))
        self.assertAlmostEqual(task.destination_position[2], TABLE_Z+.012)
        np.testing.assert_allclose(task.destination_rotation[:, 2], [0., 0., -1.])
        np.testing.assert_array_equal(live.qpos, original)

    def test_buffer_orientation_progress_requires_improvement_and_remaining_budget(self):
        model, before, _ = load()
        after = mujoco.MjData(model); after.qpos[:] = before.qpos
        before.joint('mug_free').qpos[3:] = [np.sqrt(.5), np.sqrt(.5), 0., 0.]
        after.joint('mug_free').qpos[3:] = [1., 0., 0., 0.]
        mujoco.mj_forward(model, before); mujoco.mj_forward(model, after)
        row = orientation_progress(before, after, 'mug', 0, [('plate', 'mug')])
        self.assertTrue(row['eligible'])
        self.assertAlmostEqual(row['improvement_deg'], 90.)
        self.assertFalse(orientation_progress(after, before, 'mug', 0, [])['eligible'])
        self.assertFalse(orientation_progress(before, after, 'mug', 2, [])['eligible'])
        self.assertFalse(orientation_progress(before, after, 'mug', 0, [('plate', 'fork')])['eligible'])


if __name__ == '__main__':
    unittest.main()
