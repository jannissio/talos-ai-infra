"""A failed speculative final leg must never alter the accepted buffer state."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from scripts.develop_whole_table import probe_final_transfer


class RelayContinuationTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body pos="0 0 1"><freejoint/>'
            '<geom type="sphere" size=".01"/></body></worldbody></mujoco>')
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.targets = np.zeros(0)
        self.before = (self.data.qpos.copy(), self.data.qvel.copy(), self.data.time)
        self.folder = Path('unused-test-output')
        self.buffer = self.folder/'lookahead-buffer'
        self.report = {'physical_lookahead': [], 'planning_failures': []}
        self.layout = {'targets': [{'object_id': 'bottle', 'position_m': [0., 0., 1.]}]}

    def teacher(self, model, data, layout):
        return SimpleNamespace(active=True, side='right', chosen=SimpleNamespace(name='grasp'),
                               chosen_roll=0., start=lambda **kw: None, update=lambda targets: None)

    def invoke(self, execute):
        with patch('scripts.develop_whole_table.TableTeacher', side_effect=self.teacher), \
             patch('scripts.develop_whole_table.execute', side_effect=execute):
            return probe_final_transfer(self.model, self.data, self.layout, self.targets,
                                        'bottle', self.buffer, self.folder, None,
                                        self.report, 2, 2)

    def assert_source_unchanged(self):
        np.testing.assert_array_equal(self.data.qpos, self.before[0])
        np.testing.assert_array_equal(self.data.qvel, self.before[1])
        self.assertEqual(self.data.time, self.before[2])

    def test_each_failed_leg_restarts_from_same_buffer_and_leaves_it_unchanged(self):
        starts = []
        def execute(model, trial, layout, task, targets, path, observer):
            starts.append((trial.qpos.copy(), trial.qvel.copy(), trial.time))
            for _ in range(10):mujoco.mj_step(model, trial)
            return {'demonstration_eligible': False, 'status': 'failed'}
        self.assertIsNone(self.invoke(execute))
        self.assertEqual(len(starts), 2)
        for qpos, qvel, time in starts:
            np.testing.assert_array_equal(qpos, self.before[0])
            np.testing.assert_array_equal(qvel, self.before[1])
            self.assertEqual(time, self.before[2])
        self.assert_source_unchanged()
        self.assertEqual(len(self.report['physical_lookahead']), 2)

    def test_passing_leg_is_returned_with_buffer_dependency_without_committing(self):
        def execute(model, trial, layout, task, targets, path, observer):
            mujoco.mj_step(model, trial)
            return {'demonstration_eligible': True, 'status': 'succeeded'}
        result, path, targets = self.invoke(execute)
        self.assertEqual(result['role'], 'final_setting')
        self.assertEqual(result['conditional_on_buffer_path'], 'lookahead-buffer')
        self.assertEqual(len(self.report['physical_lookahead']), 1)
        self.assertIsNot(targets, self.targets)
        self.assert_source_unchanged()


if __name__ == '__main__':
    unittest.main()
