"""Independent reset/settling fidelity for the separate contact-screen protocol."""
import unittest
import mujoco
import numpy as np

from simulation_lab.direct_contact_dinner import draw_candidate
from simulation_lab.random_dinner import draw, assess
from simulation_lab.scene import build_scene


class DirectContactDinnerContracts(unittest.TestCase):
    def test_candidates_preserve_original_draws_and_every_settling_step(self):
        model = mujoco.MjModel.from_xml_string(build_scene(seed=42, scenario='dinner', dinner_preset='task')[0])
        original = {key: getattr(model, key).copy() for key in
                    ('body_pos', 'geom_size', 'geom_friction', 'jnt_range', 'actuator_gainprm')}
        records, events = [], []
        def record(index, recipe, result, trace):
            events.append(('record', index)); records.append((recipe, result, trace))
        data, selected = draw_candidate(model, 71, maximum_candidates=2, record=record,
                                         before_candidate=lambda index: events.append(('before', index)))
        self.assertTrue(records)
        for index, (recipe, result, trace) in enumerate(records):
            self.assertEqual(events[index*2:index*2+2], [('before', index), ('record', index)])
            independently_drawn, original_recipe = draw(model, 71+index)
            self.assertEqual(recipe, original_recipe)
            np.testing.assert_array_equal(trace['qpos'][0], independently_drawn.qpos)
            self.assertTrue(result['settling_replay_exact'])
            for tick, ctrl in enumerate(trace['ctrl']):
                independently_drawn.ctrl[:] = ctrl; mujoco.mj_step(model, independently_drawn)
                np.testing.assert_array_equal(trace['qpos'][tick+1], independently_drawn.qpos)
                np.testing.assert_array_equal(trace['qvel'][tick+1], independently_drawn.qvel)
            mujoco.mj_forward(model, independently_drawn)
            self.assertEqual(result['geometry'], assess(model, independently_drawn))
        np.testing.assert_array_equal(data.qpos, records[-1][2]['qpos'][-1])
        np.testing.assert_array_equal(data.qvel, records[-1][2]['qvel'][-1])
        self.assertEqual(bool(selected['generated']), records[-1][1]['selected'])
        self.assertFalse(any(row[1]['selected'] for row in records[:-1]))
        for key, value in original.items():
            np.testing.assert_array_equal(getattr(model, key), value)
        self.assertEqual(model.neq, 0)
        self.assertFalse(np.any(data.xfrc_applied)); self.assertFalse(np.any(data.qfrc_applied))


if __name__ == '__main__':
    unittest.main()
