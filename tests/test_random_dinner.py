"""Independent geometry and reset-isolation checks for joint tabletop scenes."""
import itertools
import unittest
import mujoco
import numpy as np
from simulation_lab.dinner import OBJECTS
from simulation_lab.scene import build_scene
from simulation_lab.random_dinner import assess, body_bounds, draw, half_extents, matrix, orientation


class RandomDinnerContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_string(build_scene(seed=42, scenario='dinner', dinner_preset='task')[0])

    def test_rotated_box_support_matches_all_vertices(self):
        rotation = matrix(orientation('sideways', .73, -.51)); size = np.array([.07, .02, .05])
        vertices = np.asarray(list(itertools.product((-1, 1), repeat=3)))*size @ rotation.T
        np.testing.assert_allclose(half_extents(mujoco.mjtGeom.mjGEOM_BOX, size, rotation), np.abs(vertices).max(axis=0), atol=1e-15)

    def test_rotated_cylinder_support_bounds_dense_surface(self):
        rotation = matrix(orientation('sideways', -.31, .21)); angle = np.linspace(0, 2*np.pi, 20001)
        vertices = np.concatenate([np.c_[.03*np.cos(angle), .03*np.sin(angle), np.full(len(angle), z)] for z in (-.05, .05)]) @ rotation.T
        predicted = half_extents(mujoco.mjtGeom.mjGEOM_CYLINDER, [.03, .05, 0], rotation)
        np.testing.assert_allclose(predicted, np.abs(vertices).max(axis=0), atol=1e-8)

    def test_draw_is_repeatable_and_accounts_for_all_items(self):
        a, first = draw(self.model, 71); b, second = draw(self.model, 71)
        self.assertEqual(first, second); self.assertTrue(first['generated'])
        self.assertEqual(set(first['objects']), set(OBJECTS))
        np.testing.assert_array_equal(a.qpos, b.qpos)
        self.assertEqual(self.model.neq, 0)
        self.assertFalse(np.any(a.xfrc_applied)); self.assertFalse(np.any(a.qfrc_applied))
        for name, row in first['objects'].items():
            self.assertTrue(row['candidate_arms']); self.assertIsNone(row['rejection'])

    def test_supported_sideways_bottle_is_not_rejected_as_non_upright(self):
        data, _ = draw(self.model, 71)
        for i, name in enumerate(OBJECTS):
            adr = int(self.model.joint(name+'_free').qposadr[0])
            data.qpos[adr:adr+3] = [2+i, 0, .8]
        quat = orientation('sideways', 0.)
        low, _ = body_bounds(self.model, 'bottle', quat)
        adr = int(self.model.joint('bottle_free').qposadr[0])
        data.qpos[adr:adr+7] = [0., -.01, .76-low[2]+.001, *quat]
        data.qvel[:] = 0.; mujoco.mj_forward(self.model, data)
        for _ in range(600):mujoco.mj_step(self.model, data)
        row = assess(self.model, data)['objects']['bottle']
        self.assertTrue(row['valid'], row)
        self.assertEqual(row['realized_family'], 'sideways')


if __name__ == '__main__':unittest.main()
