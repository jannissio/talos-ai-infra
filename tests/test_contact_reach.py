"""Analytic contact certificates, with actual V24 source-pose regression cases.

All pose changes below are isolated geometry fixtures; no physics is stepped.
Actual source trace hashes and full proof inputs are archived in
docs/robotics/evidence/sideplate-pregrasp-development-v1.json.
"""
import unittest

import mujoco
import numpy as np

from simulation_lab.contact_reach import arm_contact_envelope, initial_direct_contact_separation
from simulation_lab.scene import build_scene


SIDE_PLATE = [-.04080098941043709, .2982586881003637, .7719970852895245,
              -7.420500630088884e-11, -.868342891750482, .49596433576055776, 3.382805438986538e-11]
SUCCESSFUL_SOURCES = [
    ('bottle', 'left', [.10684266919036915, -.200591748737762, .8999956279910318,
                       4.293642004158659e-12, -.8889589629120833, .4579868581720145, 9.481152921331659e-12]),
    ('bottle', 'right', [.07374282660156148, -.08768291842165883, .7599434946054586,
                        .8076051567156636, 1.6504561589380154e-7, -1.9883601963298624e-5, .5897235881757511]),
    ('glass', 'right', [.18718572251261023, -.10532909536143774, .7599425916340087,
                       .9726839488060128, -2.5527461804131476e-5, 6.561366470235799e-6, -.23213344231376717]),
    ('glass', 'right', [.21986422728686664, .08381924287173252, .7599426813870779,
                       .5938120684990729, -1.4545639788870183e-5, 9.054118918368493e-6, -.8046037701945601]),
]


class ContactReachTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_string(build_scene(seed=42, scenario='dinner', dinner_preset='task')[0])

    def setUp(self):
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

    def test_actual_sideplate_is_separated_from_both_complete_arms(self):
        for side, expected in (('left', .02890128243722434), ('right', .06488772187666225)):
            result = initial_direct_contact_separation(self.model, self.data, 'side_plate', side, pose=SIDE_PLATE)
            self.assertTrue(result['excluded'])
            self.assertEqual(result['classification'], 'initial_direct_contact_separated')
            self.assertAlmostEqual(result['gap_m'], expected, places=9)

    def test_known_physical_bottle_and_glass_sources_are_not_excluded(self):
        for item, side, pose in SUCCESSFUL_SOURCES:
            with self.subTest(item=item, side=side, pose=pose[:3]):
                result = initial_direct_contact_separation(self.model, self.data, item, side, pose=pose)
                self.assertFalse(result['excluded'])
                self.assertEqual(result['classification'], 'unresolved')

    def test_arbitrary_pose_query_matches_reset_geometry_and_mutates_nothing(self):
        before_data = {name: getattr(self.data, name).copy() for name in
                       ('qpos', 'qvel', 'ctrl', 'xpos', 'xmat', 'xanchor', 'xaxis', 'geom_xpos', 'geom_xmat')}
        before_model = {name: getattr(self.model, name).copy() for name in
                        ('body_pos', 'body_quat', 'jnt_pos', 'jnt_axis', 'jnt_range', 'geom_pos',
                         'geom_quat', 'geom_size', 'geom_contype', 'geom_conaffinity', 'mesh_vert')}
        for tilt in (0., np.pi / 2, np.pi):
            for yaw in (-np.pi, -.31, 0., 2.9):
                quat = [np.cos(yaw / 2) * np.cos(tilt / 2), np.cos(yaw / 2) * np.sin(tilt / 2),
                        np.sin(yaw / 2) * np.sin(tilt / 2), np.sin(yaw / 2) * np.cos(tilt / 2)]
                pose = np.r_[SIDE_PLATE[:3], quat]
                scratch = mujoco.MjData(self.model)
                scratch.qpos[:] = self.data.qpos
                scratch.joint('side_plate_free').qpos[:] = pose
                mujoco.mj_forward(self.model, scratch)
                query = initial_direct_contact_separation(self.model, self.data, 'side_plate', 'left', pose=pose)
                reference = initial_direct_contact_separation(self.model, scratch, 'side_plate', 'left')
                self.assertAlmostEqual(query['gap_m'], reference['gap_m'], places=12)
        for name, values in before_data.items():
            np.testing.assert_array_equal(getattr(self.data, name), values)
        for name, values in before_model.items():
            np.testing.assert_array_equal(getattr(self.model, name), values)

    def test_complete_hull_envelope_contains_actual_configurations(self):
        # Independent state samples test the invariant chain construction;
        # the certificate itself is analytic, never inferred from samples.
        rng = np.random.default_rng(9221)
        envelope = arm_contact_envelope(self.model, self.data, 'left')
        center, axis = np.asarray(envelope['circle_center']), np.asarray(envelope['axis'])
        for _ in range(12):
            scratch = mujoco.MjData(self.model)
            for joint in range(self.model.njnt):
                if self.model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE:
                    scratch.qpos[self.model.jnt_qposadr[joint]] = rng.uniform(-np.pi, np.pi)
            mujoco.mj_forward(self.model, scratch)
            normal = rng.normal(size=3)
            normal /= np.linalg.norm(normal)
            for row in envelope['geoms']:
                geom = row['geom_id']
                if self.model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
                    continue
                mesh = int(self.model.geom_dataid[geom])
                start, count = int(self.model.mesh_vertadr[mesh]), int(self.model.mesh_vertnum[mesh])
                local = self.model.mesh_vert[start:start + count]
                world = local @ scratch.geom_xmat[geom].reshape(3, 3).T + scratch.geom_xpos[geom]
                predicted = (envelope['circle_radius_m'] * np.sqrt(max(0., 1 - float(normal @ axis)**2))
                             if row['circle_centered'] else normal @ (np.asarray(envelope['root_anchor']) - center))
                predicted += row['radius_m']
                self.assertLessEqual(float(np.max((world - center) @ normal)), predicted + 1e-9)


if __name__ == '__main__':
    unittest.main()
