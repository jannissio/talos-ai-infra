"""Independent metric/depth geometry checks for the shared sensor interface."""
import unittest

import mujoco
import numpy as np

from simulation_lab.rgb_servo_cameras import calibration
from simulation_lab.table_observation import TableGrid, fuse, unproject


class TableObservationTests(unittest.TestCase):
    def test_optical_depth_is_not_ray_length(self):
        # Camera at (1, 2, 3), looking along world +Z, focal length 2 pixels.
        k = np.array([[2., 0., 1.], [0., 2., 1.], [0., 0., 1.]])
        p = k@np.column_stack((np.eye(3), -np.array([1., 2., 3.])))
        xyz = unproject(np.full((2, 2), 4.), p)
        np.testing.assert_allclose(xyz, [[[0., 1., 7.], [2., 1., 7.]],
                                        [[0., 3., 7.], [2., 3., 7.]]])

    def test_grid_roundtrip_and_outside(self):
        g = TableGrid()
        rc = np.array([[0, 0], [100, 111], [255, 319]])
        np.testing.assert_array_equal(g.pixel(g.world(rc, [.01, .1, .2])), rc)
        self.assertEqual(g.pixel([g.x_min-.01, g.y_min, 0.])[1], -4)

    def test_rendered_box_has_correct_world_height_and_pixel_center(self):
        # Deliberately tilted camera tests depth convention and image-axis signs.
        xml = '''<mujoco><visual><quality offsamples="0"/><global offwidth="160" offheight="120"/></visual>
          <worldbody><light pos="0 0 3"/><geom type="plane" size="3 3 .1"/>
          <geom type="box" size=".2 .2 .1" pos="0 0 .1" rgba="1 0 0 1"/>
          <camera name="test" pos="0 -1 1" xyaxes="1 0 0 0 .70710678 .70710678"/>
          </worldbody></mujoco>'''
        model = mujoco.MjModel.from_xml_string(xml); data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        renderer = mujoco.Renderer(model, width=160, height=120)
        try:
            renderer.update_scene(data, camera='test')
            info = calibration(renderer, 160, 120); rgb = renderer.render().copy()
            renderer.enable_depth_rendering(); depth = renderer.render().copy()
            xyz = unproject(depth, info['projection'])
            top = ((np.abs(xyz[..., 0]) < .17) & (np.abs(xyz[..., 1]) < .17)
                   & (xyz[..., 2] > .15))
            self.assertGreater(top.sum(), 100)
            self.assertLess(float(np.max(np.abs(xyz[top, 2]-.2))), 2e-5)
            grid = TableGrid(x_min=-.3, y_min=-.3, pixel_m=.01, rows=60, columns=60, table_z=0.)
            result = fuse([{'rgb': rgb, 'depth_m': depth, 'calibration': info}], grid)
            self.assertGreater(result['height_m'][result['observed']].max(), .1999)
            self.assertFalse(result['observed'].all())
        finally:
            renderer.close()


if __name__ == '__main__':
    unittest.main()
