"""Calibrated RGB-D camera input for shared spatial pick/place policies.

This module does not read body positions, segmentation or object identities.
Depth is MuJoCo's metric optical-axis depth, not Euclidean ray length.
Unknown heightmap cells remain explicitly unknown to an obstacle planner.
"""
from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from .rgb_servo_cameras import calibration
from .scene import TABLE_Z


@dataclass(frozen=True)
class TableGrid:
    x_min: float = -.5
    y_min: float = -.33
    pixel_m: float = .003125
    rows: int = 256
    columns: int = 320
    table_z: float = TABLE_Z
    maximum_height_m: float = .55

    def pixel(self, xyz):
        """World coordinates to (row, column); no silent clipping at borders."""
        xyz = np.asarray(xyz)
        return np.floor((xyz[..., [1, 0]]-[self.y_min, self.x_min])/self.pixel_m).astype(int)

    def world(self, rc, height):
        """Cell-center world XYZ; height is relative to tabletop."""
        rc = np.asarray(rc)
        xy = (rc[..., [1, 0]]+.5)*self.pixel_m+[self.x_min, self.y_min]
        return np.concatenate((xy, np.broadcast_to(np.asarray(height)+self.table_z,
                                                  xy.shape[:-1])[..., None]), axis=-1)


def unproject(depth, projection):
    """Return H×W×3 world points from optical-axis depth and a 3×4 camera P."""
    depth = np.asarray(depth)
    matrix = np.asarray(projection, dtype=np.float64)
    if depth.ndim != 2 or matrix.shape != (3, 4):
        raise ValueError('Expected a depth image and a 3 by 4 projection matrix.')
    a = matrix[:, :3]
    center = np.linalg.solve(a, -matrix[:, 3])
    v, u = np.indices(depth.shape, dtype=np.float64)
    rays = np.stack((u+.5, v+.5, np.ones_like(u)), axis=-1) @ np.linalg.inv(a).T
    forward = a[2]/np.linalg.norm(a[2])
    return center+rays*(depth/(rays@forward))[..., None]


def fuse(views, grid=TableGrid()):
    """Highest visible surface per metric cell; inputs contain camera data only."""
    colors, positions, sources = [], [], []
    for index, view in enumerate(views):
        depth, rgb = np.asarray(view['depth_m']), np.asarray(view['rgb'])
        if rgb.shape != (*depth.shape, 3):
            raise ValueError('RGB and depth shapes disagree.')
        valid = np.isfinite(depth) & (depth > 0)
        positions.append(unproject(depth, view['calibration']['projection'])[valid])
        colors.append(rgb[valid]); sources.append(np.full(int(valid.sum()), index, dtype=np.int16))
    if not positions:
        raise ValueError('At least one calibrated camera is required.')
    xyz = np.concatenate(positions); rgb = np.concatenate(colors); source = np.concatenate(sources)
    rc = grid.pixel(xyz); h = xyz[:, 2]-grid.table_z
    valid = ((rc[:, 0] >= 0) & (rc[:, 0] < grid.rows) & (rc[:, 1] >= 0)
             & (rc[:, 1] < grid.columns) & (h >= -.005) & (h <= grid.maximum_height_m))
    xyz, rgb, source, rc, h = xyz[valid], rgb[valid], source[valid], rc[valid], h[valid]
    flat = rc[:, 0]*grid.columns+rc[:, 1]
    # Deterministic highest-Z reduction, keeping RGB from that same surface.
    order = np.lexsort((np.arange(len(h)), h, flat))
    keep = order[np.r_[flat[order][1:] != flat[order][:-1], True]] if len(order) else order
    shape = (grid.rows, grid.columns)
    color = np.zeros((*shape, 3), dtype=np.uint8)
    height = np.zeros(shape, dtype=np.float32)
    observed = np.zeros(shape, dtype=bool)
    camera = np.full(shape, -1, dtype=np.int16)
    r, c = rc[keep].T
    color[r, c] = rgb[keep]; height[r, c] = np.maximum(h[keep], 0)
    observed[r, c] = True; camera[r, c] = source[keep]
    policy = np.concatenate((color.astype(np.float32), np.repeat(height[..., None], 3, axis=-1)), axis=-1)
    return {'rgb': color, 'height_m': height, 'observed': observed, 'camera_index': camera,
            'policy_image': policy, 'grid': asdict(grid),
            'observation_contract': 'calibrated_rgb_and_metric_optical_axis_depth_only'}


class TableObserver:
    """Three fixed whole-table cameras, freshly rendered at each observation."""
    def __init__(self, model, width=640, height=480, grid=TableGrid()):
        self.width, self.height, self.grid = width, height, grid
        # MSAA resolves depth at a sample location rather than the pixel center.
        # Create this sensor's framebuffer without MSAA; restore the caller's
        # render setting immediately and leave existing renderers unchanged.
        samples = int(model.vis.quality.offsamples)
        try:
            model.vis.quality.offsamples = 0
            self.renderer = mujoco.Renderer(model, width=width, height=height)
        finally:
            model.vis.quality.offsamples = samples
        self.cameras = []
        for name, azimuth, elevation, distance in (
            ('table_top', 90., -90., 1.32),
            ('table_front', 45., -60., 1.52),
            ('table_back', 225., -60., 1.52),
        ):
            cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = [0., .07, TABLE_Z+.045]
            cam.azimuth, cam.elevation, cam.distance = azimuth, elevation, distance
            self.cameras.append((name, cam))

    def observe(self, data):
        views = []
        for name, camera in self.cameras:
            self.renderer.disable_depth_rendering()
            self.renderer.update_scene(data, camera=camera)
            rgb = self.renderer.render().copy()
            info = calibration(self.renderer, width=self.width, height=self.height)
            self.renderer.enable_depth_rendering()
            depth = self.renderer.render().copy()
            self.renderer.disable_depth_rendering()
            views.append({'name': name, 'rgb': rgb, 'depth_m': depth, 'calibration': info})
        return {'time_s': float(data.time), 'views': views, **fuse(views, self.grid)}

    def close(self):
        self.renderer.close()
