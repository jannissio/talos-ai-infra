"""Experimental live RGB observer. No simulator object state is an input."""
import numpy as np
import torch
from safetensors.torch import load_file
from .rgb_servo_network import BottleKeypointNet, decode_heatmaps
from .rgb_servo_cameras import VIEWS, triangulate, project


class RgbBottleObserver:
    def __init__(self, checkpoint, device='cpu'):
        self.device = torch.device(device)
        self.network = BottleKeypointNet().to(self.device).eval()
        self.network.load_state_dict(load_file(str(checkpoint), device=device))

    def observe(self, images, calibrations):
        if set(images) != set(VIEWS) or set(calibrations) != set(VIEWS):
            raise ValueError('Three declared RGB camera views and fixed calibration are required.')
        array = np.stack([images[name] for name in VIEWS])
        if array.shape != (3, 240, 320, 3) or array.dtype != np.uint8:
            raise ValueError('Expected three uint8 RGB images of shape 240x320x3.')
        tensor = torch.from_numpy(array.transpose(0, 3, 1, 2).copy()).to(self.device).float()/255
        with torch.inference_mode():
            maps, _, vis = self.network(tensor)
            coordinates, peak, variance = decode_heatmaps(maps)
        xy = coordinates.cpu().numpy(); peak = peak.cpu().numpy(); variance = variance.cpu().numpy()
        visibility = vis.sigmoid().cpu().numpy()
        valid = (visibility >= .7) & (peak.min(1) >= .02) & (variance.max(1) <= 36.)
        result = {'status': 'refused', 'source': 'Current RGB images and fixed camera calibration only',
                  'views': {name: {'accepted': bool(valid[i]), 'visibility': float(visibility[i]),
                                    'keypoints_px': xy[i].tolist(), 'peak': peak[i].tolist(), 'variance_px2': variance[i].tolist()}
                            for i, name in enumerate(VIEWS)}}
        if not valid.all():
            return {**result, 'reason': 'Three confident views are required.'}
        matrices = [calibrations[name]['projection'] for name in VIEWS]
        points = np.asarray([triangulate(xy[:, k], matrices) for k in range(2)])
        residual = max(float(np.linalg.norm(project(points, matrix)-xy[i], axis=1).max()) for i, matrix in enumerate(matrices))
        axis = points[1]-points[0]; length = float(np.linalg.norm(axis))
        result.update(reprojection_error_px=residual, axis_length_m=length)
        if residual > 1.5 or not .097 <= length <= .109 or axis[2] <= .075:
            return {**result, 'reason': 'Views do not agree with the known upright bottle geometry.'}
        return {**result, 'status': 'observed', 'keypoints_m': points.tolist(), 'grasp_point_m': points[1].tolist(),
                'axis': (axis/length).tolist(), 'yaw_estimated': False}
