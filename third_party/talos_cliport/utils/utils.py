# Selected Apache-2.0 utility definitions from CLIPort.
import numpy as np
import torch
import kornia

def preprocess(img, dist='transporter'):
    """Pre-process input (subtract mean, divide by std)."""

    transporter_color_mean = [0.18877631, 0.18877631, 0.18877631]
    transporter_color_std = [0.07276466, 0.07276466, 0.07276466]
    transporter_depth_mean = 0.00509261
    transporter_depth_std = 0.00903967

    franka_color_mean = [0.622291933, 0.628313992, 0.623031488]
    franka_color_std = [0.168154213, 0.17626014, 0.184527364]
    franka_depth_mean = 0.872146842
    franka_depth_std = 0.195743116

    clip_color_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_color_std = [0.26862954, 0.26130258, 0.27577711]

    # choose distribution
    if dist == 'clip':
        color_mean = clip_color_mean
        color_std = clip_color_std
    elif dist == 'franka':
        color_mean = franka_color_mean
        color_std = franka_color_std
    else:
        color_mean = transporter_color_mean
        color_std = transporter_color_std

    if dist == 'franka':
        depth_mean = franka_depth_mean
        depth_std = franka_depth_std
    else:
        depth_mean = transporter_depth_mean
        depth_std = transporter_depth_std

    # convert to pytorch tensor (if required)
    if type(img) == torch.Tensor:
        def cast_shape(stat, img):
            tensor = torch.from_numpy(np.array(stat)).to(device=img.device, dtype=img.dtype)
            tensor = tensor.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            tensor = tensor.repeat(img.shape[0], 1, img.shape[-2], img.shape[-1])
            return tensor

        color_mean = cast_shape(color_mean, img)
        color_std = cast_shape(color_std, img)
        depth_mean = cast_shape(depth_mean, img)
        depth_std = cast_shape(depth_std, img)

        # normalize
        img = img.clone()
        img[:, :3, :, :] = ((img[:, :3, :, :] / 255 - color_mean) / color_std)
        img[:, 3:, :, :] = ((img[:, 3:, :, :] - depth_mean) / depth_std)
    else:
        # normalize
        img[:, :, :3] = (img[:, :, :3] / 255 - color_mean) / color_std
        img[:, :, 3:] = (img[:, :, 3:] - depth_mean) / depth_std

    # if dist == 'franka' or dist == 'transporter':
    #     print(np.mean(img[:,:3,:,:].detach().cpu().numpy(), axis=(0,2,3)),
    #           np.mean(img[:,3,:,:].detach().cpu().numpy()))

    return img

class ImageRotator:
    """Rotate for n rotations."""
    # Reference: https://kornia.readthedocs.io/en/latest/tutorials/warp_affine.html?highlight=rotate

    def __init__(self, n_rotations):
        self.angles = []
        for i in range(n_rotations):
            theta = i * 2 * 180 / n_rotations
            self.angles.append(theta)

    def __call__(self, x_list, pivot, reverse=False):
        rot_x_list = []
        for i, angle in enumerate(self.angles):
            x = x_list[i].unsqueeze(0)

            # create transformation (rotation)
            alpha: float = angle if not reverse else (-1.0 * angle)  # in degrees
            angle: torch.tensor = torch.ones(1) * alpha

            # define the rotation center
            center: torch.tensor = torch.ones(1, 2)
            center[..., 0] = pivot[1]
            center[..., 1] = pivot[0]

            # define the scale factor
            scale: torch.tensor = torch.ones(1, 2)

            # compute the transformation matrix
            M: torch.tensor = kornia.geometry.transform.get_rotation_matrix2d(center, angle, scale)

            # apply the transformation to original image
            _, _, h, w = x.shape
            x_warped: torch.tensor = kornia.geometry.transform.warp_affine(x.float(), M.to(x.device), dsize=(h, w))
            x_warped = x_warped
            rot_x_list.append(x_warped)

        return rot_x_list
