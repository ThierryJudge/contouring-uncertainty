import random
from typing import List, Dict, Callable

import torch
import torchvision.transforms.functional as F

from contour_uncertainty.augmentations.augmentation import Augmentation, to_tuple


class RandomRotation(Augmentation):
    def __init__(self, degrees, image_shape=(256, 256)):
        super().__init__()
        self.degrees = to_tuple(degrees)
        self.image_shape = image_shape

    def apply_img(self, img: torch.Tensor, angle):
        return F.rotate(img, angle)

    def apply_mask(self, mask: torch.Tensor, angle):
        if mask.ndim < 3:
            mask = mask[None]
            return F.rotate(mask, angle).squeeze(0)
        else:
            return F.rotate(mask, angle)

    def apply_keypoints(self, keypoints: torch.Tensor, angle):
        return self._rotate_keypoints(keypoints, angle)

    def un_apply_img(self, img: torch.Tensor, angle):
        return F.rotate(img, -angle)

    def un_apply_mask(self, mask: torch.Tensor, angle):
        if mask.ndim < 3:
            mask = mask[None]
            return F.rotate(mask, -angle).squeeze(0)
        else:
            return F.rotate(mask, -angle)

    def un_apply_keypoints(self, keypoints: torch.Tensor, angle):
        return self._rotate_keypoints(keypoints, -angle)

    def _rotate_keypoints(self, keypoints, angle):
        offset_x, offset_y = self.image_shape[1] / 2, self.image_shape[0] / 2
        adjusted_x = (keypoints[..., 0] - offset_x)
        adjusted_y = (keypoints[..., 1] - offset_y)

        angle = torch.deg2rad(torch.tensor(angle))
        c, s = torch.cos(angle), torch.sin(angle)

        keypoints = torch.zeros_like(keypoints)

        qx = offset_x + c * adjusted_x + s * adjusted_y
        qy = offset_y + -s * adjusted_x + c * adjusted_y

        keypoints[..., 0] = qx
        keypoints[..., 1] = qy

        return keypoints

    def get_params(self):
        angle = float(torch.empty(1).uniform_(float(self.degrees[0]), float(self.degrees[1])).item())
        return {'angle': angle}


class RandomTranslation(Augmentation):
    def __init__(self, dx=0, dy=0):
        super().__init__()
        self.dx = to_tuple(dx)
        self.dy = to_tuple(dy)

    def apply_img(self, img: torch.Tensor, tx, ty):
        return F.affine(img, translate=[tx, ty], angle=0, scale=1, shear=0)

    def apply_mask(self, mask: torch.Tensor, tx, ty):
        if mask.ndim < 3:
            mask = mask[None]
            return F.affine(mask, translate=[tx, ty], angle=0, scale=1, shear=0).squeeze(0)
        else:
            return F.affine(mask, translate=[tx, ty], angle=0, scale=1, shear=0)

    def apply_keypoints(self, keypoints: torch.Tensor, tx, ty):
        return self._translate_keypoints(keypoints, tx, ty)

    def un_apply_img(self, img: torch.Tensor, tx, ty):
        return F.affine(img, translate=[-tx, -ty], angle=0, scale=1, shear=0)

    def un_apply_mask(self, mask: torch.Tensor, tx, ty):
        if mask.ndim < 3:
            mask = mask[None]
            return F.affine(mask, translate=[-tx, -ty], angle=1, scale=1, shear=0).squeeze(0)
        else:
            return F.affine(mask, translate=[-tx, -ty], angle=1, scale=1, shear=0)

    def un_apply_keypoints(self, keypoints: torch.Tensor, tx, ty):
        return self._translate_keypoints(keypoints, -tx, -ty)

    def _translate_keypoints(self, keypoints, tx, ty):

        keypoints = torch.clone(keypoints)
        keypoints[..., 0] += tx
        keypoints[..., 1] += ty

        return keypoints

    def get_params(self):
        return {
            'tx': random.randint(self.dx[0], self.dx[1]),
            'ty': random.randint(self.dy[0], self.dy[1])
        }
