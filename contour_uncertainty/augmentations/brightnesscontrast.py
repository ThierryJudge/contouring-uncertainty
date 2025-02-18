import random
import torchvision.transforms.functional as F
import torch

from contour_uncertainty.augmentations.augmentation import Augmentation, to_tuple


class RandomBrightnessContrast(Augmentation):
    def __init__(self, brightness_limit=0, contrast_limit=0):
        super().__init__()
        self.brightness_limit = to_tuple(brightness_limit)
        self.contrast_limit = to_tuple(contrast_limit)

    def apply_img(self, img: torch.Tensor, alpha, beta):
        img = F.adjust_brightness(img, alpha)
        img = F.adjust_contrast(img, beta)

        return img

    def un_apply_img(self, img: torch.Tensor, alpha, beta):
        return img

    def get_params(self):
        return {
            "alpha": 1.0 + random.uniform(self.contrast_limit[0], self.contrast_limit[1]),
            "beta": 1.0 + random.uniform(self.brightness_limit[0], self.brightness_limit[1]),
        }
