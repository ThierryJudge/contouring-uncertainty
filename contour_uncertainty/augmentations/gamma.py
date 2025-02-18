import random
from typing import List, Dict, Callable

import torch
import torchvision.transforms.functional as F

from contour_uncertainty.augmentations.augmentation import Augmentation, to_tuple


class RandomGamma(Augmentation):

    def __init__(self, gamma_limit=(0.99, 1.01)):
        super().__init__()
        self.gamma_limit = to_tuple(gamma_limit)

    def apply_img(self, img: torch.Tensor, gamma: float):
        return F.adjust_gamma(img, gamma)

    def un_apply_img(self, img: torch.Tensor, gamma: float):
        return img

    def get_params(self):
        return {"gamma": random.uniform(self.gamma_limit[0], self.gamma_limit[1])}
