from typing import Literal

import numpy as np

from contour_uncertainty.data.lung.config import RLUNG, HEART, LLUNG, Label
from contour_uncertainty.utils.contour import reconstruction, linear_reconstruction
from contour_uncertainty.utils.skew_umap import skew_umap
from contour_uncertainty.utils.umap import uncertainty_map
from contour_uncertainty.data.utils import ContourToMask, UMap, SkewUMap
from matplotlib import pyplot as plt


def split_landmarks(landmarks):
    p1 = RLUNG
    p2 = p1 + LLUNG
    p3 = p2 + HEART

    rl = landmarks[:p1]
    ll = landmarks[p1:p2]
    h = landmarks[p2:p3]

    return rl, ll, h


class LungContourToMask(ContourToMask):
    @staticmethod
    def __call__(landmarks, shape=(256, 256), labels=None, apply_argmax: bool = True,
                 reconstruction_type: Literal['spline', 'linear'] = 'spline'):

        if reconstruction_type == 'linear':
            rec = linear_reconstruction
        elif reconstruction_type == 'spline':
            rec = reconstruction
        else:
            raise ValueError

        landmarks = landmarks.round().astype(int).squeeze()

        assert landmarks.ndim == 2 and landmarks.shape[1] == 2

        rl, ll, h = split_landmarks(landmarks)

        rl_mask = rec(rl, shape[0], shape[1])
        ll_mask = rec(ll, shape[0], shape[1])
        h_mask = rec(h, shape[0], shape[1])

        seg_map = np.zeros((3,) + shape, dtype=int)

        seg_map[Label.LUNG] += rl_mask
        seg_map[Label.LUNG] += ll_mask
        seg_map[Label.HEART] = h_mask
        seg_map[Label.BG] = seg_map.sum(0) == 0

        # f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
        # ax1.imshow(seg_map[Label.BG], cmap='gray')
        # ax2.imshow(seg_map[Label.LUNG], cmap='gray')
        # ax3.imshow(seg_map[Label.HEART], cmap='gray')
        # ax4.imshow(seg_map.argmax(0))
        # plt.show()

        # f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
        # ax1.imshow(rl_mask, cmap='gray')
        # ax2.imshow(ll_mask, cmap='gray')
        # ax3.imshow(h_mask, cmap='gray')
        # ax4.imshow(seg_map)

        if apply_argmax:
            seg_map = seg_map.argmax(0)

        return seg_map


class LungUMap(UMap):
    @staticmethod
    def __call__(mu, cov, labels=None):
        mu_rl, mu_ll, mu_h = split_landmarks(mu)
        cov_rl, cov_ll, cov_h = split_landmarks(cov)

        _, rl_umap = skew_umap(mu_rl, cov_rl, np.zeros_like(mu_rl))
        _, ll_umap = skew_umap(mu_ll, cov_ll, np.zeros_like(mu_ll))
        _, h_umap = skew_umap(mu_h, cov_h, np.zeros_like(mu_h))

        umap = np.clip(rl_umap / rl_umap.max() + ll_umap / ll_umap.max() + h_umap / h_umap.max(), a_min=0, a_max=1)
        # umap = umap / 2

        # f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
        # ax1.imshow(rl_umap)
        # ax2.imshow(ll_umap)
        # ax3.imshow(h_umap)
        # ax4.imshow(umap)

        return umap


class LungSkewUmap(SkewUMap):
    @staticmethod
    def __call__(mu, cov, alpha, labels=None):
        mu_rl, mu_ll, mu_h = split_landmarks(mu)
        cov_rl, cov_ll, cov_h = split_landmarks(cov)
        alpha_rl, alpha_ll, alpha_h = split_landmarks(alpha)

        rl_projected_mode, rl_umap = skew_umap(mu_rl, cov_rl, alpha_rl)
        ll_projected_mode, ll_umap = skew_umap(mu_ll, cov_ll, alpha_ll)
        h_projected_mode, h_umap = skew_umap(mu_h, cov_h, alpha_h)

        umap = np.clip(rl_umap / rl_umap.max() + ll_umap / ll_umap.max() + h_umap / h_umap.max(), a_min=0, a_max=1)
        # umap = umap / 2

        projected_mode = np.concatenate([rl_projected_mode, ll_projected_mode, h_projected_mode], axis=0)

        # f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
        # ax1.imshow(rl_umap)
        # ax2.imshow(ll_umap)
        # ax3.imshow(h_umap)
        # ax4.imshow(umap)

        return projected_mode, umap
