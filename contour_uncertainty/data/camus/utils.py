from typing import Literal

import numpy as np

from contour_uncertainty.utils.contour import reconstruction, linear_reconstruction, contour_spline
from contour_uncertainty.utils.skew_umap import skew_umap
from contour_uncertainty.utils.umap import uncertainty_map
from vital.data.camus.config import Label
from matplotlib import pyplot as plt
from contour_uncertainty.data.utils import ContourToMask, UMap, SkewUMap
from skimage import draw


def split_landmarks(landmarks):
    p1 = len(landmarks) // 2
    p2 = p1 + len(landmarks) // 2

    lv = landmarks[:p1]
    myo = landmarks[p1:p2]

    return lv, myo


def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=int)
    mask[fill_row_coords, fill_col_coords] = 1
    return mask


class USContourToMask(ContourToMask):
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

        if Label.MYO in labels:
            lv, myo = split_landmarks(landmarks)

            lv_spline = contour_spline(lv, n=1000).round().astype(int)
            myo_spline = contour_spline(myo, n=1000).round().astype(int)
            polygon = np.concatenate([lv_spline, np.flip(myo_spline, axis=0)])

            # mask = poly2mask(polygon[:, 1], polygon[:, 0], shape)

            # plt.figure()
            # plt.imshow(mask)
            # plt.show()

            lv_mask = rec(lv, shape[0], shape[1])
            # myo_mask = rec(myo, shape[0], shape[1])
            myo_mask = poly2mask(polygon[:, 1], polygon[:, 0], shape)

            seg_map = np.zeros((3,) + shape, dtype=int)

            seg_map[Label.LV] = lv_mask
            seg_map[Label.MYO] = np.clip(myo_mask - lv_mask, a_min=0, a_max=1)
            seg_map[Label.BG] = seg_map.sum(0) == 0

            # seg_map = np.zeros(shape, dtype=int)
            # seg_map[np.where(myo_mask)] = Label.MYO
            # seg_map[np.where(lv_mask)] = Label.LV

            # f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
            # ax1.imshow(seg_map[Label.BG], cmap='gray')
            # ax2.imshow(seg_map[Label.LV], cmap='gray')
            # ax3.imshow(seg_map[Label.MYO], cmap='gray')
            # ax4.imshow(seg_map.argmax(0))
            # plt.show()

            if apply_argmax:
                seg_map = seg_map.argmax(0)


        else:
            seg_map = rec(landmarks, shape[0], shape[1])

            if not apply_argmax:
                # For consistency add channel dimension if not argmax is applied
                seg_map = seg_map[None]

        # f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
        # ax1.imshow(rl_mask, cmap='gray')
        # ax2.imshow(ll_mask, cmap='gray')
        # ax3.imshow(h_mask, cmap='gray')
        # ax4.imshow(seg_map)

        return seg_map


class USUMap(UMap):
    @staticmethod
    def __call__(mu, cov, labels=None):
        if Label.MYO in labels:
            mu_lv, mu_myo = split_landmarks(mu)
            cov_lv, cov_myo, = split_landmarks(cov)

            _, lv_umap = skew_umap(mu_lv, cov_lv, np.zeros_like(mu_lv))
            _, myo_umap = skew_umap(mu_myo, cov_myo, np.zeros_like(mu_myo))

            umap = np.clip(lv_umap / lv_umap.max() + myo_umap / myo_umap.max(), a_min=0, a_max=1)
            umap = umap / 2
        else:
            umap = uncertainty_map(mu, cov)
            umap = umap / umap.max()
            # umap = umap / 2

        # f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
        # ax1.imshow(lv_umap)
        # ax2.imshow(myo_umap)
        # ax3.imshow(h_umap)
        # ax4.imshow(umap)

        return umap


class USSkewUmap(SkewUMap):
    @staticmethod
    def __call__(mu, cov, alpha, labels=None):
        if Label.MYO in labels:
            mu_lv, mu_myo = split_landmarks(mu)
            cov_lv, cov_myo, = split_landmarks(cov)
            alpha_lv, alpha_myo = split_landmarks(alpha)

            lv_projected_mode, lv_umap = skew_umap(mu_lv, cov_lv, alpha_lv)
            myo_projected_mode, myo_umap = skew_umap(mu_myo, cov_myo, alpha_myo)

            umap = np.clip(lv_umap / lv_umap.max() + myo_umap / myo_umap.max(), a_min=0, a_max=1)
            umap = umap / 2

            projected_mode = np.concatenate([lv_projected_mode, myo_projected_mode], axis=0)
        else:
            projected_mode, umap = skew_umap(mu, cov, alpha, linear_close=True)
            umap = umap / umap.max()
            # umap = umap / 2

        # f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
        # ax1.imshow(rl_umap)
        # ax2.imshow(ll_umap)
        # ax3.imshow(h_umap)
        # ax4.imshow(umap)

        return projected_mode, umap
