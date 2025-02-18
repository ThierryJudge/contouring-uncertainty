import functools
import functools

import numpy as np
from scipy import ndimage
from skimage.measure import find_contours

from vital.data.camus.config import Label
from vital.utils.image.us.measure import EchoMeasure


def big_blob(seg):
    seg_out = np.zeros_like(seg)
    for label in [1]:
        # Find each blob in the image
        lbl, num = ndimage.measurements.label(np.isin(seg, label))

        # Count the number of elements per label
        count = np.bincount(lbl.flat)

        if not np.any(count[1:]):
            return seg

        # Select the largest blob
        maxi = np.argmax(count[1:]) + 1

        # Remove the other blobs
        lbl[lbl != maxi] = 0

        seg_out[np.where(lbl)] = label

    return seg_out


def get_contour_from_mask(mask, find_base=False):
    mask = mask.squeeze()
    if np.sum(mask)==0:
        return np.zeros((1, 2))

    if find_base:
        base_fn = functools.partial(
            EchoMeasure._extract_landmarks_from_polar_contour,
            labels=Label.LV,
            polar_smoothing_factor=5e-3,  # 5e-3 was determined empirically
            apex=False,
        )
        try:
            contour = EchoMeasure._endo_epi_contour(mask, Label.LV, base_fn)
        except:
            return np.zeros((1, 2))
    else:
        structure_mask = np.isin(mask, Label.LV)
        contour = find_contours(structure_mask, level=0.9)[0]

    return contour