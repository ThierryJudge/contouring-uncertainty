from typing import Literal

import numpy as np

from contour_uncertainty.utils.contour import contour_spline
from vital.data.camus.config import Label
from vital.utils.image.us.measure import EchoMeasure
from scipy.spatial import distance


def lv_FAC(ed_mask: np.ndarray, es_mask: np.ndarray) -> float:
    """Computes fractional area change.

    FAC = (ED_area - ES_area) / ED_area
    Args:
        ed_mask: array (H,W) mask for ED instant
        es_mask: array (H,W) mask for ES instant

    Returns:
        fractional area change
    """
    ed_area = lv_area(ed_mask)
    es_area = lv_area(es_mask)

    # from matplotlib import pyplot as plt
    # plt.imshow(ed_mask.squeeze())
    # plt.show()

    return (ed_area - es_area) / ed_area


def perimeter(contours):
    """

    Args:
        contours: ([N], K, 2) points

    Returns:

    """
    if contours.ndim == 2:
        c = contour_spline(contours)
        return np.sum([distance.euclidean(c[i], c[i + 1]) for i in range(c.shape[0] - 1)])
    else:
        p = []
        for contour in contours:
            c = contour_spline(contour)
            p.append(np.sum([distance.euclidean(c[i], c[i + 1]) for i in range(c.shape[0] - 1)]))
        return np.array(p)


def global_longitudinal_strain(ed_contour: np.ndarray, es_contour: np.ndarray, spline: bool = True) -> float:
    """ Computes global strain.

    Args:
        ed_contour: array (K,2) of points for ED contour
        es_contour: array (K,2) of points for ES contour
        spline: bool, whether to us a spline on the contours.

    Returns:
        global strain
    """
    if spline:
        ed_contour = contour_spline(ed_contour)
        es_contour = contour_spline(es_contour)


    ed_len = np.sum([distance.euclidean(ed_contour[i], ed_contour[i + 1]) for i in range(len(ed_contour) - 1)])
    es_len = np.sum([distance.euclidean(es_contour[i], es_contour[i + 1]) for i in range(len(es_contour) - 1)])

    return (ed_len - es_len) / ed_len


def compute_gls(frames):
    lv_longitudinal_lengths = perimeter(frames)

    # Compute the GLS for each frame in the sequence
    ed_lv_longitudinal_length = lv_longitudinal_lengths[0]
    gls = ((lv_longitudinal_lengths - ed_lv_longitudinal_length) / ed_lv_longitudinal_length) * 100
    return gls


def compute_FAC(frames):
    lv_areas = lv_area(frames, voxelarea=None)

    # Compute the GLS for each frame in the sequence
    ed_lv_area = lv_areas[0]
    fac = ((lv_areas - ed_lv_area) / ed_lv_area) * 100
    return fac


def lv_area(mask, voxelarea=None):
    return EchoMeasure.structure_area(mask, Label.LV, voxelarea=voxelarea)


def metric_error(prediction: float, gt: float, type: Literal['absolute', 'relative'] = 'absolute') -> float:
    """Computes percentage error between two values.

    Args:
        prediction: predicted value
        gt: groundtruth value
        type: Type of error; relative of absolute

    Returns:
        percentage error
    """
    error = np.abs(prediction - gt)
    if type == 'relative':
        error /= gt
    return error
