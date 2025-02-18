import numpy as np
import scipy
from skimage.morphology import convex_hull_image

from contour_uncertainty.data.camus.measure import ContourMeasure
from contour_uncertainty.utils.contour import reconstruction
from vital.data.camus.config import Label
from vital.utils.image.us.measure import EchoMeasure


def get_contour_points(segmentation, points_dict):
    lv_points = lv_contour(segmentation, points_dict[Label.LV])
    myo_points = myo_contour(segmentation, points_dict[Label.MYO])
    # la_points = la_contour(segmentation, (lv_points[0], lv_points[-1]), points_dict[Label.ATRIUM])

    lv_points = np.flip(lv_points, axis=-1)
    myo_points = np.flip(myo_points, axis=-1)
    # la_points = np.flip(la_points, axis=-1)

    return lv_points, myo_points  # , la_points


def lv_contour(segmentation, nb_points):
    lv_edge = ContourMeasure.structure_edge(segmentation=segmentation, label=Label.LV)

    base = np.array(ContourMeasure._endo_base(segmentation, Label.LV, Label.MYO))
    apex = np.array(ContourMeasure.lv_apex(segmentation))

    path1 = ContourMeasure.get_path(lv_edge, tuple(apex), tuple(base[0]))
    path2 = ContourMeasure.get_path(lv_edge, tuple(apex), tuple(base[1]))

    points_per_side = (nb_points + 1) // 2

    path1_points_idx = np.linspace(0, len(path1) - 1, points_per_side).astype(int)
    path2_points_idx = np.linspace(0, len(path2) - 1, points_per_side).astype(int)

    points = np.concatenate(
        (
            base[0][None],
            path1[path1_points_idx[1:-1]],
            apex[None],
            path2[-path2_points_idx[1:-1]],
            base[1][None],
        ),
        axis=0,
    )
    return points


def myo_contour(segmentation, nb_points, label: int = Label.MYO):
    myo = np.isin(segmentation, Label.MYO)

    myo_convex = convex_hull_image(myo)

    myo_points = EchoMeasure._extract_landmarks_from_polar_contour(
        myo_convex, 1, polar_smoothing_factor=5e-3, debug_plots=False
    )

    myo_edge = ContourMeasure.structure_edge(segmentation=myo_convex, label=1)  # Mask only contains filled MYO

    myo_points = myo_points.round().astype(int)

    path1 = ContourMeasure.get_path(myo_edge, tuple(myo_points[0]), tuple(myo_points[1]))
    path2 = ContourMeasure.get_path(myo_edge, tuple(myo_points[0]), tuple(myo_points[2]))

    points_per_side = (nb_points + 1) // 2

    path1_points_idx = np.linspace(0, len(path1) - 1, points_per_side).astype(int)
    path2_points_idx = np.linspace(0, len(path2) - 1, points_per_side).astype(int)

    myo_points = np.concatenate(
        (
            myo_points[1][None],
            path1[path1_points_idx[1:-1]],
            myo_points[0][None],
            path2[-path2_points_idx[1:-1]],
            myo_points[2][None],
        ),
        axis=0,
    )
    return myo_points


def la_contour(segmentation, basal_points: tuple, nb_points):
    la_edge = ContourMeasure.structure_edge(segmentation=segmentation, label=Label.ATRIUM)

    # ADD points along edge if part of LA is outside image
    la_points = np.where(np.isin(segmentation, Label.ATRIUM))
    side = np.where(la_points[0] == 255)[0]
    la_edge[la_points[0][side], la_points[1][side]] = 1

    la_edge_points = np.where(la_edge)
    la_edge_points = np.concatenate((la_edge_points[0][..., None], la_edge_points[1][..., None]), axis=1)

    index1 = np.argmin(scipy.spatial.distance.cdist(la_edge_points, basal_points[0][None]))
    index2 = np.argmin(scipy.spatial.distance.cdist(la_edge_points, basal_points[1][None]))

    base1_temp = la_edge_points[index1]
    base2_temp = la_edge_points[index2]

    # print(contour[0], base1_temp)
    # print(contour[-1], base2_temp)
    # print(np.array([base1_temp, base2_temp]))

    base_mid = np.array([base1_temp, base2_temp]).mean(axis=0).round().astype(int)

    # Set midpoint between basal points to 0 in edge to not find in path.

    if base1_temp[1] < 254:
        la_edge[base_mid[0] - 2:base_mid[0] + 2, base_mid[1] - 2:base_mid[1] + 2] = 0
    # print('mid', base_mid)

    # from matplotlib import pyplot as plt
    # f, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.imshow(np.isin(segmentation, Label.ATRIUM), cmap='gray')
    #
    # ax1.scatter(basal_points[0][1], basal_points[0][0])
    # ax1.scatter(basal_points[1][1], basal_points[1][0])
    #
    # ax2.imshow(la_edge.squeeze(), cmap='gray')
    # plt.show()

    la_path = ContourMeasure.get_path(la_edge, tuple(base1_temp), tuple(base2_temp))
    # la_path = ContourMeasure.get_path(la_edge, tuple(base1_temp), tuple(base2_temp))

    path_points_idx = np.linspace(0, len(la_path) - 1, nb_points).astype(int)
    la_points = la_path[-path_points_idx[1:-1]]
    return la_points


def multiclass_reconstruction(points: np.ndarray, height: int, width: int, points_dict: dict):
    # points (K, 2)

    lv_points = points[:points_dict[Label.LV]]
    lv = reconstruction(lv_points, height, width)

    myo_points = points[points_dict[Label.LV]:points_dict[Label.LV] + points_dict[Label.MYO]]
    myo_full = reconstruction(myo_points, height, width)

    la_points = points[points_dict[Label.LV] + points_dict[Label.MYO]:]
    # la_points = la_points[1:-1]
    la_points = np.concatenate([lv_points[0][None], la_points, lv_points[-1][None]], axis=0)
    la = reconstruction(la_points, height, width)

    # f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    # ax1.imshow(lv.squeeze())
    # ax1.scatter(lv_points[:, 0], lv_points[:, 1])
    # for i in range(lv_points.shape[0]):
    #     ax1.annotate(str(i), (lv_points[i, 0], lv_points[i, 1]))
    #
    # ax2.imshow(myo_full.squeeze())
    # ax2.scatter(myo_points[:, 0], myo_points[:, 1])
    # for i in range(myo_points.shape[0]):
    #     ax2.annotate(str(i), (myo_points[i, 0], myo_points[i, 1]))
    #
    # ax3.imshow(la.squeeze())
    # ax3.scatter(la_points[:, 0], la_points[:, 1])
    # for i in range(la_points.shape[0]):
    #     ax3.annotate(str(i), (la_points[i, 0], la_points[i, 1]))
    #
    # plt.show()

    map = np.zeros((height, width))

    map[np.where(myo_full)] = Label.MYO
    map[np.where(lv)] = Label.LV
    map[np.where(la)] = Label.ATRIUM

    return map
