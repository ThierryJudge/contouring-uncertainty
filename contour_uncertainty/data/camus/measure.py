import logging
import sys
from collections import deque
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_fill_holes
from skimage import morphology
from skimage.draw import line
from skimage.morphology import convex_hull_image, erosion
from vital.data.camus.config import Label
from vital.data.config import SemanticStructureId
from vital.utils.decorators import auto_cast_data, batch_function
from vital.utils.image.measure import Measure, T
from vital.utils.image.us.measure import EchoMeasure

logger = logging.getLogger(__name__)


class ContourMeasure(EchoMeasure):

    @staticmethod
    @auto_cast_data
    @batch_function(item_ndim=2)
    def structure_apex(
        segmentation: T,
        label: SemanticStructureId,
        base_coords: Tuple,
    ) -> T:
        if np.isnan(base_coords).any():
            # Early return if we couldn't reliably estimate the landmarks at the base of the left ventricle
            return np.nan

        # Identify the midpoint at the base of the left ventricle
        base_mid = np.array(base_coords).mean(axis=0)

        # Compute the distance from all pixels in the image to `lv_base_midpoint`
        mask = np.ones_like(segmentation, dtype=bool)
        mask[tuple(base_mid.round().astype(int))] = False
        dist_to_base_mid = ndimage.distance_transform_edt(mask)

        # Find the point within the left ventricle mask with maximum distance
        strucure = np.isin(segmentation, label)
        apex_coords = np.unravel_index(np.argmax(dist_to_base_mid * strucure), segmentation.shape)

        return apex_coords

    @staticmethod
    @auto_cast_data
    @batch_function(item_ndim=2)
    def lv_apex(
        segmentation: T,
        lv_labels: SemanticStructureId = Label.LV.value,
        myo_labels: SemanticStructureId = Label.MYO.value,
    ) -> T:
        """Measures the LV length as the distance between the LV's base midpoint and its furthest point at the apex.

        Args:
            segmentation: ([N], H, W), Segmentation map.
            lv_labels: Labels of the classes that are part of the left ventricle.
            myo_labels: Labels of the classes that are part of the myocardium. The location of the myocardium is
                necessary to identify the markers at the base of the left ventricle.

        Returns:
            ([N], 1), Length of the left ventricle, or NaNs for the images where the LV base's midpoint cannot be
            reliably estimated.
        """
        # Identify the base of the left ventricle
        lv_base_coords = EchoMeasure._endo_base(segmentation, lv_labels=lv_labels, myo_labels=myo_labels)
        lv_apex_coords = ContourMeasure.structure_apex(segmentation, lv_labels, lv_base_coords)
        return lv_apex_coords

    @staticmethod
    @auto_cast_data
    @batch_function(item_ndim=2)
    def myo_apex(
        segmentation: T,
        myo_base_coords,
        myo_labels: SemanticStructureId = Label.MYO.value,
    ) -> T:
        """Measures the LV length as the distance between the LV's base midpoint and its furthest point at the apex.

        Args:
            segmentation: ([N], H, W), Segmentation map.
            lv_labels: Labels of the classes that are part of the left ventricle.
            myo_labels: Labels of the classes that are part of the myocardium. The location of the myocardium is
                necessary to identify the markers at the base of the left ventricle.

        Returns:
            ([N], 1), Length of the left ventricle, or NaNs for the images where the LV base's midpoint cannot be
            reliably estimated.
        """
        myo_apex_coords = ContourMeasure.structure_apex(segmentation, myo_labels, myo_base_coords)
        return myo_apex_coords

    @staticmethod
    def structure_edge(
        segmentation: np.ndarray,
        label: SemanticStructureId,
    ) -> np.ndarray:
        mask = np.isin(segmentation, label).astype(int)
        edge = mask ^ erosion(mask, footprint=np.ones((3, 3)))
        return edge

    @staticmethod
    def myo_edge(segmentation: np.ndarray, myo_labels: SemanticStructureId = Label.MYO) -> np.ndarray:
        myo_mask = np.isin(segmentation, myo_labels).astype(int)
        myo_mask = convex_hull_image(myo_mask)
        myo_edge = myo_mask ^ erosion(myo_mask, footprint=np.ones((3, 3)))
        return myo_edge

    @staticmethod
    def get_path(img, start, end):
        height, width = img.shape

        # All 8 directions
        delta = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]

        # Store the results of the BFS as the shortest distance to start
        grid = [[sys.maxsize for _ in range(width)] for _ in range(height)]
        grid[start[0]][start[1]] = 0

        # The actual BFS algorithm
        bfs = deque([start])
        found = False
        while len(bfs) > 0:
            y, x = bfs.popleft()
            # We've reached the end!
            if (y, x) == end:
                found = True
                break

            # Look all 8 directions for a good path
            for dy, dx in delta:
                yy, xx = y + dy, x + dx
                # If the next position hasn't already been looked at and it's white
                if 0 <= yy < height and 0 <= xx < width and grid[y][x] + 1 < grid[yy][xx] and img[yy][xx] != 0:
                    grid[yy][xx] = grid[y][x] + 1
                    bfs.append((yy, xx))

        if found:
            # Now rebuild the path from the end to beginning
            path = []
            y, x = end
            while grid[y][x] != 0:
                for dy, dx in delta:
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < height and 0 <= xx < width and grid[yy][xx] == grid[y][x] - 1:
                        path.append((yy, xx))
                        y, x = yy, xx
            # Get rid of the starting point from the final path
            path.pop()

            return np.array(path)
        else:
            plt.figure()
            plt.imshow(img)
            plt.scatter(start[0], start[1], label='start')
            plt.scatter(end[0], end[1], label='end')
            plt.legend()
            plt.show()
