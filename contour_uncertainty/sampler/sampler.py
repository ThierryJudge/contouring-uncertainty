from typing import Sequence

import numpy as np
import math

class Sampler:

    def __call__(self, mu: np.array, cov: np.array, n: int) -> np.ndarray:
        """Sample contours from predicted point distributions.

        Args:
            mu: Point distribution means (K, 2)
            cov: Point distribution covariance matrices (K, 2, 2)
            n: Number of contours to sample

        Returns:
            Sampled contours (N, K, 2)
        """
        raise NotImplementedError

    @classmethod
    def sample_points(cls, mu: np.array, cov: np.array, n: int = 1, sample_indices: Sequence = None):
        """ Sample each point distribution independently and combine in array.

        Args:
            mu: Point distribution means (K, 2)
            cov: Point distribution covariance matrices (K, 2, 2)
            n: Number of contours to sample
            sample_indices: Indices of points to sample

        Returns:
            Sampled contours (N, len(sampled_points), 2) or (N, K, 2)
        """
        points = []
        sample_indices = sample_indices or range(mu.shape[0])
        for j in sample_indices:
            x, y = np.random.multivariate_normal(mu[j].squeeze(), cov[j], n, check_valid='ignore').T
            points.append([x, y])

        points = np.array(points).swapaxes(0, -1).swapaxes(1, 2).squeeze()
        return points

    @staticmethod
    def get_points_order(nb_points: int = 21, nb_initial_points: int = 3, levels: int = None):
        """Get point order by repeatedly splitting points in half.

        Args:
            nb_points: Number of points in contour
            nb_initial_points: number of initial points to sample freely.
            levels: Number of levels to split. If None, split until all points are included.

        Returns:
            List of list of point indices.
        """
        # initial_points = np.round(np.linspace(0, 21 - 1, nb_initial_points)).astype(int).tolist()
        initial_points = np.round(np.linspace(0, 21 - 1, nb_initial_points)).astype(int).tolist()
        levels = levels or int(math.log(nb_points, 2))
        all_points, point_order = [], []
        all_points.extend(initial_points)
        for i in range(levels):
            level_points = []
            for j in range(len(all_points) - 1):
                if all_points[j] + 1 != all_points[j + 1]:
                    point = (all_points[j] + all_points[j + 1]) / 2
                    point = math.ceil(point) if point > nb_points / 2 else math.floor(point)  # Round towards the base.
                    level_points.append(int(point))
            if len(level_points) == 0:
                break
            all_points.extend(level_points)
            all_points.sort()
            point_order.append(level_points)

        return initial_points, point_order
