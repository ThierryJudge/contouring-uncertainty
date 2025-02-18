from typing import Sequence, Optional

import numpy as np

from contour_uncertainty.sampler.sampler import Sampler


class NaiveSampler(Sampler):

    def __init__(self, sample_indices: Optional[Sequence] = None):
        self.sample_indices = sample_indices

    def __call__(self, mu: np.array, cov: np.array, n: int) -> np.ndarray:
        """Sample contours from predicted point distributions.

        Args:
            mu: Point distribution means (K, 2)
            cov: Point distribution covariance matrices (K, 2, 2)
            n: Number of contours to sample

        Returns:
            Sampled contours (N, K, 2)
        """
        sample_indices = self.sample_indices if self.sample_indices is not None else list(range(len(mu)))
        return self.sample_points(mu, cov, n, sample_indices=sample_indices)
