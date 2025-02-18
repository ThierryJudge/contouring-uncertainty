import numpy as np
import torch


class BivariateDistribution:
    def __init__(self):
        pass

    @classmethod
    def _arrange_shapes(cls, x: torch.Tensor, loc: torch.Tensor, cov: torch.Tensor, alpha: torch.Tensor = None):
        """ Verify and arrange shapes.

        Args:
            x: Input (2,) (N, 2)
            loc: Location (N, 2) or (2,)
            cov: Covariance matrix (N, 2, 2) or (2,2)
            alpha: alpha (skew) parameter (N, 2) or (2,)

        Returns:
            x: reshaped (N, 2)
            loc: Location (N, 2)
            cov: Covariance matrix (N, 2, 2)
            alpha: alpha (skew) parameter (N, 2)
        """

        assert x.shape[-1] == 2, f'Last dim of x should be 2. Shape is now {x.shape}'
        assert x.ndim < 3, f'x ndim must be 1 or 2. Shape is now {x.shape}'
        x = x if x.ndim == 2 else x.unsqueeze(0)

        assert loc.shape[-1] == 2, f'Last dim of loc should be 2. Shape is now {loc.shape}'
        assert loc.ndim < 3, f'loc ndim must be 1 or 2. Shape is now {loc.shape}'
        loc = loc if loc.ndim == 2 else loc.unsqueeze(0)

        assert cov.shape[-1] == 2 and cov.shape[-2] == 2, f'Last 2 dims cov should be 2. Shape is now {cov.shape}'
        assert cov.ndim == 2 or cov.ndim == 3, f'cov ndim must be 2 or 3. Shape is now {loc.shape}'
        cov = cov if cov.ndim == 3 else cov.unsqueeze(0)

        if alpha is not None:
            assert alpha.shape[-1] == 2, f'Last dim of alpha should be 2. Shape is now {alpha.shape}'
            assert alpha.ndim < 3, f'alpha ndim must be 1 or 2. Shape is now {alpha.shape}'
            alpha = alpha if alpha.ndim == 2 else alpha.unsqueeze(0)

            return x, loc, cov, alpha

        else:
            return x, loc, cov



    @classmethod
    def logpdf(cls, x, loc, cov, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def pdf(cls, x, loc, cov, *args, **kwargs):
        return torch.exp(cls.logpdf(x, loc, cov, *args, **kwargs))


    @classmethod
    def nll(cls, y, mu, cov, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def plot(cls, ax, mu, cov, *args, **kwargs):
        """ Plot set of distributions

        Returns:

        """
        raise NotImplementedError

    @classmethod
    def mode(cls, mu, cov, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def conditional_variance(cls, mu, cov, angle=0, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def conditional(cls, mu, cov, x=None, y=None, angle=0, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def marginal(cls, mu, cov, axis: int, angle=0, *args, **kwargs):
        raise NotImplementedError


    @classmethod
    def det(cls, matrix):
        return matrix[:, 0, 0] * matrix[:, 1, 1] - matrix[:, 0, 1] * matrix[:, 1, 0]






