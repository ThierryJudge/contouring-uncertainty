import numpy as np
import torch
from scipy.stats import (multivariate_normal as mvn, norm)


class multivariate_skewnorm:
    """Class used for validation.

    Source: https://gregorygundersen.com/blog/2020/12/29/multivariate-skew-normal/
    """

    def __init__(self, a, pos=None, cov=None):
        self.dim = len(a)
        self.a = np.asarray(a)
        self.mean = np.zeros(self.dim) if pos is None else np.asarray(pos)
        self.cov = np.eye(self.dim) if cov is None else np.asarray(cov)

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def logpdf(self, x):
        # print(x.shape)
        x = mvn._process_quantiles(x, self.dim)
        # print(x.shape)
        pdf = mvn(self.mean, self.cov).logpdf(x)
        # print(np.sum(pdf))
        # print(pdf.shape)
        # print('alpha')
        # print(np.dot(x, self.a).flatten())
        cdf = norm(0, 1).logcdf(np.dot(x, self.a))
        # print(norm(0, 1).logcdf(-20))
        # exit(0)
        # print(np.sum(cdf))
        # print(cdf.shape)
        return np.log(2) + pdf + cdf


def cov2corr(cov):
    """
    convert batch of covariance matrix to batch of correlation matrix

    Copied from https://www.statsmodels.org/dev/generated/statsmodels.stats.moment_helpers.cov2corr.html

    Parameters
    ----------
    cov : array_like, 2d
        covariance matrix, see Notes

    Returns
    -------
    corr : ndarray (subclass)
        correlation matrix
    return_std : bool
        If this is true then the standard deviation is also returned.
        By default only the correlation matrix is returned.

    Notes
    -----
    This function does not convert subclasses of ndarrays. This requires that
    division is defined elementwise. np.ma.array and np.matrix are allowed.
    """

    if cov.ndim == 2:
        cov = cov[None]

    std = torch.sqrt(torch.diagonal(cov, dim1=1, dim2=2))
    # print(std.shape)
    corr = cov / torch.bmm(std.unsqueeze(2), std.unsqueeze(1))

    # std = torch.sqrt(torch.diag(cov))
    # print(std.shape)
    # corr = cov / torch.outer(std, std)
    return corr, std


def matrix_pow(matrix: torch.Tensor, p: float) -> torch.Tensor:
    r"""
    Power of a matrix using Eigen Decomposition.
    Args:
        matrix: matrix
        p: power
    Returns:
        Power of a matrix
    """
    print('matrix power')
    vals, vecs = torch.eig(matrix, eigenvectors=True)
    print(vals.shape, vals)
    print(vecs.shape, vecs)
    vals = torch.view_as_complex(vals.contiguous())
    print('vals', vals.shape, vals)
    vals_pow = vals.pow(p)
    print('val pow', vals_pow)
    vals_pow = torch.view_as_real(vals_pow)[:, 0]
    print(vals_pow)
    print(torch.diag(vals_pow))
    matrix_pow = torch.matmul(vecs, torch.matmul(torch.diag(vals_pow), torch.inverse(vecs)))
    return matrix_pow


def batch_matrix_pow(matrix: torch.Tensor, p: float) -> torch.Tensor:
    r"""
    Power of a matrix using Eigen Decomposition.

    https://discuss.pytorch.org/t/pytorch-square-root-of-a-positive-semi-definite-matrix/100138/4

    Args:
        matrix: matrix
        p: power
    Returns:
        Power of a matrix
    """
    # print('Matrix power 2 ')
    vals, vecs = torch.linalg.eig(matrix)
    # print(vals.shape, vals)
    # print(vecs.shape, vecs)

    vals = vals.contiguous()
    # print('vals', vals.shape, vals)
    vals_pow = vals.pow(p)
    # print('val pow', vals_pow[0])
    vals_pow = vals_pow.real
    # print('val pow view', vals_pow[0])
    vecs = vecs.real
    # print(vecs.shape)
    # print(torch.diag_embed(vals_pow).shape)
    # print(vals_pow[0])
    # print(torch.diag_embed(vals_pow)[0])
    matrix_pow = torch.matmul(vecs, torch.matmul(torch.diag_embed(vals_pow), torch.inverse(vecs)))
    return matrix_pow


def rotate_cov(cov, theta):
    """ Rotate

    Args:
        cov: [2,2]
        theta: float, angle in radians

    Returns:

    """
    c, s = torch.cos(theta), torch.sin(theta)
    R = torch.tensor([[c, -s], [s, c]]).float()
    return R @ cov @ R.T


def rotate_alpha(alpha, theta):
    c, s = torch.cos(theta), torch.sin(theta)
    R = torch.tensor([[c, -s], [s, c]]).float()
    return R @ alpha


def get_meshgrid(shape=(256, 256)):
    xx = np.linspace(0, shape[1], shape[1])
    yy = np.linspace(0, shape[0], shape[0])
    X, Y = np.meshgrid(xx, yy)
    pos = torch.tensor(np.dstack((X, Y)))
    return pos
