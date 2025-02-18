import math

import numpy as np
import torch
from scipy.stats import norm
from torch.distributions import Normal

from contour_uncertainty.utils.plotting import confidence_ellipse


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

    std = torch.sqrt(torch.diagonal(cov, dim1=1, dim2=2))
    # print(std.shape)
    corr = cov / torch.bmm(std.unsqueeze(2), std.unsqueeze(1))

    # std = torch.sqrt(torch.diag(cov))
    # print(std.shape)
    # corr = cov / torch.outer(std, std)
    return corr, std


class MultivariateSkewNorm:
    def __init__(self):
        pass

    @classmethod
    def _arrange_shapes(cls, x: torch.Tensor, loc: torch.Tensor, cov: torch.Tensor, alpha: torch.Tensor):
        """ Verify and arrange shapes

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

        assert alpha.shape[-1] == 2, f'Last dim of alpha should be 2. Shape is now {alpha.shape}'
        assert alpha.ndim < 3, f'alpha ndim must be 1 or 2. Shape is now {alpha.shape}'
        alpha = alpha if alpha.ndim == 2 else alpha.unsqueeze(0)

        return x, loc, cov, alpha

    @classmethod
    def multivariate_normal_logpdf(cls, x, loc, cov):
        K = x.shape[-1]

        # print(x.shape)
        # print(loc.shape)
        # print(cov.shape)

        x = torch.unsqueeze(x, -1)
        loc = torch.unsqueeze(loc, -1)

        eps = 1e-7

        term1 = K / 2 * torch.log(torch.tensor(2 * torch.pi))
        term2 = torch.log(torch.det(cov)) / 2
        # print('det', (cov[:, 0, 0] * cov[:, 1, 1] - cov[:, 1, 0] * cov[:, 0, 1]).mean())
        # term2 = torch.log((cov[:, 0, 0] * cov[:, 1, 1] - cov[:, 1, 0] * cov[:, 0, 1]) + eps)
        term3 = ((x - loc).transpose(-1, -2) @ torch.inverse(cov)) @ (x - loc)

        # print('term1', term1.mean())
        # print('term2', term2.mean())
        # print('term3', term3.mean())
        #
        # print((x - loc).transpose(-1, -2).shape)
        # print(((x - loc).transpose(-1, -2) @ torch.inverse(cov)).shape)

        return - term1.squeeze() - term2.squeeze() - term3.squeeze() / 2
        # return term2.squeeze() + term3.squeeze() / 2

    @classmethod
    def unit_normal_logcdf(cls, x):
        cdf = 0.5 * (1 + torch.erf(x / math.sqrt(2)))
        return torch.log(cdf + 1e-7)

    @classmethod
    def affine(cls, x, loc, cov, alpha):
        # cor, std = cov2corr(cov)
        # S = torch.diag_embed(std)
        # x_affine = torch.bmm(alpha.transpose(-1, -2), torch.inverse(S)) @ (x - loc)

        x_affine = torch.bmm(alpha.transpose(-1, -2), batch_matrix_pow(cov, -0.5)) @ (x - loc)

        return x_affine

    @classmethod
    def logpdf(cls, x, loc, cov, alpha):
        # print('x shape', x.shape)
        # print('loc shape', loc.shape)
        # print('cov shape', cov.shape)
        # print('alpha shape', alpha.shape)
        shape = x.shape
        shape, dim = shape[:-1], shape[-1]
        x = x.reshape(-1, dim)

        # print('x shape', x.shape)

        x, loc, cov, alpha = cls._arrange_shapes(x, loc, cov, alpha)

        # print('loc shape', loc.shape)
        # print('cov shape', cov.shape)
        # print('alpha shape', alpha.shape)

        pdf = cls.multivariate_normal_logpdf(x, loc, cov)

        # print('pdf shape', pdf.shape)

        x_affine = cls.affine(x.unsqueeze(-1), loc.unsqueeze(-1), cov, alpha.unsqueeze(-1))

        cdf = MultivariateSkewNorm.unit_normal_logcdf(x_affine.squeeze()).squeeze()
        # x_temp = alpha.unsqueeze(-1).transpose(-1, -2) @ (x - loc).unsqueeze(-1)
        # x_temp = torch.bmm(alpha.unsqueeze(-1).transpose(-1, -2), torch.inverse(S)) @ (x - loc).unsqueeze(-1)
        # cdf = MultivariateSkewNorm.unit_normal_logcdf(x_temp.squeeze())

        # print('cdf shape', cdf.shape)

        # print('pdf', pdf.mean())
        # print('cdf', cdf.mean())
        # print(pdf)

        logpdf = torch.log(torch.tensor(2)) + pdf + cdf

        # print('log pdf shape', logpdf.shape)

        # print(logpdf.shape)

        return logpdf.reshape(shape)

    @classmethod
    def pdf(cls, x, loc, cov, alpha):
        return np.exp(cls.logpdf(x, loc, cov, alpha))

    @classmethod
    def nll(cls, y, mu, cov, alpha):
        loss_term1 = torch.log(torch.det(cov)).squeeze()
        loss_term2 = (((mu - y).transpose(-1, -2) @ torch.inverse(cov)) @ (mu - y)).squeeze()

        # x_affine = alpha.unsqueeze(-1).transpose(-1, -2) @ (mu - y)
        # print(alpha.unsqueeze(-1).transpose(-1, -2).shape)
        # print(batch_matrix_pow(cov, -0.5).shape)
        # print((mu - y).unsqueeze(-1).shape)
        # x_affine = cls.affine(mu, y, cov, alpha)
        # print(x_affine.shape)
        x_affine = torch.bmm(alpha.transpose(-1, -2), batch_matrix_pow(cov, -0.5)) @ (mu - y)

        loss_term3 = cls.unit_normal_logcdf(x_affine.squeeze()).squeeze()

        loss = loss_term1 + loss_term2 - loss_term3

        return loss, loss_term1, loss_term2, loss_term3


def plot_skewed_normals(ax, mu, cov, alpha, skip=1, xx=None, yy=None, color='red', flip_y = True):
    assert mu.ndim == 2 and mu.shape[-1] == 2, f'Wrong shape for mu: {mu.shape}'
    assert cov.ndim == 3 and cov.shape[-1] == 2 and cov.shape[-2] == 2, f'Wrong shape for cov: {cov.shape}'
    assert alpha.ndim == 2 and alpha.shape[-1] == 2, f'Wrong shape for alpha: {alpha.shape}'

    xx = xx if xx is not None else np.linspace(0, 256, 256)
    yy = yy if yy is not None else np.linspace(0, 256, 256)
    X, Y = np.meshgrid(xx, yy)
    pos = np.dstack((X, Y))

    if flip_y:
        alpha = np.copy(alpha)
        alpha[:, 1] = -alpha[:, 1]
    # alpha[:, 0] = -alpha[:, 0]

    for i in range(0, len(mu), skip):
        z = MultivariateSkewNorm.pdf(torch.tensor(pos).float(), torch.tensor(mu[i]), torch.tensor(cov[i]),
                                     torch.tensor(alpha[i]))
        z = z / torch.max(z)
        max = np.unravel_index(np.argmax(z.numpy(), axis=None), z.numpy().shape)

        # z2 = np.exp(MultivariateSkewNorm.multivariate_normal_logpdf(torch.tensor(pos).float(), torch.tensor(mu[i]), torch.tensor(cov[i])))
        # z2 = z2 / torch.max(z2)
        # ax.contour(X, Y, z2, [0.15], colors='magenta', linewidths=0.5)

        # TL = (1 - torch.erf(torch.tensor([1, 2, 3]) / sqrt(2)))
        ax.contour(X, Y, z, [0.15], colors=color, linewidths=1)
        # ax.scatter(max[1], max[0], label='mode', c='c', s=5)

        # confidence_ellipse(mu[i, 0], mu[i, 1], cov[i], ax, n_std=2, linestyle=':')


def get_mode(mu, cov, alpha):
    assert mu.ndim == 2 and mu.shape[-1] == 2, f'Wrong shape for mu: {mu.shape}'
    assert cov.ndim == 3 and cov.shape[-1] == 2 and cov.shape[-2] == 2, f'Wrong shape for cov: {cov.shape}'
    assert alpha.ndim == 2 and alpha.shape[-1] == 2, f'Wrong shape for alpha: {alpha.shape}'

    xx = np.linspace(0, 256, 256)
    yy = np.linspace(0, 256, 256)
    X, Y = np.meshgrid(xx, yy)
    pos = torch.tensor(np.dstack((X, Y)))

    alpha = torch.clone(alpha)
    alpha[:, 1] = -alpha[:, 1]

    mode = []
    for i in range(len(mu)):
        z = MultivariateSkewNorm.pdf(pos.float(), mu[i], cov[i], alpha[i])
        z = z / torch.max(z)
        max = np.unravel_index(np.argmax(z.numpy(), axis=None), z.numpy().shape)
        mode.append(np.flip(max))

    return np.array(mode)


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


if __name__ == '__main__':
    from scipy.linalg import fractional_matrix_power

    # A = torch.tensor([[70, 50], [50, 120]]).float()
    # A = A[None].repeat_interleave(10, dim=0).squeeze()

    A = []
    for i in range(10):
        N = 2
        a = np.random.rand(N, N)
        m = np.tril(a) + np.tril(a, -1).T
        A.append(m)

    A = torch.tensor(A).float()

    B = batch_matrix_pow(A, -0.5)

    cor, std = cov2corr(A)
    S = torch.diag_embed(std)
    S = torch.inverse(S)

    for i in range(len(A)):
        print(i)
        print(A[i])
        print(S[i])
        print(fractional_matrix_power(A[i], -0.5))
        print(B[i])
