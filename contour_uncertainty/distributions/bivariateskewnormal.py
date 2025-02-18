import math

import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import skewnorm
from torch.distributions import MultivariateNormal

from contour_uncertainty.distributions.bivariatedistribution import BivariateDistribution
from contour_uncertainty.distributions.bivariatenormal import BivariateNormal
from contour_uncertainty.distributions.utils import batch_matrix_pow, multivariate_skewnorm, cov2corr, rotate_cov, \
    rotate_alpha
from scipy.stats import (multivariate_normal as mvn,
                         norm)

class BivariateSkewNormal(BivariateDistribution):
    log2 = torch.log(torch.tensor(2))

    @classmethod
    def logpdf(cls, x, loc, cov, alpha):
        shape = x.shape
        shape, dim = shape[:-1], shape[-1]
        x = x.reshape(-1, dim)

        x, loc, cov, alpha = cls._arrange_shapes(x, loc, cov, alpha)

        normal_log_pdf = BivariateNormal.logpdf(x, loc, cov)

        x_affine = cls.affine(x.unsqueeze(-1), loc.unsqueeze(-1), cov, alpha.unsqueeze(-1))

        unit_cdf = BivariateSkewNormal.unit_normal_logcdf(x_affine.squeeze()).squeeze()
        logpdf = cls.log2 + normal_log_pdf + unit_cdf

        return logpdf.reshape(shape)

    @classmethod
    def affine(cls, x, loc, cov, alpha):
        # cor, std = cov2corr(cov)
        # S = torch.diag_embed(std)
        # x_affine = torch.bmm(alpha.transpose(-1, -2), torch.inverse(S)) @ (x - loc)

        x_affine = torch.bmm(alpha.transpose(-1, -2), batch_matrix_pow(cov, -0.5)) @ (x - loc)

        return x_affine

    @classmethod
    def unit_normal_logcdf(cls, x):
        cdf = 0.5 * (1 + torch.erf(x / math.sqrt(2)))
        return torch.log(cdf + 1e-7)

    @classmethod
    def nll(cls, y, mu, cov, alpha):
        term1 = torch.log(torch.det(cov)).squeeze()
        term2 = (((mu - y).transpose(-1, -2) @ torch.inverse(cov)) @ (mu - y)).squeeze()
        
        x_affine = cls.affine(y, mu, cov, alpha)
        
        term3 = cls.unit_normal_logcdf(x_affine.squeeze()).squeeze()
        # nll = term1 + term2 - term3
        nll = 0.5 * term1 + 0.5 * term2 - term3
        return nll, term1, term2, term3

    @classmethod
    def plot(cls, ax, mu, cov, *args, **kwargs):
        """ Plot set of distributions

        Returns:

        """
        raise NotImplementedError

    @classmethod
    def mode(cls, mu, cov, alpha):
        corr, w = cov2corr(cov)

        alpha_star = torch.sqrt(alpha.T @ corr @ alpha)  # in [0, inf]
        m0_start = univariate_mode(0, 1, alpha_star)

        mode = mu + m0_start / alpha_star * w @ corr @ alpha

        return mode.squeeze()

    @classmethod
    def conditional_variance(cls, mu, cov, angle=0, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def conditional(cls, mu, cov, x=None, y=None, angle=0, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def marginal(cls, mu, cov, alpha, axis: int, angle=torch.tensor(0), *args, **kwargs):
        assert axis == 0 or axis == 1

        cov = rotate_cov(cov, -angle)

        alpha = torch.tensor(alpha).clone()
        alpha[1] = -alpha[1]
        alpha = rotate_alpha(alpha, -angle)

        corr, w = cov2corr(cov)
        corr = corr.squeeze()

        # mu_1, var_1, alpha_1 = mu[0], cov[0, 0], alpha[0]
        # mu_2, var_2, alpha_2 = mu[1], cov[1, 1], alpha[1]
        #
        # corr_11 = corr[0, 0]
        # corr_22 = corr[1, 1]
        # corr_12 = corr[0, 1]    # Equal to corr[1, 0]
        # corr_21 = corr[1, 0]    # Equal to corr[1, 0]

        # if axis == 0:
        #     corr_22_1 = corr_22 - corr_21 * corr_12 / corr_11
        #     alpha_1_2 = (alpha_1 + (1 / corr_11) * corr_12 * alpha_2) / torch.sqrt(1 + alpha_2 * corr_22_1 * alpha_2)
        #     return mu_1, var_1, alpha_1_2
        # else:
        #     corr_11_2 = corr_11 - corr_12 * corr_21 / corr_22
        #     alpha_2_1 = (alpha_2 + (1 / corr_22) * corr_21 * alpha_1) / torch.sqrt(1 + alpha_1 * corr_11_2 * alpha_1)
        #     return mu_2, var_2, alpha_2_1

        not_axis = 1 - axis

        mu_1 = mu[axis]
        var_1 = cov[axis, axis]
        alpha_1 = alpha[axis]

        corr_11 = corr[axis, axis]
        corr_22 = corr[not_axis, not_axis]
        corr_12 = corr[0, 1]  # Equal to corr[1, 0]

        alpha_2 = alpha[not_axis]

        corr_22_1 = corr_22 - corr_12 * corr_12 / corr_11
        alpha_1_2 = (alpha_1 + (1 / corr_11) * corr_12 * alpha_2) / torch.sqrt(1 + alpha_2 * corr_22_1 * alpha_2)
        return mu_1, var_1, alpha_1_2

    @classmethod
    def rvs_slow(cls, mu, cov, alpha, size=1):
        dim = mu.shape[-1]
        std_mvn = mvn(mu.cpu().detach().numpy(), cov.cpu().detach().numpy())
        x = np.empty((size, dim))

        # Apply rejection sampling.
        n_samples = 0
        while n_samples < size:
            z = std_mvn.rvs(size=1)
            u = np.random.uniform(0, 2 * std_mvn.pdf(z))
            z = torch.tensor(z)
            if not u > cls.pdf(torch.tensor(z).float(), mu, cov, alpha):
                x[n_samples] = z
                n_samples += 1

        # Rescale based on correlation matrix.
        chol = np.linalg.cholesky(cov.cpu().detach().numpy())
        x = (chol @ x.T).T

        return x

    @classmethod
    def rvs_fast(cls, mu, cov, alpha, size=1):

        # print(mu.shape)
        # print(cov.shape)
        # print(alpha.shape)
        # print(size)
        # exit(0)

        dim = mu.shape[-1]
        aCa = alpha.T @ cov @ alpha
        delta = (1 / torch.sqrt(1 + aCa)) * cov @ alpha


        # cov_star = np.block([[np.ones(1), delta.T],
        #                      [delta[:, None], cov]])

        cov_star = torch.zeros((3,3), device=mu.device)
        cov_star[0, 0] = 1
        cov_star[1:3, 0] = delta
        cov_star[0, 1:3] = delta
        cov_star[1:3, 1:3] = cov

        # x = mvn(torch.zeros(dim + 1), cov_star).rvs(size)
        x = MultivariateNormal(torch.zeros(dim + 1, device=mu.device), cov_star, validate_args=False).sample(size)
        if x.ndim == 1:
            x = x[None]

        x0, x1 = x[:, 0], x[:, 1:]
        inds = x0 <= 0
        x1[inds] = -1 * x1[inds]
        x1 = x1 + mu[None]
        return x1



def delta(alpha):
    return alpha / torch.sqrt(1 + torch.square(alpha))


def skewness(alpha):
    x = torch.tensor(2 / torch.pi)
    term1 = ((4 - torch.pi) / 2)
    num2 = torch.pow(delta(alpha) * torch.sqrt(x), 3)

    denum2 = torch.pow(1 - 2 * torch.square(delta(alpha)) / torch.pi, 3 / 2)

    return term1 * num2 / denum2


def m0(alpha):
    x = torch.tensor(2 / torch.pi)
    mu_z = torch.sqrt(x) * delta(alpha)
    sigma_z = torch.sqrt(1 - torch.square(mu_z))
    m0 = mu_z - skewness(alpha) * sigma_z / 2 - torch.sign(alpha) / 2 * torch.exp(- 2 * torch.pi / torch.abs(alpha))

    return m0


def univariate_mode(mu, sigma, alpha):
    return mu + sigma * m0(alpha)


def check_univariate_mode():
    loc = 3
    scale = 2
    alpha = 3
    x = np.linspace(-5, 5, 100)
    y = skewnorm.pdf(x, alpha, loc, scale)
    numerical_mode = x[y.argmax()]

    mode = univariate_mode(torch.tensor(loc), torch.tensor(scale), torch.tensor(alpha))

    print(numerical_mode, mode)

    plt.figure()
    plt.plot(x, y)
    plt.axvline(x=numerical_mode, label='numerical mode', c='r')
    plt.axvline(x=mode, label='estimated mode', c='b')
    plt.legend()

    plt.show()


def check_bivariate_mode():
    xx = np.linspace(0, 256, 256)
    yy = np.linspace(0, 256, 256)
    X, Y = np.meshgrid(xx, yy)
    pos = torch.tensor(np.dstack((X, Y))).float()

    mu = torch.tensor([100., 150.])
    cov = torch.tensor([[10., -5.],
                        [-5., 10.]])
    alpha = torch.tensor([5., 0.])

    Z = BivariateSkewNormal.pdf(x=pos, loc=mu, cov=cov, alpha=alpha).numpy()

    numerical_mode = np.unravel_index(np.argmax(Z, axis=None), Z.shape)

    estimated_mode = BivariateSkewNormal.mode(mu, cov, alpha)

    print(numerical_mode, estimated_mode)

    f, (ax1) = plt.subplots(1, 1)
    ax1.imshow(Z)
    ax1.scatter(numerical_mode[1], numerical_mode[0], label='numerical mode', c='r')
    ax1.scatter(estimated_mode[0], estimated_mode[1], label='estimated mode', c='b')
    ax1.legend()
    plt.show()


def check_scipy_equivalence():
    xx = np.linspace(-2, 2, 200)
    yy = np.linspace(-2, 2, 200)
    X, Y = np.meshgrid(xx, yy)
    pos = torch.tensor(np.dstack((X, Y))).float()

    mu = torch.tensor([0., 0.])
    cov = torch.tensor([[2., 0.5],
                        [0.5, 2.]])
    alpha = torch.tensor([5., 1.])

    Z1 = multivariate_skewnorm(a=alpha, cov=cov).pdf(pos)
    mode1 = np.unravel_index(np.argmax(Z1, axis=None), Z1.shape)
    Z2 = BivariateSkewNormal.pdf(x=pos, loc=mu, cov=cov, alpha=alpha).numpy()
    mode2 = np.unravel_index(np.argmax(Z2, axis=None), Z2.shape)

    print("Error mean: ", np.mean(np.abs(Z1 - Z2)))
    print(mode1, mode2)

    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(Z1)
    ax2.imshow(Z2)
    ax3.imshow(np.abs(Z1 - Z2))
    plt.show()


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    xx = np.linspace(0, 256, 256)
    yy = np.linspace(0, 256, 256)
    X, Y = np.meshgrid(xx, yy)
    extent = [X.min(), X.max(), Y.max(), Y.min()]
    pos = torch.tensor(np.dstack((X, Y))).float()

    mu = torch.tensor([100., 150.])
    cov = torch.tensor([[40., 0.],
                        [0., 40.]])
    alpha = torch.tensor([5, 5]).float()

    # mu = torch.tensor([123.8661, 79.4819])
    # cov = torch.tensor([[28.5017, 16.7787],
    #                     [16.7787, 31.6524]])
    # alpha = torch.tensor([4.9846, -3.5453])

    Z = BivariateSkewNormal.pdf(x=pos, loc=mu, cov=cov, alpha=alpha).numpy()

    # Marginal
    mu_x, var_x, alpha_x = BivariateSkewNormal.marginal(mu, cov, alpha, axis=0)
    mu_y, var_y, alpha_y = BivariateSkewNormal.marginal(mu, cov, alpha, axis=1)

    samples = BivariateSkewNormal.rvs_fast(mu, cov, alpha, size=1000)
    print(samples.shape)
    print(samples)
    print(mu)

    print("X", mu_x, var_x, alpha_x)
    print("Y", mu_y, var_y, alpha_y)

    x2 = np.linspace(0, 255, 1000)
    px = skewnorm.pdf(x2, alpha_x, mu_x, np.sqrt(var_x))
    py = skewnorm.pdf(x2, alpha_y, mu_y, np.sqrt(var_y))

    # v = np.array([0.73236312, -0.68091428])
    v = np.array([1, 2])
    v = v / np.linalg.norm(v)
    zero = np.array([1, 0]).T
    angle = np.arctan2(np.cross(zero, v), np.dot(zero, v))

    mu_v, var_v, alpha_v = BivariateSkewNormal.marginal(mu, cov, alpha, axis=0, angle=torch.tensor(angle))
    print("V", mu_v, var_v, alpha_v)
    x_v = np.linspace(-10, 10, 1000)
    p_v = skewnorm.pdf(x2, alpha_v, mu_v, np.sqrt(var_v))

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,20))
    ax1.imshow(Z)
    # ax1.imshow(Z, extent=extent)
    ax1.scatter(samples[:, 0], samples[:, 1], label='samples', c='r', s=1)
    ax1.quiver(*mu, v[0], v[1], color='r')

    divider = make_axes_locatable(ax1)
    ax1.set_aspect('equal')
    ax_x = divider.append_axes("bottom", 1.0, pad=0.5, sharex=ax1)
    ax_x.plot(x2, px)
    ax_y = divider.append_axes("right", 1.0, pad=0.5, sharey=ax1)
    ax_y.plot(py, x2)

    # ax1.set_xticks([])
    # ax1.set_yticks([])
    # ax_x.set_xticks([])
    # ax_x.set_yticks([])
    # ax_y.set_xticks([])
    # ax_y.set_yticks([])
    # ax2.set_xticks([])
    # ax2.set_yticks([])
    # # ax_x.set_axis_off()
    # # ax_y.set_axis_off()
    # # ax2.set_axis_off()

    ax2.set_title("Marginal along vector")
    ax2.plot(x_v, p_v, label='marginal')

    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
    #                     hspace=0, wspace=0.01)
    # plt.margins(0, 1)

    plt.show()
