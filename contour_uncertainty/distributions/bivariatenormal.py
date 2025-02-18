import math

import numpy as np
import torch
from torch.distributions import MultivariateNormal

from contour_uncertainty.distributions.bivariatedistribution import BivariateDistribution
from contour_uncertainty.distributions.utils import get_meshgrid, rotate_cov


class BivariateNormal(BivariateDistribution):
    log2pi = torch.log(torch.tensor(2) * torch.pi)

    @classmethod
    def logpdf(cls, x, loc, cov, *args, **kwargs):
        shape = x.shape
        shape, dim = shape[:-1], shape[-1]
        x = x.reshape(-1, dim)

        x, loc, cov = cls._arrange_shapes(x, loc, cov)

        K = x.shape[-1]

        x = torch.unsqueeze(x, -1)
        loc = torch.unsqueeze(loc, -1)

        term1 = K / 2 * cls.log2pi
        term2 = torch.log(cls.det(cov)) / 2
        term3 = ((x - loc).transpose(-1, -2) @ torch.inverse(cov)) @ (x - loc)

        logpdf = - term1.squeeze() - term2.squeeze() - term3.squeeze() / 2
        return logpdf.reshape(shape)

    @classmethod
    def pdf(cls, x, loc, cov, *args, **kwargs):
        return torch.exp(cls.logpdf(x, loc, cov, *args, **kwargs))

    @classmethod
    def nll(cls, y, mu, cov, *args, **kwargs):
        term1 = torch.log(torch.det(cov))
        term2 = (((mu - y).transpose(-1, -2) @ torch.inverse(cov)) @ (mu - y))
        nll = term1 + term2
        return nll, term1, term2

    @classmethod
    def plot(cls, ax, mu, cov, *args, **kwargs):
        """ Plot set of distributions

        Returns:

        """
        raise NotImplementedError

    @classmethod
    def mode(cls, mu, cov, *args, **kwargs):
        return mu

    @classmethod
    def conditional_variance(cls, mu, cov, angle: torch.Tensor, *args, **kwargs):
        xx = cov[0, 0]
        xy = cov[0, 1]
        yy = cov[1, 1]
        c = torch.cos(angle)
        s = torch.sin(angle)
        sigma = torch.det(cov) / (xx * s ** 2 + yy * c ** 2 - 2 * xy * s * c)
        return sigma

    @classmethod
    def marginal(cls, mu, cov, axis: int, angle=torch.tensor(0), *args, **kwargs):
        """

        Args:
            mu:
            cov:
            axis: x=0 or y=1 axis
            angle: Angle along which to compute the marginal (will rotate cov by -angle)
            *args:
            **kwargs:

        Returns:

        """
        assert axis == 0 or axis == 1
        cov = rotate_cov(cov, -angle)

        return mu[axis], cov[axis, axis]

    @classmethod
    def rvs(cls, mu, cov, size=(1,)):
        return MultivariateNormal(mu.squeeze(), cov, validate_args=False).sample(size)


def check_scipy_equivalence():
    from scipy.stats import multivariate_normal

    pos = torch.tensor(get_meshgrid()).float()

    mu = torch.tensor([100., 100.])
    cov = torch.tensor([[25., 4.],
                        [4., 50.]])

    Z1 = multivariate_normal(mean=mu.numpy(), cov=cov.numpy()).pdf(pos.numpy())
    Z2 = BivariateNormal.pdf(x=pos, loc=mu, cov=cov).numpy()

    print("Error mean: ", np.mean(np.abs(Z1 - Z2)))

    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(Z1)
    ax2.imshow(Z2)
    ax3.imshow(np.abs(Z1 - Z2))
    plt.show()


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from contour_uncertainty.utils.plotting import confidence_ellipse
    from scipy.stats import norm

    pos = torch.tensor(get_meshgrid()).float()
    # mu = torch.tensor([100., 150.])
    # cov = torch.tensor([[50., 10.],
    #                     [10., 75.]])

    mu = torch.tensor([123.8661, 79.4819])
    cov = torch.tensor([[28.5017, 16.7787],
                        [16.7787, 31.6524]])

    Z = BivariateNormal.pdf(x=pos, loc=mu, cov=cov).numpy()

    v = np.array([0.73236312, -0.68091428])

    v = v / np.linalg.norm(v)
    # v[1] = -v[1]
    zero = np.array([1, 0]).T
    angle = np.arctan2(np.cross(zero, v), np.dot(zero, v))

    sigma_cond = np.sqrt(BivariateNormal.conditional_variance(mu, cov, torch.tensor(angle)).numpy())
    mu_marg, sigma_marg = BivariateNormal.marginal(mu, cov, 0, torch.tensor(angle))
    sigma_marg = np.sqrt(sigma_marg.numpy())

    print('Sigma conditional', sigma_cond)
    print('MU marginal', mu_marg)
    print('Sigma marginal', sigma_marg)

    x1 = np.linspace(-2 * sigma_cond, 2 * sigma_cond, 1000)
    y_cond = norm.pdf(x1, 0, sigma_cond)
    y_marg = norm.pdf(x1, 0, sigma_marg)

    mu = mu.numpy()
    p1 = mu + v * sigma_cond * 2
    p2 = mu - v * sigma_cond * 2
    points = np.array([p1 * alpha + (1 - alpha) * p2 for alpha in np.linspace(0, 1, 1000)]).round().astype(int)
    values = Z[points[:, 1], points[:, 0]]


    mu_x, var_x = BivariateNormal.marginal(mu, cov, axis=0)
    mu_y, var_y = BivariateNormal.marginal(mu, cov, axis=1)
    x2 = np.linspace(0, 255, 1000)
    px = norm.pdf(x2, mu_x, np.sqrt(var_x))
    py = norm.pdf(x2, mu_y, np.sqrt(var_y))

    f, (ax1, ax12, ax2) = plt.subplots(1, 3)
    ax1.imshow(Z)
    # ax1.quiver(*mu, v[0], v[1], color='red')
    ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], c='r', marker="o", markersize=2, linestyle='--')
    confidence_ellipse(mu[0], mu[1], cov, ax1, n_std=2)
    divider = make_axes_locatable(ax1)
    ax1.set_aspect('equal')
    ax_x = divider.append_axes("bottom", 1.0, pad=0.5, sharex=ax1)
    ax_x.plot(x2, px)
    ax_y = divider.append_axes("right", 1.0, pad=0.5, sharey=ax1)
    ax_y.plot(py, x2)

    cov2 = rotate_cov(cov, torch.tensor(-angle))
    Z2 = BivariateNormal.pdf(x=pos, loc=torch.tensor(mu), cov=cov2).numpy()
    mu_x2, var_x2 = BivariateNormal.marginal(mu, cov, axis=0)
    x3 = np.linspace(0, 255, 1000)
    px2 = norm.pdf(x3, mu_x2, np.sqrt(var_x2))

    ax12.imshow(Z2)
    # ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], c='r', marker="o", markersize=2, linestyle='--')
    confidence_ellipse(mu[0], mu[1], cov2, ax12, n_std=2)
    divider = make_axes_locatable(ax12)
    ax12.set_aspect('equal')
    ax_x2 = divider.append_axes("bottom", 1.0, pad=0.5, sharex=ax12)
    ax_x2.plot(x2, px)


    ax2.set_title("Univariate Gaussian along vector")
    ax2.plot(x1, y_cond, label='conditional')
    ax2.plot(x1, y_marg, label='marginal')
    ax22 = ax2.twinx()
    ax22.plot(x1, values)
    ax2.legend()

    plt.show()
