import numpy as np
from scipy.stats import norm
from scipy.stats import skewnorm
from skimage.draw import line

from contour_uncertainty.utils.contour import contour_spline
from contour_uncertainty.utils.uncertainty_projection import projected_uncertainty


def uncertainty_map(mu_p, cov_p, shape=(256, 256), close=True):
    u, v = projected_uncertainty(mu_p, cov_p, all=True)
    u = np.array(u)
    v = np.array(v)

    map = np.zeros(shape)

    thickness = 20
    std_range = 2
    linspace = np.linspace(-std_range, std_range, 100)
    scale = 1

    for k, i in enumerate(linspace):
        mu = mu_p + v * u[..., None] * i
        c = contour_spline(mu, close=False)
        mu = mu.astype(int)
        rr, cc = line(mu[-1, 1], mu[-1, 0], mu[0, 1], mu[0, 0])

        c = c.round().astype(int).clip(max=255, min=0)
        map[c[:, 1], c[:, 0]] = norm.pdf(i, loc=0, scale=scale)
        if close:
            map[rr.clip(max=255, min=0), cc.clip(max=255, min=0)] = norm.pdf(i, loc=0, scale=scale)

    return map
