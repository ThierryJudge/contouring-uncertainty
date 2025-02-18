import numpy as np
import scipy
from scipy.stats import norm
from scipy.stats import skewnorm
from skimage.draw import line

from contour_uncertainty.utils.contour import contour_spline, reconstruction
from contour_uncertainty.utils.uncertainty_projection import projected_uncertainty


def skew_umap(mu, cov, alpha, shape=(256, 256), close=True, linear_close = False):
    alpha = np.array(alpha)
    u, v, alpha_proj = projected_uncertainty(mu, cov, alpha.copy(), all=True, linear_close=linear_close)
    cov_width = 2
    resolution = 1000

    projected_mode = np.zeros_like(mu)

    N = 100
    values_linspace = np.linspace(0, 0.95, N)
    umap_contours = np.zeros((2 * N, len(mu), 2))
    umap_weights = np.zeros(2 * N)

    for index in range(len(mu)):
        # Get reference points
        p1 = mu[index] + v[index] * u[index] * cov_width
        p2 = mu[index] - v[index] * u[index] * cov_width

        # Compute projected distribution

        x = np.linspace(-3 * u[index], 3 * u[index], resolution)
        y = skewnorm.pdf(x, alpha_proj[index], 0, u[index])
        y = y / y.max()
        mode_y = y.max()
        mode_x = x[y.argmax()]

        mode_x_proj = y.argmax() / len(y)
        projected_mode[index] = (p1 * mode_x_proj + (1 - mode_x_proj) * p2)

        # Compute contours from projected distribution
        for i, val in enumerate(values_linspace):
            val = mode_y - val
            mode_plus = (np.argmin(np.abs(y[x > mode_x] - val)) + y.argmax()) / len(y)
            mode_minus = (np.argmin(np.abs(y[x < mode_x] - val))) / len(y)

            mode_plus = p1 * mode_plus + (1 - mode_plus) * p2
            mode_minus = p1 * mode_minus + (1 - mode_minus) * p2

            # print(N, N - i - 1, N + i)
            umap_contours[N - i - 1, index] = mode_minus
            umap_contours[N + i, index] = mode_plus

            umap_weights[N - i - 1] = norm.pdf(i, loc=0, scale=N/2)
            umap_weights[N + i] = norm.pdf(i, loc=0, scale=N/2) 

            # umap_weights[N - i - 1] = val
            # umap_weights[N + i] = val

    umap = np.zeros(shape)

    rec = []
    for i in range(len(umap_contours)):
        segmap = reconstruction(umap_contours[i], 256, 256)
        rec.append(segmap)

        # c = contour_spline(umap_contours[i])
        # c = c.round().astype(int).clip(max=255, min=0)
        # umap[c[:, 1], c[:, 0]] = umap_weights[i]
        #
        # if close:
        #     p = umap_contours[i].astype(int)
        #     rr, cc = line(p[-1, 1], p[-1, 0], p[0, 1], p[0, 0])
        #     umap[rr.clip(max=255, min=0), cc.clip(max=255, min=0)] = umap_weights[i]

    rec = np.array(rec)


    rec_mean = np.average(rec, axis=0, weights=umap_weights)
    rec_mean = np.concatenate([rec_mean[None], 1 - rec_mean[None]], axis=0)
    umap = scipy.stats.entropy(rec_mean, axis=0)

    return projected_mode, umap