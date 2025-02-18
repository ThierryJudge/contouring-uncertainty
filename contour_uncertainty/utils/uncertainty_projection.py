import numpy as np
import torch
from scipy import interpolate
import math

from contour_uncertainty.distributions.bivariatenormal import BivariateNormal
from contour_uncertainty.distributions.bivariateskewnormal import BivariateSkewNormal
from contour_uncertainty.distributions.utils import rotate_alpha


def projected_uncertainty_value(mu, cov, use_eigenvalue: bool = True):
    """ Find contour uncertainty by projection uncertainty perpendicular to contour"""
    uncertainties, _ = projected_uncertainty(mu, cov, use_eigenvalue=use_eigenvalue)
    return np.sum(uncertainties)


def projected_uncertainty(mu, cov, alpha=None, use_eigenvalue: bool = True, all=False, linear_close = False):
    """ Find contour uncertainty by projection uncertainty perpendicular to contour

    Args:
        mu: contour point means (K, 2)
        cov: contour points covariance matrices (K, 2, 2)
        use_eigenvalue: if True, basal and apex uncertainties are sum of eigenvalue decomposition of cov
                        else, use sqrt(det(cov)).
        all: return projection for all values, if false, cov det or eigenvalue are returned for basal and apex
        linear_close: if true, assumes spline is closed with straight line between point 0 and point -1. 

    Returns
        projected uncertainty for each point (except basal and apex) and projection vectors.
    """
    tck, u = interpolate.splprep([mu[:, 0], mu[:, 1]], k=3, s=0)
    unew = np.linspace(0, 1.01, 1000)
    # sample_out = interpolate.splev(unew, tck)
    # sample_out = np.concatenate([s[..., None] for s in sample_out], axis=1)

    der_out = interpolate.splev(unew, tck, der=1)
    der_out = np.concatenate([s[..., None] for s in der_out], axis=1)

    uncertainties = []
    projections = []
    if alpha is not None:
        alpha_proj = []

    for index in range(0, mu.shape[0], 1):
        i = np.argmin(np.abs(u[index] - unew))
        v = der_out[i] / np.linalg.norm(der_out[i])
        v = np.flip(v)
        v[1] = -v[1]

        if index in [0, mu.shape[0] // 2, mu.shape[0] - 1] and not all:
            w, _ = np.linalg.eig(cov[index])
            uncertainties.append(np.sum(np.sqrt(w)))
        else:
            zero = np.array([1, 0]).T
            angle = np.arctan2(np.cross(zero, v), np.dot(zero, v))

            if linear_close and index == 0:
                v1 = mu[1] - mu[0]
                v1 = v1 / np.linalg.norm(v1)
                v2 = mu[-1] - mu[0]
                v2 = v2 / np.linalg.norm(v2)
                v = (v1 + v2) / 2
                v = v / np.linalg.norm(v)
            if linear_close and index == len(mu)-1:
                v1 = mu[-1] - mu[-2]
                v1 = v1 / np.linalg.norm(v1)
                v2 = mu[-1] - mu[0]
                v2 = v2 / np.linalg.norm(v2)
                v = (v1 + v2) / 2
                v = v / np.linalg.norm(v)

            # # sigma = np.linalg.det(cov[index]) / (cov[index, 0, 0] * math.sin(angle) ** 2 + cov[index, 1, 1] * math.cos(angle) ** 2 - (cov[index, 0, 1] + cov[index, 1, 0]) * math.sin(angle) * math.cos(angle))
            # a = cov[index, 0, 0]
            # b = cov[index, 0, 1]
            # c = cov[index, 1, 0]
            # d = cov[index, 1, 1]
            # sigma = np.linalg.det(cov[index]) / (a * math.sin(angle) ** 2 +
            #                                      d * math.cos(angle) ** 2 -
            #                                      (b + c) * math.sin(angle) * math.cos(angle))
            #
            # # projected_unc = np.dot(v.T, cov[index]).dot(v)
            # uncertainties.append(np.sqrt(sigma))
            # if alpha is not None:
            #     alpha_index = alpha[index]
            #     alpha_index[1] = -alpha_index[1]
            #     alpha_proj.append(np.dot(alpha_index, v))

            # print(index, v)

            if alpha is not None:
                mu_v, var_v, alpha_v = BivariateSkewNormal.marginal(mu[index], cov[index], alpha[index], axis=0,
                                                                    angle=torch.tensor(angle))
                # a = cov[index, 0, 0]
                # b = cov[index, 0, 1]
                # c = cov[index, 1, 0]
                # d = cov[index, 1, 1]
                # var_v = np.linalg.det(cov[index]) / (a * math.sin(angle) ** 2 +
                #                                      d * math.cos(angle) ** 2 -
                #                                      (b + c) * math.sin(angle) * math.cos(angle))
                #
                # # alpha_v = rotate_alpha(alpha[index], torch.tensor(angle))
                # alpha_v = np.dot(alpha[index], v)
                # print(index, np.dot(alpha[index], alpha[index]), alpha_v)


                uncertainties.append(np.sqrt(var_v))
                alpha_proj.append(alpha_v)
            else:
                # sigma = np.linalg.det(cov[index]) / (cov[index, 0, 0] * math.sin(angle) ** 2 + cov[index, 1, 1] * math.cos(angle) ** 2 - (cov[index, 0, 1] + cov[index, 1, 0]) * math.sin(angle) * math.cos(angle))
                # a = cov[index, 0, 0]
                # b = cov[index, 0, 1]
                # c = cov[index, 1, 0]
                # d = cov[index, 1, 1]
                # sigma = np.linalg.det(cov[index]) / (a * math.sin(angle) ** 2 +
                #                                      d * math.cos(angle) ** 2 -
                #                                      (b + c) * math.sin(angle) * math.cos(angle))
                _, sigma = BivariateNormal.marginal(mu[index], cov[index], axis=0, angle=torch.tensor(angle))
                # _, sigma = BivariateNormal.conditional_variance(mu[index], cov[index], axis=0, angle=torch.tensor(angle))
                uncertainties.append(np.sqrt(sigma))
        projections.append(v)

    if alpha is not None:
        return np.array(uncertainties), np.array(projections), np.array(alpha_proj)
    else:
        return np.array(uncertainties), np.array(projections)


if __name__ == '__main__':
    import numpy as np
    from scipy import interpolate
    from matplotlib import pyplot as plt

    from contour_uncertainty.utils.plotting import confidence_ellipse

    data = np.load('deeplabv3_pred2.npy', allow_pickle=True).item()
    # patient = data['patient0051-2CH_1']
    # patient = data['patient0208-2CH_0']
    # patient = data['patient0052-2CH_1']
    patient = data['patient0194-2CH_0']
    img = patient['img'].squeeze()
    mu = np.array(patient['pred'])
    sigma = np.array(patient['sigma'])
    gt = np.array(patient['gt'])

    n_std = 2

    print(mu.shape)

    tck, u = interpolate.splprep([mu[:, 0], mu[:, 1]], k=3, s=0)
    # unew = np.arange(0, 1.01, 0.01)
    unew = np.linspace(0, 1.01, 1000)
    sample_out = interpolate.splev(unew, tck)
    sample_out = np.concatenate([s[..., None] for s in sample_out], axis=1)

    der_out = interpolate.splev(unew, tck, der=1)
    der_out = np.concatenate([s[..., None] for s in der_out], axis=1)

    f, ax = plt.subplots(1, 2)
    ax = ax.ravel()
    ax[0].imshow(img.squeeze(), cmap="gray")
    ax[1].imshow(img.squeeze(), cmap="gray")
    ax[0].scatter(mu[:, 0], mu[:, 1], s=10)
    ax[1].scatter(mu[:, 0], mu[:, 1], s=10)
    ax[0].plot(sample_out[:, 0], sample_out[:, 1])
    ax[1].plot(sample_out[:, 0], sample_out[:, 1])
    uncertainty = 0

    # Projections
    u, v = projected_uncertainty(mu, sigma)

    for index in range(0, mu.shape[0], 1):
        confidence_ellipse(mu[index, 0], mu[index, 1], sigma[index], ax[0], n_std=n_std)
        print(index, mu[index], sigma[index], u[index], v[index])
        if index in [0, mu.shape[0] // 2, mu.shape[0] - 1]:
            confidence_ellipse(mu[index, 0],
                               mu[index, 1],
                               sigma[index],
                               ax[1],
                               n_std=n_std)
        else:
            confidence_ellipse(mu[index, 0],
                               mu[index, 1],
                               sigma[index],
                               ax[1],
                               n_std=n_std)
            # v[index][1] = -v[index][1]  # Flip y value
            p1 = mu[index] + v[index] * u[index] * n_std
            p2 = mu[index] - v[index] * u[index] * n_std

            ax[1].plot([p1[0], p2[0]], [p1[1], p2[1]], c='r', marker="o", markersize=2)

    plt.show()
