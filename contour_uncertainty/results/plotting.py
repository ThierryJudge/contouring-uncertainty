import os
from typing import List, Any

import numpy as np
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from pytorch_lightning import Callback
from vital.data.config import Tags

from contour_uncertainty.data.config import ContourTags
from contour_uncertainty.utils.contour import contour_spline
from contour_uncertainty.utils.plotting import confidence_ellipse
from contour_uncertainty.utils.uncertainty_projection import projected_uncertainty


def softmax(X):
    X_exp = np.exp(X)
    partition = X_exp.sum((-1, -2), keepdims=True)
    return X_exp / partition  # The broadcasting mechanism is applied here


class Plotting(Callback):
    """Evaluator to generate uncertainty visualisations."""

    def __init__(self, nb_upload=10, nb_figures=50, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nb_figures = nb_figures
        self.nb_upload = nb_upload
        self.count = 0

        os.mkdir("figures")

    def on_predict_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: List[Any]) -> None:
        """Called when the predict epoch ends."""
        n_std = 2
        data = {}
        for view in outputs[0]:
            if self.count < self.nb_figures:
                self.count += 1
                img = view[Tags.img]
                contour_gt = view[ContourTags.contour]
                contour_mu = view[ContourTags.contour_mu]
                contour_cov = view[ContourTags.contour_cov]
                contour_samples = view[ContourTags.contour_samples]
                for i in range(img.shape[0]):
                    data[f"{view[Tags.id].replace('/', '-')}_{i}"] = {'img': img[i].squeeze(),
                                                                      'pred': contour_mu[i],
                                                                      'gt': contour_gt[i],
                                                                      'sigma': contour_cov[i]}
                    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
                    f.set_figheight(9)
                    f.set_figwidth(16)
                    ax1.imshow(img[i].squeeze())
                    ax1.set_title("Image")
                    ax1.scatter(contour_mu[i, :, 0], contour_mu[i, :, 1], s=5, c="r", label="Pred")

                    if contour_gt.ndim == 4:
                        for k in range(len(contour_gt)):
                            ax1.scatter(contour_gt[k, i, :, 0], contour_gt[k, i, :, 1], s=5, label=f"GT_{i}")
                    else:
                        ax1.scatter(contour_gt[i, :, 0], contour_gt[i, :, 1], s=5, c="b", label="GT")

                    for k in range(0, contour_mu.shape[1], 1):
                        confidence_ellipse(contour_mu[i, k, 0], contour_mu[i, k, 1], contour_cov[i, k], ax1)

                    # Projections
                    ax2.imshow(img[i].squeeze(), cmap='gray')
                    u, v = projected_uncertainty(contour_mu[i], contour_cov[i])

                    for index in range(0, contour_mu.shape[1], 1):
                        if index in [0, contour_mu.shape[1] // 2, contour_mu.shape[1] - 1]:
                            confidence_ellipse(contour_mu[i, index, 0],
                                               contour_mu[i, index, 1],
                                               contour_cov[i, index],
                                               ax2,
                                               n_std=n_std)
                        else:
                            confidence_ellipse(contour_mu[i, index, 0],
                                               contour_mu[i, index, 1],
                                               contour_cov[i, index],
                                               ax2,
                                               n_std=n_std)

                            p1 = contour_mu[i][index] + v[index] * u[index] * n_std
                            p2 = contour_mu[i][index] - v[index] * u[index] * n_std

                            ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], c='r', marker="o", markersize=2)

                    # Probability MAP
                    # x, y = np.mgrid[0:img.shape[2]:1, 0:img.shape[3]:1]
                    # pos = np.dstack((x, y))
                    # map = np.zeros((img.shape[2], img.shape[3]))
                    # for k in range(len(contour_mu[i])):
                    #     rv = multivariate_normal(contour_mu[i, k], contour_cov[i, k])
                    #     map += rv.pdf(pos)
                    #
                    # map = np.transpose(map)
                    # ax3.imshow(map)

                    # Samples
                    ax3.imshow(img[i].squeeze())
                    for j in range(min(5, contour_samples.shape[1])):
                        sample_contour = contour_spline(contour_samples[i, j])

                        # distance_map = np.min(cdist(sample_contour, mean_contour), axis=1)
                        # distance_map = (distance_map - np.min(distance_map)) / (np.max(distance_map) - np.min(distance_map))

                        # s = sample_contour.round().astype(int)
                        # distance_map = prob_map[s[:, 0], s[:, 1]]
                        # distance_map = (distance_map - np.min(distance_map)) / (np.max(distance_map) - np.min(distance_map))

                        # distance_map = np.sqrt((pred_contour[:, 0] - sample_contour[:, 0]) ** 2 + (pred_contour[:, 1] - sample_contour[:, 1]) ** 2)

                        # colorline(sample_contour[:, 0], sample_contour[:, 1], z=distance_map, cmap="plasma", ax=ax[2])
                        ax3.plot(sample_contour[:, 0], sample_contour[:, 1])
                        ax3.scatter(contour_samples[i, j, :, 0], contour_samples[i, j, :, 1], s=10)

                    ax3.scatter(contour_mu[i, :, 0], contour_mu[i, :, 1], s=10, c='r', label='Initial shape')
                    ax3.scatter(contour_gt[i, :, 0], contour_gt[i, :, 1], s=10, c='b', label='Gt')
                    # for k in range(0, mu_p.shape[0], 1):
                    #     confidence_ellipse(mu_p[k, 0],
                    #                        mu_p[k, 1],
                    #                        cov_p[k],
                    #                        ax)
                    ax3.legend()

                    plt.savefig(f"figures/{view[Tags.id].replace('/', '-')}_{i}.png", dpi=100)
                    plt.close()

        np.save('data', data)
