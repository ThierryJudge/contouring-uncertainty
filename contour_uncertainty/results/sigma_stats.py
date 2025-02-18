from pprint import pprint
from typing import List, Any

import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as LA
from pytorch_lightning import Callback
from scipy.stats import pearsonr

from contour_uncertainty.data.camus.dataset import ContourTags
from contour_uncertainty.utils.plotting import confidence_ellipse


class SigmaStats(Callback):

    def on_predict_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: List[Any]) -> None:
        # mpl.rcParams['text.usetex'] = True
        # mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'

        sigmas, distances = [], []

        for view in outputs[0]:
            contour_gt = view[ContourTags.contour].cpu().numpy()
            contour_pred = view[ContourTags.contour_pred].cpu().numpy()
            contour_sigma = view[ContourTags.contour_sigma].cpu().numpy()

            # dist_y = np.abs(mu_pred[:, :, 0] - y[:, :, 0])
            # dist_x = np.abs(mu_pred[:, :, 1] - y[:, :, 1])
            dist = np.sqrt(
                (contour_pred[:, :, 1] - contour_gt[:, :, 1]) ** 2 + (contour_pred[:, :, 0] - contour_gt[:, :, 0]) ** 2)
            distances.extend(dist)
            sigmas.extend(contour_sigma)

        distances = np.array(distances)
        print(distances.shape)
        distances = distances.mean(axis=0)

        sigmas = np.array(sigmas)
        sigmas = sigmas.mean(axis=0)
        print(sigmas.shape)
        det = LA.det(sigmas) ** 0.25

        f, ax = plt.subplots(1, 1)
        f.set_figheight(9)
        f.set_figwidth(16)
        # Use last contour for plotting purposes.
        ax.scatter(contour_pred[0, :, 0], contour_pred[0, :, 1], s=10, c="r")
        ax.set_title("Average Covariance matrix")
        ax.set_xlim([0, 256])
        ax.set_ylim([256, 0])

        for k in range(0, contour_pred.shape[1], 1):
            matrix = f"{sigmas[k, 0, 0]:.2f} & {sigmas[k, 0, 1]:.2f} \\\\ {sigmas[k, 1, 0]:.2f} & {sigmas[k, 1, 1]:.2f}"
            # ax.text(contour_pred[0, k, 1], contour_pred[0, k, 0], r'$\begin{bmatrix}' + matrix + r'\end{bmatrix}$')
            confidence_ellipse(contour_pred[0, k, 1],
                               contour_pred[0, k, 0],
                               sigmas[k],
                               ax)

        plt.savefig(f"sigma_average.png", dpi=100)
        # plt.savefig(self.upload_dir / f"sigma_average.png", dpi=100)
        plt.close()

        f, ax = plt.subplots(1, 1)
        ax.scatter(contour_pred[0, :, 0], contour_pred[0, :, 1], s=10, c="r")
        ax.set_title("Average Covariance matrix determinant")
        ax.set_xlim([0, 256])
        ax.set_ylim([256, 0])

        for k in range(0, contour_pred.shape[1], 1):
            ax.text(contour_pred[0, k, 0], contour_pred[0, k, 1], f"{det[k]:.2f}")
            confidence_ellipse(contour_pred[0, k, 0],
                               contour_pred[0, k, 1],
                               sigmas[k],
                               ax)

        plt.savefig(f"sigma_average_det.png", dpi=100)
        # plt.savefig(self.upload_dir / f"sigma_average_det.png", dpi=100)
        plt.close()

        # mpl.rcParams['text.usetex'] = False
        # mpl.rcParams['text.latex.preamble'] = None

        f, ax = plt.subplots(1, 1)
        ax.scatter(contour_pred[0, :, 0], contour_pred[0, :, 1], s=10, c="r")
        ax.set_title("Average Error")
        ax.set_xlim([0, 256])
        ax.set_ylim([256, 0])

        for k in range(0, contour_pred.shape[1], 1):
            ax.text(contour_pred[0, k, 0], contour_pred[0, k, 1], f"{distances[k]:.2f}")

        plt.savefig(f"average_distance.png", dpi=100)
        # plt.savefig(self.upload_dir / f"average_distance.png", dpi=100)
        plt.close()

        plt.figure()
        plt.scatter(det, distances)
        plt.xlabel("Determinant")
        plt.ylabel("Distance")
        plt.savefig("distance_det_correlation.png", dpi=100)
        # plt.savefig(self.upload_dir / "distance_det_correlation.png", dpi=100)
        plt.close()

        # np.save(self.upload_dir / self.CORRELATION_FILE_NAME, {"conf": confidences, "acc": accuracies})

        corr, _ = pearsonr(det, distances)

        logs = {'distance_det_correlation': corr}
        pprint(logs)
