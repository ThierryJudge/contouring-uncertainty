from pprint import pprint
from random import sample
from typing import List, Any

import numpy as np
import pandas as pd
from pytorch_lightning import Callback

from contour_uncertainty.data.config import ContourTags, BatchResult
from contour_uncertainty.results.calibration import Calibration
from contour_uncertainty.results.clinical.clinical_metrics import bbox
from contour_uncertainty.task.segmentation.segmentation_uncertainty import get_contour
from scripts.utils.calibration import plot_calibration
from vital.data.camus.config import Label
from vital.data.config import Tags
from matplotlib import pyplot as plt
from scipy.stats import norm
import seaborn as sns
import cv2
from skimage.morphology import erosion

from contour_uncertainty.utils.skew_normal import plot_skewed_normals
from contour_uncertainty.utils.uncertainty_projection import projected_uncertainty
from contour_uncertainty.utils.contour import contour_spline
from skimage.draw import line
from skimage.morphology import dilation, erosion
import os
from contour_uncertainty.utils.plotting import confidence_ellipse
from contour_uncertainty.utils.skew_umap import skew_umap
import torch
from contour_uncertainty.data.lung.utils import split_landmarks as split_lung
from contour_uncertainty.data.camus.utils import split_landmarks as split_us


class UncertaintyErrorMutualInfo(Callback):
    """Evaluator for uncertainty error overlap.

    Args:
        uncertainty_threshold: threshold for the uncertainty to generate binary mask.
    """

    def __init__(self, bins=20, max_figures=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bins = bins
        self.max_figures = max_figures

        os.mkdir("mi-figures")

    @staticmethod
    def compute_morph_unc(pred: np.ndarray, thickness: int):
        dilatation_mask = np.copy(pred)
        erosion_mask = np.copy(pred)

        prev_erosion = np.copy(erosion_mask)
        prev_dilatation = np.copy(dilatation_mask)

        uncertainty_map = np.zeros_like(pred).squeeze()

        for j in range(thickness):
            dilatation_mask = dilation(dilatation_mask, footprint=np.ones((3, 3)))
            erosion_mask = erosion(erosion_mask, footprint=np.ones((3, 3)))

            erosion_edges = prev_erosion ^ erosion_mask
            dilatation_edges = prev_dilatation ^ dilatation_mask

            prev_erosion = np.copy(erosion_mask)
            prev_dilatation = np.copy(dilatation_mask)

            # uncertainty_map = uncertainty_map + (1 + -j / thickness) * (erosion_edges.clip(max=1) + dilatation_edges.clip(max=1)).astype(float)
            uncertainty_map = uncertainty_map + norm.pdf(j, loc=0, scale=1.5) * (
                    erosion_edges.clip(max=1) + dilatation_edges.clip(max=1)).astype(float)

        return uncertainty_map


    @staticmethod
    def compute_mi(error, uncertainty, norm=True):
        """Computes mutual information between error and uncertainty.

        Args:
            error: numpy binary array indicating error.
            uncertainty: numpy float array indicating uncertainty.

        Returns:
            mutual_information
        """
        hist_2d, x_edges, y_edges = np.histogram2d(error.ravel(), uncertainty.ravel())

        pxy = hist_2d / float(np.sum(hist_2d))
        px = np.sum(pxy, axis=1)  # marginal for x over y
        py = np.sum(pxy, axis=0)  # marginal for y over x
        px_py = px[:, None] * py[None, :]  # Broadcast to multiply marginals
        # Now we can do the calculation using the pxy, px_py 2D arrays
        nzs = pxy > 0  # Only non-zero pxy values contribute to the sum

        mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

        if norm:
            hx = -np.sum(np.multiply(px[px > 0], np.log(px[px > 0])))
            hy = -np.sum(np.multiply(py[py > 0], np.log(py[py > 0])))
            mi = 2 * mi / (hx + hy)

        return mi.item()

    def on_predict_epoch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: List[List[BatchResult]]
    ) -> None:
        """Computes uncertainty error overlap for all patients.

        Args:
            outputs: List of view results including image, prediction, groundtruth and uncertainty prediction.

        Returns:
            average uncertainty-error overlap.
        """

        metrics = {}
        error_sums = []
        c = 0

        confidences, accuracies, preds, gts = Calibration.get_pixel_confidences_and_accuracies(outputs[0],
                                                                                               prob_string='entropy_map')

        for c, res in enumerate(outputs[0]):
            for i in range(res.img.shape[0]):
                error = 1 * ~np.equal(res.pred[i], res.gt[i])
                error_sums.append(error.sum())

                contour_umap = res.uncertainty_map[i]
                contour_mi = self.compute_mi(error, contour_umap)

                edge_umap = self.compute_morph_unc(res.pred[i], 5)
                edge_mi = self.compute_mi(error, edge_umap)

                if f"{res.id}_{i}" in metrics.keys():
                    print(f"{res.id}_{i}")

                metrics[f"{res.id}_{i}"] = {
                    "umap_mi": contour_mi,
                    "edge_mi": edge_mi,
                }

                if res.entropy_map is not None:
                    sample_umap = res.entropy_map[i]
                    sample_mi = self.compute_mi(error, sample_umap)
                    metrics[f"{res.id}_{i}"]["entropy_mi"] = sample_mi

                f, ax = plt.subplots(1, 1)
                ax.set_axis_off()

                ax.imshow(res.img[i].squeeze(), cmap='gray')

                rmin, rmax, cmin, cmax = bbox(res.gt[i], pad=30)
                ax.set_ylim(rmax, rmin)
                ax.set_xlim(cmin, cmax)

                if res.mu is not None:
                    for index in range(10):
                        contour = contour_spline(res.contour_samples[i, 0, index, :21], close=True)
                        ax.plot(contour[:, 0], contour[:, 1], linewidth=1.5, zorder=0)
                    ax.scatter(res.mode[i, :, 0], res.mode[i, :, 1], s=5, c="r", label="Pred")
                    for index in range(0, res.mu.shape[1], 1):
                        confidence_ellipse(res.mu[i, index, 0], res.mu[i, index, 1], res.cov[i, index], ax, n_std=2)
                else:
                    for index in range(10):
                        contour = get_contour(res.pred_samples[i, 0, index].squeeze().round())
                        ax.plot(contour[:, 1], contour[:, 0], linewidth=1.5, zorder=0)

                rmin, rmax, cmin, cmax = bbox(res.entropy_map[i], pad=5)
                ins = ax.inset_axes([0.75, 0.75, 0.25, 0.25])
                # ins.set_axis_off()
                ins.imshow(res.entropy_map[i, rmin:rmax, cmin:cmax], cmap='gray')
                ins.set_xticks([])
                ins.set_yticks([])
                ins.spines['top'].set_edgecolor('white')
                ins.spines['right'].set_edgecolor('white')
                ins.spines['left'].set_edgecolor('white')
                ins.spines['bottom'].set_edgecolor('white')

                ins2 = ax.inset_axes([0, 0.75, 0.25, 0.25])
                # ins2.set_axis_off()
                ins2.set_xticks([])
                ins2.set_yticks([])
                ins2.imshow(error[rmin:rmax, cmin:cmax], cmap='gray')
                ins2.spines['top'].set_edgecolor('white')
                ins2.spines['right'].set_edgecolor('white')
                ins2.spines['left'].set_edgecolor('white')
                ins2.spines['bottom'].set_edgecolor('white')

                with sns.axes_style("darkgrid"):
                    ins3 = ax.inset_axes([0, 0, 0.3, 0.3])
                    ins3.set_xticks([])
                    ins3.set_yticks([])
                    fg = preds[2 * c + i] + gts[2 * c + i]
                    not_bg = fg != 0  # Background class is always 0
                    sample_confidences = confidences[2 * c + i][not_bg]
                    sample_accuracies = accuracies[2 * c + i][not_bg]

                    twin_ax = ins3.twinx()
                    twin_ax.set_xticks([])
                    twin_ax.set_yticks([])
                    plot_calibration(ins3, sample_confidences, sample_accuracies, 'method', twinax=twin_ax)
                    ins3.plot([0, 1], [0, 1], "--", c="k", label="Perfect calibration")

                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                plt.margins(0, 0)
                plt.tight_layout()
                plt.savefig(f"mi-figures/{res.id.replace('/', '-')}_{i}.png", dpi=300, bbox_inches='tight',
                            pad_inches=0)
                plt.show()
                plt.close()
                # plt.show()

        df = pd.DataFrame(metrics).T
        print(df)

        results = {}
        for s in ['entropy', 'umap', 'edge']:
            if f'{s}_mi' in df.columns:
                mi_list = np.array(df[f'{s}_mi'])

                mean = np.mean(mi_list)

                print(mi_list.shape)
                print(len(error_sums))
                print(np.array(error_sums).shape)

                weighted_mi = np.average(mi_list, weights=error_sums)

                results[f"{s}_mi"] = weighted_mi
                results[f"{s}_mi(mw)"] = mean
                results[f"{s}_mi(median)"] = np.median(mi_list)

                plt.figure()
                plt.boxplot(mi_list)
                plt.ylim([0, 0.25])
                plt.savefig(f"{s}_boxplot.png")
                plt.close()

        pprint(results)
        trainer.logger.log_metrics(results)