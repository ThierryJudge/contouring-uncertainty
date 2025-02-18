from typing import Dict, List, Optional, Any

import numpy as np
import scipy.stats
import skimage
import torch
from matplotlib import pyplot as plt
from numpy import linalg as LA
from typing import Tuple

from skimage.morphology import erosion
from uncertainties import ufloat
from vital.data.camus.config import CamusTags, Instant, Label
from vital.data.config import Tags
import cv2
from contour_uncertainty.data.config import ContourTags, BatchResult
from contour_uncertainty.task.uncertainty import UncertaintyTask
from contour_uncertainty.utils.clinical import lv_FAC, global_longitudinal_strain, lv_area, perimeter
from contour_uncertainty.utils.contour import reconstruction
from contour_uncertainty.utils.uncertainty_projection import projected_uncertainty_value


class ContourUncertaintyTask(UncertaintyTask):


    def convert_to_mask(self, mu: np.ndarray, image_shape: Tuple, contour_samples=None, soft_mask=False):
        n = image_shape[0]
        pred = np.array(
            [self.contour_to_mask_fn(mu[i], image_shape[-2:], self.hparams.data_params.labels) for i in range(n)])

        if contour_samples is not None:
            T_e = contour_samples.shape[1]
            T_a = contour_samples.shape[2]
            pred_samples = []
            for i in range(n):
                for j in range(T_e):
                    for k in range(T_a):
                        mask = self.contour_to_mask_fn(
                            contour_samples[i, j, k],
                            image_shape[-2:],
                            self.hparams.data_params.labels,
                            apply_argmax=False
                        )
                        if soft_mask:
                            mask = mask.squeeze()
                            # for _ in range(2):
                            #     mask = erosion(mask, footprint=np.ones((3, 3)))
                            mask = skimage.filters.gaussian(mask.squeeze(), sigma=(5, 5), truncate=1)
                            mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
                            mask = mask[None]
                        pred_samples.append(mask)

            pred_samples = np.array(pred_samples).reshape(n, T_e, T_a, *image_shape[1:])
        else:
            pred_samples = None

        return pred, pred_samples

    def on_predict_start(self):
        print("PREDICT EPOCH START")
        self.umap_fn = staticmethod(self.trainer.datamodule.umap_fn)
        self.contour_to_mask_fn = staticmethod(self.trainer.datamodule.contour_to_mask_fn)
        self.skew_umap_fn = staticmethod(self.trainer.datamodule.skew_umap_fn)

    def on_fit_start(self):
        super().on_fit_start()
        self.umap_fn = staticmethod(self.trainer.datamodule.umap_fn)
        self.contour_to_mask_fn = staticmethod(self.trainer.datamodule.contour_to_mask_fn)
        self.skew_umap_fn = staticmethod(self.trainer.datamodule.skew_umap_fn)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> BatchResult:

        res = self._predict_step(batch)
        n = res.img.shape[0]

        if res.pred_samples is not None:
            res.entropy_map = np.array([self.sample_entropy(res.pred_samples[i].astype(float)) for i in range(n)])

            if len(self.hparams.data_params.labels) <= 2:
                res.pred_samples = res.pred_samples.squeeze(3)  # Squeeze channel dim if necessary [N, T, (C), H, W]
            else:
                res.pred_samples = res.pred_samples.argmax(3)

        ######### Compute point uncertainty ###############
        cov_xx = res.cov[:, :, 0, 0] ** 0.5
        cov_yy = res.cov[:, :, 1, 1] ** 0.5
        cov_det = LA.det(res.cov) ** 0.25
        cov_eigval, _ = LA.eig(res.cov)
        cov_eigval = np.sqrt(cov_eigval)
        cov_det = LA.det(res.cov) ** 0.25
        cov_eigval, _ = LA.eig(res.cov)
        cov_eigval = np.sqrt(cov_eigval)


        res.point_uncertainty = {'cov_xx': cov_xx,
                                 'cov_yy': cov_yy,
                                 'cov_det': cov_det,
                                 'cov_eigval_sum': cov_eigval.sum(-1),
                                 }

        if res.post_cov is not None:
            post_cov_eigval, _ = LA.eig(res.post_cov)
            post_cov_eigval = np.sqrt(post_cov_eigval)
            post_dict = {'post_cov_xx': res.post_cov[:, :, 0, 0] ** 0.5,
                         'post_cov_yy': res.post_cov[:, :, 1, 1] ** 0.5,
                         'post_cov_det': LA.det(res.post_cov) ** 0.25,
                         'post_cov_eigval_sum': post_cov_eigval.sum(-1)
                         }
            res.point_uncertainty.update(post_dict)

        ########### Compute instant uncertainty ############
        cov_det_mean = np.mean(cov_det, axis=-1)  # (n,)
        cov_eigenvalue_mean = np.mean(cov_eigval, axis=(-1, -2))  # (n,)
        cov_projection = np.array([projected_uncertainty_value(res.mu[i], res.cov[i]) for i in range(n)])  # (n, )

        mask = res.pred != Label.BG
        # IF mask sum is 0, use prediction sum to avoid inf uncertainty
        umap_mean = np.sum(res.uncertainty_map.squeeze(), axis=(-2, -1)) / np.sum(mask, axis=(-2, -1))

        res.instant_uncertainty = {'cov_det_mean': cov_det_mean,
                                   'cov_eigenvalue_mean': cov_eigenvalue_mean,
                                   'cov_projection': cov_projection,
                                   'umap_mean': umap_mean,
                                   }

        if res.entropy_map is not None:
            entropy_map_mean = np.sum(res.entropy_map.squeeze(), axis=(-2, -1)) / np.sum(mask, axis=(-2, -1))
            res.instant_uncertainty['entropy_mean'] = entropy_map_mean

        return res

    def _predict_step(self, batch: Any) -> BatchResult:
        raise NotImplementedError

    def log_images(
            self,
            title: str,
            num_images: int,
            axes_content: Dict[str, np.ndarray],
            info: Optional[List[str]] = None,
            scatter: np.ndarray = None,
            scatter_yerr: np.ndarray = None,
            scatter_xerr: np.ndarray = None,
    ):
        """Log images to Logger if it is a TensorBoardLogger or CometLogger.

        Args:
            title: Name of the figure.
            num_images: Number of images to log.
            axes_content: Mapping of axis name and image.
            info: Additional info to be appended to title for each image.
        """
        for i in range(num_images):
            fig, axes = plt.subplots(1, len(axes_content.keys()), squeeze=False)
            if info is not None:
                name = f"{title}_{info[i]}_{i}"
            else:
                name = f"{title}_{i}"
            plt.suptitle(name)
            axes = axes.ravel()
            for j, (ax_title, img) in enumerate(axes_content.items()):
                axes[j].imshow(img[i].squeeze())
                axes[j].set_title(ax_title)

                if scatter is not None:
                    if scatter_xerr is not None:
                        axes[j].errorbar(
                            scatter[i, :, 0],
                            scatter[i, :, 1],
                            xerr=scatter_xerr[i],
                            yerr=scatter_yerr[i],
                            c="r",
                            fmt="o",
                            markersize=1,
                        )
                    else:
                        axes[j].scatter(scatter[i, :, 1], scatter[i, :, 0], s=10, c="r")

            self.upload_fig(fig, "{}_{}".format(title, i))

            plt.close()

    def scatter_image(self, num_images: int, axes_content: Dict[str, np.ndarray], scatter: Dict = {}):
        for i in range(num_images):
            fig, axes = plt.subplots(1, len(axes_content.keys()), squeeze=False)
            plt.suptitle(f"{'val' if self.is_val_step else 'train'}_sample {i}")
            axes = axes.ravel()
            for j, (ax_title, img) in enumerate(axes_content.items()):

                axes[j].imshow(img[i].squeeze())
                axes[j].set_title(ax_title)

                for key, data in scatter.items():
                    axes[j].scatter(data[i, :, 0], data[i, :, 1], s=10, label=key)

            plt.legend()

            self.upload_fig(fig, f"{'val' if self.is_val_step else 'train'}_sample {i}")

            plt.close()
