from typing import Dict, Tuple

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional
from torch import Tensor

from contour_uncertainty.data.camus.dataset import ContourTags
from contour_uncertainty.task.regression.aleatoric import AleatoricUncertaintyTask
from contour_uncertainty.task.regression.dsnt.utils import flat_softmax, dsnt, normalized_to_pixel_coordinates, \
    euclidean_losses
from vital.data.config import Tags


class DSNTAleatoric(AleatoricUncertaintyTask):
    """
    Reference: https://github.com/anibali/dsntnn
    """

    def __init__(
            self,
            covar: bool = True,
            mse_weight: float = 1,
            log_penalty_weight: float = 1,
            iterations: int = 1,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

    def configure_model(self) -> nn.Module:
        """Configure the network architecture used by the system."""

        in_shape = self.hparams.data_params.in_shape
        out_shape = self.hparams.data_params.out_shape
        return hydra.utils.instantiate(
            self.hparams.model,
            input_shape=in_shape,
            output_shape=(out_shape[0], in_shape[0], in_shape[1]),
        )

    def _shared_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:  # noqa: D102
        x, y = batch[Tags.img], batch[ContourTags.contour]

        image_size = x.shape[2]

        # Forward
        heatmaps = self.model(x)

        heatmaps = flat_softmax(heatmaps)
        coords, var, covar = dsnt(heatmaps)
        covar = covar if self.hparams.covar else 0

        pixel_coords = normalized_to_pixel_coordinates(coords, image_size)
        pixel_var = var * (image_size / 2) ** 2
        pixel_covar = covar * (image_size / 2) ** 2
        pixel_sigma = self.get_cov_matrix(pixel_var[..., 0], pixel_var[..., 1], pixel_covar)

        distance_loss = euclidean_losses(pixel_coords, y)

        mu_flat = torch.flatten(pixel_coords, 0, 1).unsqueeze(-1)
        y_flat = torch.flatten(y, 0, 1).unsqueeze(-1)
        Sigma_flat = torch.flatten(pixel_sigma, 0, 1)
        loss_term1 = self.hparams.log_penalty_weight * torch.log(torch.det(Sigma_flat))
        loss_term2 = self.hparams.mse_weight * (
                ((mu_flat - y_flat).transpose(-1, -2) @ torch.inverse(Sigma_flat)) @ (mu_flat - y_flat))

        loss = (loss_term1 + loss_term2).mean()

        logs = {'loss': loss, 'distance_loss': distance_loss.mean(),
                'loss_term1': loss_term1.mean(), 'loss_term2': loss_term2.mean()}

        if self.is_val_step:

            gt = batch[Tags.gt]
            pred = []
            for i in range(len(x)):
                pred.append(self.contour_to_mask_fn(pixel_coords[i].cpu().squeeze().numpy(),
                                                    (gt.shape[1], gt.shape[2]),
                                                    labels=self.hparams.data_params.labels,
                                                    reconstruction_type='linear'))
            pred = np.array(pred)
            dice = self.dice(pred, gt.cpu().numpy())
            logs['dice'] = dice
            if batch_idx == 0 and self.hparams.log_figures:
                self.log_images(
                    title="Reconstruction",
                    num_images=5,
                    axes_content={
                        "Image": x.cpu().squeeze().numpy(),
                        "Gt": gt.squeeze().cpu().numpy(),
                        "Pred": pred,
                    },
                )

                self.log_images(
                    title="Heatmaps",
                    num_images=5,
                    axes_content={
                        "Heat": heatmaps.sum(1).squeeze().cpu().numpy(),
                    },
                )
                self.plot_image(
                    num_images=5,
                    axes_content={
                        "Image": x.cpu().squeeze().numpy(),
                    },
                    mu=pixel_coords.cpu().squeeze().numpy(),
                    mu_gt=y.cpu().squeeze().numpy(),
                    Sigma=pixel_sigma.cpu().squeeze().numpy()
                )

        return logs

    def predict_on_batch(self, img, model):

        image_size = img.shape[2]

        heatmaps = flat_softmax(model(img))
        coords, var, covar = dsnt(heatmaps)
        covar = covar if self.hparams.covar else 0

        pixel_coords = normalized_to_pixel_coordinates(coords, image_size)
        pixel_var = var * (image_size / 2) ** 2
        pixel_covar = covar * (image_size / 2) ** 2
        pixel_sigma = self.get_cov_matrix(pixel_var[..., 0], pixel_var[..., 1], pixel_covar)

        return pixel_coords, pixel_sigma

    def predict(self, img, scale=False) -> Tuple:  # noqa: D102
        S, cov = [], []

        self.hparams.t_e = len(self.model) if self.ensembling else self.hparams.t_e

        for i in range(self.hparams.t_e):
            model = self.model[i] if self.ensembling else self.model
            pixel_coords, pixel_sigma = self.predict_on_batch(img, model)
            S.append(pixel_coords)
            cov.append(pixel_sigma)

        S = torch.stack(S).swapaxes(1, 0)  # (T, N, K, 2)
        cov = torch.stack(cov).swapaxes(1, 0)

        S = S.cpu().detach()
        cov = cov.cpu().detach()


        return S, cov
