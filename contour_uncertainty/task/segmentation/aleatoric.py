from typing import Dict, Tuple

import hydra
import numpy as np
import scipy
import torch
import torch.distributions as distributions
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torchmetrics.utilities.data import to_onehot
from vital.data.config import Tags
from vital.metrics.train.metric import DifferentiableDiceCoefficient

from contour_uncertainty.task.segmentation.segmentation_uncertainty import SegmentationUncertaintyTask


class AleatoricUncertainty(SegmentationUncertaintyTask):
    """Aleatoric uncertainty system.

    Args:
        iterations: number of mc dropout iterations.
    """

    def __init__(self, iterations: int = 10, is_log_sigma: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters("iterations", "is_log_sigma")
        self.iterations = iterations
        self.is_log_sigma = is_log_sigma
        self._dice = DifferentiableDiceCoefficient(include_background=False, reduction="none", apply_activation=False)

        assert not (is_log_sigma and len(self.hparams.data_params.labels) > 1), "Does not work with >1 labels"

    def configure_model(self) -> nn.Module:
        """Configure the network architecture used by the system."""
        return hydra.utils.instantiate(
            self.hparams.model,
            input_shape=self.hparams.data_params.in_shape,
            output_shape=self.hparams.data_params.out_shape,
            ssn_rank=1  # Use two heads, one for pred, one for variance
        )

    def _shared_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:  # noqa: D102
        x, y = batch[Tags.img], batch[Tags.gt]

        # Forward
        logits, sigma = self(x)  # (N, C, H, W), (N, C, H, W)
        binary = logits.shape[1] == 1

        sigma = F.softplus(sigma)

        if self.is_log_sigma:
            distribution = distributions.Normal(logits, torch.exp(sigma))
        else:
            distribution = distributions.Normal(logits, sigma + 1e-8)

        x_hat = distribution.rsample((self.iterations,))

        if binary:
            mc_expectation = torch.sigmoid(x_hat).mean(dim=0)
            ce = F.binary_cross_entropy(mc_expectation.squeeze(), y.float())
        else:
            mc_expectation = F.softmax(x_hat, dim=2).mean(dim=0)
            log_probs = mc_expectation.log()
            ce = F.nll_loss(log_probs, y)

        dice_values = self._dice(mc_expectation, y)
        dices = {f"dice_{label}": dice for label, dice in zip(self.hparams.data_params.labels[1:], dice_values)}
        mean_dice = dice_values.mean()

        loss = (self.hparams.ce_weight * ce) + (self.hparams.dice_weight * (1 - mean_dice))

        if self.is_val_step and batch_idx == 0 and self.hparams.log_figures:
            mc = mc_expectation.detach().cpu()
            if binary:
                y_hat = torch.sigmoid(logits).round()
                sigma_pred = sigma
                uncertainty_map = scipy.stats.entropy(np.concatenate([mc, 1 - mc], axis=1), axis=1)
            else:
                y_hat = logits.argmax(dim=1)
                prediction_onehot = to_onehot(y_hat, num_classes=len(self.hparams.data_params.labels)).type(torch.bool)
                sigma_pred = torch.where(prediction_onehot, sigma, sigma * 0).sum(dim=1)
                uncertainty_map = scipy.stats.entropy(mc, axis=1)

            self.log_images(
                title="Sample",
                num_images=5,
                axes_content={
                    "Image": x.cpu().squeeze().numpy(),
                    "Gt": y.squeeze().cpu().numpy(),
                    "Pred": y_hat.detach().cpu().squeeze().numpy(),
                    "Sigma": sigma_pred.detach().cpu().squeeze().numpy(),
                    "Entropy": uncertainty_map.squeeze(),
                },
            )

        # Format output
        return {"loss": loss, "ce": ce, "dice": mean_dice, **dices}

    def predict_on_batch(self, img, model):
        logits, sigma = model(img)
        sigma = F.softplus(sigma)

        if self.is_log_sigma:
            distribution = distributions.Normal(logits, torch.exp(sigma))
        else:
            distribution = distributions.Normal(logits, sigma + 1e-8)

        samples = distribution.rsample((self.hparams.t_a,))

        if logits.shape[1] == 1:
            y_hat = torch.sigmoid(logits)
            samples = torch.sigmoid(samples)
            sigma = sigma.squeeze(1)
        else:
            y_hat = F.softmax(logits, dim=1)
            prediction_onehot = to_onehot(y_hat.argmax(1), num_classes=len(self.hparams.data_params.labels)).type(
                torch.bool)
            sigma = torch.where(prediction_onehot, sigma, sigma * 0).sum(dim=1)
            samples = F.softmax(samples, dim=2)

        return y_hat, sigma, samples

    def predict(self, img: Tensor) -> Tuple:
        """Computes the prediction and the uncertainty map for one view.

        Args:
            img: Input image for one view [N, C, H, W] where N is the number of instants.

        Returns:
            prediction (N, K, H, W) and uncertainty map (N, H, W)
        """

        preds, sigmas, samples = [], [], []

        for i in range(self.hparams.t_e):
            model = self.model[i] if self.ensembling else self.model
            pred, sigma, batch_samples = self.predict_on_batch(img, model)
            preds.append(pred)
            samples.append(batch_samples.swapaxes(1, 0))
            sigmas.append(sigma)

        preds = torch.stack(preds).swapaxes(1, 0)
        samples = torch.stack(samples).swapaxes(1, 0)
        sigmas = torch.stack(sigmas).swapaxes(1, 0)

        pred = preds.mean(1)
        sigma = sigmas.mean(1)

        return pred, sigma.cpu().numpy(), samples
