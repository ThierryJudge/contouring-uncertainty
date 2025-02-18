from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from scipy import ndimage
from torch import Tensor
from torch.nn import functional as F

from contour_uncertainty.data.config import BatchResult
from contour_uncertainty.task.segmentation.utils import big_blob
from contour_uncertainty.task.uncertainty import UncertaintyTask
from vital.data.config import Tags
from vital.metrics.train.metric import DifferentiableDiceCoefficient


class SegmentationUncertaintyTask(UncertaintyTask):

    def __init__(self, ce_weight: float = 0.1, dice_weight: float = 1, *args, **kwargs):
        """Initializes the metric objects used repeatedly in the train/eval loop.

        Args:
            ce_weight: float, weight for cross-entropy term in loss.
            dice_weight: float, weight for dice term in loss.
            *args: Positional arguments to pass to the parent's constructor.
            **kwargs: Keyword arguments to pass to the parent's constructor.
        """
        super().__init__(*args, **kwargs)
        self._dice = DifferentiableDiceCoefficient(include_background=False, reduction="none")

    def compute_loss(self, y, y_hat):
        if y_hat.shape[1] == 1:
            ce = F.binary_cross_entropy_with_logits(y_hat.squeeze(), y.type_as(y_hat))
        else:
            ce = F.cross_entropy(y_hat, y)

        dice_values = self._dice(y_hat, y)
        dices = {f"dice/{label}": dice for label, dice in zip(self.hparams.data_params.labels[1:], dice_values)}
        mean_dice = dice_values.mean()

        loss = (self.hparams.ce_weight * ce) + (self.hparams.dice_weight * (1 - mean_dice))

        return loss, dices, mean_dice, ce

    def _shared_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:  # noqa: D102
        x, y = batch[Tags.img], batch[Tags.gt]

        y_hat = self.model(x)

        if getattr(self.model, 'deep_supervision', None) and not self.is_val_step:
            loss, dices, mean_dice, ce = self.compute_loss(y, y_hat[0])
            for i, pred in enumerate(y_hat[1:]):
                downsampled_label = nn.functional.interpolate(y.unsqueeze(0).float(), pred.shape[2:]).squeeze(0)
                l, _, _, _ = self.compute_loss(downsampled_label.long(), pred)
                loss += 0.5 ** (i + 1) * l
            c_norm = 1 / (2 - 2 ** (-len(y_hat)))
            loss = c_norm * loss
        else:
            loss, dices, mean_dice, ce = self.compute_loss(y, y_hat)

        if self.is_val_step and batch_idx == 0 and self.hparams.log_figures:
            if y_hat.shape[1] == 1:
                y_hat = torch.sigmoid(y_hat).round()
            else:
                y_hat = y_hat.argmax(dim=1)

            self.log_images(
                title="Sample",
                num_images=min(5, x.shape[0]),
                axes_content={
                    "Image": x.cpu().squeeze().numpy(),
                    "Gt": y.squeeze().cpu().numpy(),
                    "Pred": y_hat.detach().cpu().squeeze().numpy(),
                },
            )

        # Format output
        return {"loss": loss, "ce": ce, "dice": mean_dice, **dices}

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> BatchResult:
        img = batch[Tags.img]
        n = img.shape[0]
        gt = batch[Tags.gt].cpu().numpy()

        # Return pred and samples (and uncertainty map)
        pred, umap, samples = self.predict(img)  # (N, C, H, W), (N, H, W), (N, Te, Ta, C, H, W)

        samples = samples.cpu().numpy()

        '''
        Post process samples 
            1. Fill holes 
            2. Only select largest object. 
        Currently work on binary segmentation. 
        '''

        samples_raw = np.copy(samples) # save copy of samples without rounding
        samples = samples.round()


        for i in range(samples.shape[0]):
            for j in range(samples.shape[1]):
                for k in range(samples.shape[2]):
                    s = samples[i,j,k]
                    s = ndimage.binary_fill_holes(s.squeeze())[None]
                    s = big_blob(s)
                    samples[i,j,k] = s

        samples = samples_raw * samples # Multiply with raw samples to preserve probabilities.

        # UNCOMMENT to see effect of post-processing.
        # from matplotlib import pyplot as plt
        # f, axes = plt.subplots(2, 5)
        # # axes = axes.ravel()
        # for i in range(5):
        #     axes[0, i].imshow(samples_raw[0, 0, i].squeeze())
        #     axes[1, i].imshow(samples[0, 0, i].squeeze())

        entropy = np.array([self.sample_entropy(samples[i]) for i in range(n)])  # Do this before argmax / round

        # Remove any entropy too close to the boarder of the image.
        pad = 10
        entropy[:, 0:pad] = 0
        entropy[:, -pad:] = 0
        entropy[:, :, 0:pad] = 0
        entropy[:, :, -pad:] = 0

        if pred.shape[1] == 1:
            pred = pred.round().squeeze().cpu().numpy().astype(int)
            samples = samples.round().astype(int).squeeze(axis=3)  # Squeeze channel axis
        else:
            pred = pred.argmax(1).cpu().numpy()
            samples = samples.argmax(3)

        instant_uncertainty = {'umap_mean': umap.mean((-2, -1)),
                               'entropy_mean': entropy.mean((-2, -1))}

        voxelspacing, instants = self.get_voxelspacing_and_instants(batch)

        res = BatchResult(id=batch[Tags.id],
                          labels=self.hparams.data_params.labels,
                          img=img.cpu().numpy(),
                          gt=gt,
                          pred=pred,
                          pred_samples=samples.astype(bool),
                          uncertainty_map=umap,
                          entropy_map=entropy,
                          instant_uncertainty=instant_uncertainty,
                          instants=instants,
                          voxelspacing=voxelspacing,
                          )
        return res

    def predict(self, img: Tensor) -> Tuple:  # noqa: D102
        raise NotImplementedError

    def log_images(
            self, title: str, num_images: int, axes_content: Dict[str, np.ndarray], info: Optional[List[str]] = None
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
            name = f"{title}_{info[i]}_{i}" if info is not None else f"{title}_{i}"
            plt.suptitle(name)
            axes = axes.ravel()
            for j, (ax_title, img) in enumerate(axes_content.items()):
                axes[j].imshow(img[i].squeeze())
                axes[j].set_title(ax_title)

            self.upload_fig(fig, "{}_{}".format(title, i))

            plt.close()
