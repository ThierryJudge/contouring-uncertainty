from typing import Tuple

import numpy as np
import scipy.stats
import torch
from torch import Tensor
from torch.nn import functional as F

from contour_uncertainty.task.segmentation.segmentation_uncertainty import SegmentationUncertaintyTask


class McDropoutUncertainty(SegmentationUncertaintyTask):
    """MC Dropout system.

    Args:
        iterations: number of mc dropout iterations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, img: Tensor) -> Tuple:

        """Computes the prediction and the uncertainty map for one view.

        Args:
            img: Input image for one view [N, C, H, W] where N is the number of instants.

        Returns:
            prediction (N, K, H, W) and uncertainty map (N, H, W)
        """
        logits = [self(img) for _ in range(self.hparams.t_e)]

        if logits[0].shape[1] == 1:
            probs = [torch.sigmoid(logits[i]).detach() for i in range(self.hparams.t_e)]
            probs = torch.stack(probs, dim=0)
            y_hat = probs.mean(0)
            y_hat_prime = np.concatenate([y_hat.cpu().numpy(), 1 - y_hat.cpu().numpy()], axis=1)
            uncertainty_map = scipy.stats.entropy(y_hat_prime, axis=1)
        else:
            probs = [F.softmax(logits[i], dim=1).detach() for i in range(self.hparams.iterations)]
            probs = torch.stack(probs, dim=0)
            y_hat = probs.mean(0)
            uncertainty_map = scipy.stats.entropy(y_hat.cpu().numpy(), axis=1)

        probs = torch.transpose(probs, 0, 1).unsqueeze(3)

        return y_hat, uncertainty_map, probs
