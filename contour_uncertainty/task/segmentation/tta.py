from typing import Tuple, Dict

import numpy as np
import scipy.stats
import torch
from torch import Tensor
from torch.nn import functional as F

from contour_uncertainty.task.segmentation.segmentation_uncertainty import SegmentationUncertaintyTask


class TTAUncertainty(SegmentationUncertaintyTask):
    """Test-time augmentation task. Only for inference.
    """

    def _shared_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:  # noqa: D102 
        raise NotImplemented("No training supported for TTA tasks. Please test with pre-trained model")




    def predict_on_batch(self, img, model):
        """Computes the prediction and the uncertainty map for one view.

        Args:
            img: Input image for one view [N, C, H, W] where N is the number of instants.

        Returns:
            prediction (N, K, H, W) and uncertainty map (N, H, W)
        """
        logits = []

        for _ in range(self.hparams.t_a):
            params = self.trainer.datamodule.tta_transforms.get_params()
            items = self.trainer.datamodule.tta_transforms.apply({'image': img}, params=params)

            pred = model(items['image'])
            items = self.trainer.datamodule.tta_transforms.un_apply({'mask': pred}, params=params)

            logits.append(items['mask'])
        


        if logits[0].shape[1] == 1:
            probs = [torch.sigmoid(logits[i]).detach() for i in range(self.hparams.t_a)]
            probs = torch.stack(probs, dim=0)
            y_hat = probs.mean(0)
            y_hat_prime = np.concatenate([y_hat.cpu().numpy(), 1 - y_hat.cpu().numpy()], axis=1)
            uncertainty_map = scipy.stats.entropy(y_hat_prime, axis=1)
        else:
            probs = [F.softmax(logits[i], dim=1).detach() for i in range(self.hparams.iterations)]
            probs = torch.stack(probs, dim=0)
            y_hat = probs.mean(0)
            uncertainty_map = scipy.stats.entropy(y_hat.cpu().numpy(), axis=1)

        # Set pixels that moved outside of image to 0 uncertainty
        pad = 10
        uncertainty_map[:, 0:pad] = 0
        uncertainty_map[:, -pad:] = 0
        uncertainty_map[:, :, 0:pad] = 0
        uncertainty_map[:, :, -pad:] = 0

        return y_hat, uncertainty_map, probs

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

        pred = preds.mean(1)

        uncertainty_map = np.ones((2, 256, 256))

        return pred, uncertainty_map, samples
