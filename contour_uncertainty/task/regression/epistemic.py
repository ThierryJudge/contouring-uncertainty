from typing import Dict, Any

import numpy as np
import torch
from torch import Tensor
from vital.data.config import Tags

from contour_uncertainty.data.config import ContourTags, BatchResult
from contour_uncertainty.task.regression.contour_uncertainty import ContourUncertaintyTask
from vital.data.camus.config import CamusTags


class EpistemicUncertaintyTask(ContourUncertaintyTask):

    def predict(self, img) -> Tensor:  # noqa: D102
        """Return set of contours
        Args:
            img:

        Returns:

        """
        raise NotImplementedError

    def _predict_step(self, batch: Any) -> BatchResult:
        img = batch[Tags.img]
        contour = batch[ContourTags.contour]
        gt = batch[Tags.gt].cpu().numpy() if Tags.gt in batch.keys() else None
        n = img.shape[0]

        contour_samples = self.predict(img)  # (T, N, K, 2)

        mu = contour_samples.mean(0)
        cov = torch.mean((contour_samples - mu)[..., None] * (contour_samples - mu)[..., None].swapaxes(-1, -2), dim=0)

        contour_samples = contour_samples.permute(1, 0, 2, 3)  # (n, T, K, 2)

        mu = mu.cpu().numpy()
        cov = cov.cpu().numpy()
        contour_samples = contour_samples.cpu().numpy()

        pred, pred_samples = self.convert_to_mask(mu, img.shape, contour_samples)

        pred_samples = pred_samples.astype(float)

        uncertainty_map = np.array([self.umap_fn(mu[i], cov[i], self.hparams.data_params.labels) for i in range(n)])

        if CamusTags.metadata in batch.keys():
            instants = batch[CamusTags.metadata].instants
            voxelspacing = batch[CamusTags.metadata].voxelspacing
            voxelspacing = voxelspacing * batch[CamusTags.metadata].gt.shape / gt.shape
        else:
            instants = None
            voxelspacing = None

        res = BatchResult(id=batch[Tags.id],
                          labels=self.hparams.data_params.labels,
                          img=img.cpu().numpy(),
                          contour=contour.cpu().numpy(),
                          gt=gt,
                          mu=mu,
                          mode=mu,
                          cov=cov,
                          contour_samples=contour_samples,  # (n, T, K, 2),
                          pred_samples=pred_samples,  # TODO find where to apply argmax
                          pred=pred,
                          uncertainty_map=uncertainty_map,
                          instants=instants,
                          voxelspacing=voxelspacing[1:],)

        return res
