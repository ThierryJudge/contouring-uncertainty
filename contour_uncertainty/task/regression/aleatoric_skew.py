from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import torch
from hydra.utils import to_absolute_path
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger

from contour_uncertainty.sampler.posterior_shape_model.psm import PosteriorShapeModelSampler
from contour_uncertainty.sampler.posterior_shape_model.psm_skew import SkewPosteriorShapeModelSampler
from contour_uncertainty.sampler.posterior_shape_model.psm_skew_sequence import SequenceSkewPSMSampler
from contour_uncertainty.sampler.posterior_shape_model.sequence_sampler import SequencePSMSampler
from contour_uncertainty.utils.contour import contour_spline
from contour_uncertainty.utils.plotting import confidence_ellipse
from vital.data.camus.config import CamusTags
from vital.data.config import Tags

from contour_uncertainty.data.config import ContourTags, BatchResult
from contour_uncertainty.task.regression.aleatoric import AleatoricUncertaintyTask
from contour_uncertainty.utils.skew_normal import plot_skewed_normals
from contour_uncertainty.utils.skew_umap import skew_umap


class SkewUncertaintyTask(AleatoricUncertaintyTask):

    def __init__(self, psm_path: str, seq_psm_path, skew_indices: List[int] = None,*args, **kwargs):

        super().__init__(psm_path, seq_psm_path, *args, **kwargs)

        self.skew_indices = list(range(self.hparams.data_params.out_shape[0])) if skew_indices is None else skew_indices

        print(self.skew_indices)

        if self.hparams.sequence_sampler:
            print("Using sequence sampler")
            self.sampler = SequenceSkewPSMSampler(sequence_psm_path=Path(to_absolute_path(seq_psm_path)),
                                              psm_path=Path(to_absolute_path(psm_path)))
        else:
            print("Not using sequence sampler")
            self.sampler = SkewPosteriorShapeModelSampler(psm_path=Path(to_absolute_path(psm_path)),
                                                          skew_indices=skew_indices)


    def predict(self, img) -> Tuple:  # noqa: D102
        raise NotImplementedError

    def sample(self, mu, cov, alpha, T):
        n = mu.shape[0]
        contour_samples = [self.sampler(mu[:, t_e], cov[:, t_e], alpha[:, t_e], n=T).numpy() for t_e in
                           range(mu.shape[1])]
        contour_samples=np.array(contour_samples).transpose(1, 0, 2, 3, 4)
        return contour_samples

    def _predict_step(self, batch: Any) -> BatchResult:
        img = batch[Tags.img]
        contour = batch[ContourTags.contour]
        gt = batch[Tags.gt].cpu().numpy() if Tags.gt in batch.keys() else None
        n = img.shape[0]

        mu, cov, alpha = self.predict(img)

        contour_samples = self.sample(mu, cov, alpha, 25) # (N, T_e, T_a, K, 2)

        mu_mean = mu.mean(dim=1, keepdim=True)

        cov_al = cov.mean(1)
        cov_ep = torch.mean((mu - mu_mean)[..., None] * (mu - mu_mean)[..., None].swapaxes(-1, -2), dim=1)

        mu = mu.mean(dim=1)
        alpha = alpha.mean(dim=1)
        cov = cov_ep + cov_al

        mu = mu.cpu().numpy()
        cov = cov.cpu().numpy()
        alpha = alpha.cpu().numpy()
        contour = contour.cpu().numpy()

        post_mu = contour_samples.mean(axis=(1,2))
        post_cov = np.zeros((n, 21, 2, 2))
        for idx in range(contour_samples.shape[0]):
            for k in range(contour_samples.shape[3]):
                post_cov[idx, k] = np.cov(contour_samples[idx, :, :, k].reshape(-1, 2).T)


        mode = []
        umap = []
        for i in range(n):
            projected_mode, u = self.skew_umap_fn(mu[i], cov[i], alpha[i], self.hparams.data_params.labels)
            mode.append(projected_mode)
            umap.append(u)

        mode = np.array(mode)
        umap = np.array(umap)

        pred, pred_samples = self.convert_to_mask(mode, img.shape, contour_samples)

        if CamusTags.metadata in batch.keys():
            instants = batch[CamusTags.metadata].instants
            voxelspacing = np.array(batch[CamusTags.metadata].voxelspacing)
            if batch[CamusTags.metadata].gt is not None:
                voxelspacing = voxelspacing * batch[CamusTags.metadata].gt.shape / gt.shape
            voxelspacing = voxelspacing[1:]
        else:
            instants = None
            voxelspacing = None

        res = BatchResult(id=batch[Tags.id],
                          labels=self.hparams.data_params.labels,
                          img=img,
                          contour=contour,
                          gt=gt,
                          mu=mu,
                          mode=mode,
                          cov=cov,
                          contour_samples=contour_samples,
                          pred_samples=pred_samples,
                          pred=pred,
                          uncertainty_map=umap,
                          alpha=alpha,
                          instants=instants,
                          voxelspacing=voxelspacing,
                          post_mu=post_mu,
                          post_cov=post_cov
                          )

        return res

    def plot_image_skew(self, num_images: int, axes_content: Dict[str, np.ndarray], mu_gt, mu, cov, alpha):
        for i in range(num_images):
            fig, axes = plt.subplots(1, len(axes_content.keys()), squeeze=False)
            plt.suptitle(f'Sample {i}')
            axes = axes.ravel()
            for j, (ax_title, img) in enumerate(axes_content.items()):
                axes[j].imshow(img[i].squeeze())
                axes[j].set_title(ax_title)
                axes[j].scatter(mu[i, :, 0], mu[i, :, 1], s=10, c="r")
                axes[j].scatter(mu_gt[i, :, 0], mu_gt[i, :, 1], s=10, c="b")

                plot_skewed_normals(axes[j], mu[i], cov[i], alpha[i])

            if isinstance(self.trainer.logger, TensorBoardLogger):
                self.trainer.logger.experiment.add_figure("{}_{}".format(f'Sample', i), fig, self.current_epoch)
            if isinstance(self.trainer.logger, CometLogger):
                self.trainer.logger.experiment.log_figure("{}_{}".format(f'Sample', i), fig, step=self.current_epoch)

            plt.close()
