from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
from hydra.utils import to_absolute_path
from matplotlib import pyplot as plt
from numpy import linalg as LA
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger
from scipy.stats import pearsonr, multivariate_normal
from torch.nn import functional as F

from contour_uncertainty.sampler.posterior_shape_model.sequence_sampler import SequencePSMSampler
from vital.data.camus.config import CamusTags, Label
from vital.data.config import Tags

from contour_uncertainty.data.config import BatchResult, ContourTags
from contour_uncertainty.sampler.posterior_shape_model.psm import PosteriorShapeModelSampler
from contour_uncertainty.task.regression.contour_uncertainty import ContourUncertaintyTask
from contour_uncertainty.utils.plotting import confidence_ellipse
from contour_uncertainty.utils.umap import uncertainty_map


class AleatoricUncertaintyTask(ContourUncertaintyTask):

    def __init__(
            self,
            psm_path: str,
            seq_psm_path,
            sequence_sampler=False,
            *args,
            **kwargs
    ):
        """Initializes the metric objects used repeatedly in the train/eval loop.

        Args:
            *args: Positional arguments to pass to the parent's constructor.
            **kwargs: Keyword arguments to pass to the parent's constructor.
        """
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        if self.hparams.sequence_sampler:
            print("Using sequence sampler")
            self.sampler = SequencePSMSampler(sequence_psm_path=Path(to_absolute_path(seq_psm_path)),
                                              psm_path=Path(to_absolute_path(psm_path)))
        else:
            print("Not using sequence sampler")
            self.sampler = PosteriorShapeModelSampler(psm_path=Path(to_absolute_path(psm_path)))

    def predict(self, img) -> Tuple:  # noqa: D102
        raise NotImplementedError

    def sample(self, mu: torch.Tensor, cov: torch.Tensor, t_a: int):
        """ Sample using PSM sampler (standard or sequence).

        Args:
            mu: Point prediction mean
            cov: Point prediction covariance
            T: int, number of samples

        Returns:
            Samples
        """
        n = mu.shape[0]
        if isinstance(self.sampler, SequencePSMSampler):
            # contour_samples = [self.sampler(mu[:, t_e], cov[:, t_e], t_a, debug_img=img.cpu().numpy()).numpy() for t_e in range(mu.shape[1])]
            contour_samples = [self.sampler(mu[:, t_e], cov[:, t_e], n=t_a).numpy() for t_e in range(mu.shape[1])]
            contour_samples = np.array(contour_samples).transpose((2, 0, 1, 3, 4))  # (N, T_e, T_a, K, 2)
        elif isinstance(self.sampler, PosteriorShapeModelSampler):
            contour_samples = [
                [self.sampler(mu[i, t_e], cov[i, t_e], n=t_a).numpy() for t_e in range(mu.shape[1])] for i in range(n)
            ]
            contour_samples = np.array(contour_samples)  # (N, T_e, T_a, K, 2)
        else:
            raise ValueError("Unrecognized sampler type")

        return contour_samples

    def _predict_step(self, batch: Any) -> BatchResult:
        img = batch[Tags.img]
        contour = batch[ContourTags.contour]
        gt = batch[Tags.gt].cpu().numpy() if Tags.gt in batch.keys() else None
        n = img.shape[0]

        mu, cov = self.predict(img)  # (N, T_e, K, 2), (N, T_e, K, 2, 2)

        contour_samples = self.sample(mu, cov, self.hparams.t_a)

        mu_mean = mu.mean(dim=1, keepdim=True)
        cov_al = cov.mean(1)
        cov_ep = torch.mean((mu - mu_mean)[..., None] * (mu - mu_mean)[..., None].swapaxes(-1, -2), dim=1)
        mu = mu.mean(dim=1).cpu().numpy()
        cov = (cov_al + cov_ep).cpu().numpy()

        post_mu = contour_samples.mean(axis=(2))
        post_cov = np.zeros((n, contour_samples.shape[1], 21, 2, 2))
        for idx in range(contour_samples.shape[0]):
            for i in range(contour_samples.shape[1]):
                for k in range(contour_samples.shape[3]):

                    post_cov[idx, i, k] = np.cov(contour_samples[idx, i, :, k].reshape(-1, 2).T)

        post_mu_mean = post_mu.mean(axis=1, keepdims=True)
        post_cov_al = post_cov.mean(1)
        post_cov_ep = np.mean((post_mu - post_mu_mean)[..., None] * (post_mu - post_mu_mean)[..., None].swapaxes(-1, -2), axis=1)
        post_cov = post_cov_ep + post_cov_al
        post_mu = post_mu.mean(axis=1)

        pred, pred_samples = self.convert_to_mask(mu, img.shape, contour_samples)
        pred = pred_samples.mean(axis=(1, 2)).squeeze().round().astype(int)


        uncertainty_map = np.array([self.umap_fn(mu[i], cov[i], self.hparams.data_params.labels) for i in range(n)])

        voxelspacing, instants = self.get_voxelspacing_and_instants(batch)

        res = BatchResult(id=batch[Tags.id],
                          labels=self.hparams.data_params.labels,
                          img=img,
                          contour=contour.cpu().numpy(),
                          gt=gt,
                          mu=mu,
                          mode=mu,
                          cov=cov,
                          contour_samples=contour_samples,
                          pred_samples=pred_samples,
                          pred=pred,
                          uncertainty_map=uncertainty_map,
                          instants=instants,
                          voxelspacing=voxelspacing,
                          post_mu=post_mu,
                          post_cov=post_cov
                          )
        return res


    def get_cov_matrix(self, var_x, var_y, covar_xy=0):
        Sigma = torch.zeros((var_x.shape[0], var_x.shape[1], 2, 2), device=self.device)
        Sigma[:, :, 0, 0] = var_x
        Sigma[:, :, 0, 1] = covar_xy
        Sigma[:, :, 1, 0] = covar_xy
        Sigma[:, :, 1, 1] = var_y
        return Sigma

    def plot_image(self, num_images: int, axes_content: Dict[str, np.ndarray], mu, Sigma, mu_gt):
        for i in range(num_images):
            fig, axes = plt.subplots(1, len(axes_content.keys()), squeeze=False)
            plt.suptitle(f'Sample {i}')
            axes = axes.ravel()
            for j, (ax_title, img) in enumerate(axes_content.items()):

                axes[j].imshow(img[i].squeeze())
                axes[j].set_title(ax_title)
                axes[j].scatter(mu[i, :, 0], mu[i, :, 1], s=10, c="r")
                if mu_gt is not None:
                    axes[j].scatter(mu_gt[i, :, 0], mu_gt[i, :, 1], s=10, c="b")
                for k in range(mu.shape[1]):
                    axes[j].annotate(str(k), (mu[i, k, 0], mu[i, k, 1]), c='c')

                if Sigma is not None:
                    for k in range(0, mu.shape[1], 1):
                        confidence_ellipse(mu[i, k, 0], mu[i, k, 1], Sigma[i, k], axes[j])

            if isinstance(self.trainer.logger, TensorBoardLogger):
                self.trainer.logger.experiment.add_figure("{}_{}".format(f'Sample', i), fig, self.current_epoch)
            if isinstance(self.trainer.logger, CometLogger):
                self.trainer.logger.experiment.log_figure("{}_{}".format(f'Sample', i), fig, step=self.current_epoch)

            plt.close()
