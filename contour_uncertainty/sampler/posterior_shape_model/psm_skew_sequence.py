import random
from logging import debug
from pathlib import Path
from typing import List

import numpy as np
import torch

from contour_uncertainty.distributions.bivariatenormal import BivariateNormal
from contour_uncertainty.distributions.bivariateskewnormal import BivariateSkewNormal
from contour_uncertainty.sampler.posterior_shape_model.posteriorshapemodel import posterior_shape_model, pca
from contour_uncertainty.sampler.posterior_shape_model.psm_skew import SkewPosteriorShapeModelSampler
from contour_uncertainty.sampler.posterior_shape_model.utils import index_to_flat

from matplotlib import pyplot as plt

from contour_uncertainty.utils.plotting import confidence_ellipse
from contour_uncertainty.utils.skew_normal import plot_skewed_normals


class SequenceSkewPSMSampler(SkewPosteriorShapeModelSampler):

    def __init__(self, psm_path: Path, sequence_psm_path: Path, levels: int = 3, skew_indices: List[int] = None):
        super().__init__(psm_path, levels, skew_indices)

        data = np.load(str(sequence_psm_path), allow_pickle=True).item()
        self.seq_mu, self.seq_Q = torch.tensor(data['mu'], dtype=torch.float), torch.tensor(data['Q'],
                                                                                            dtype=torch.float)
        self.seq_mean = torch.tensor(data['scaler_mean'], dtype=torch.float)
        self.seq_scale = torch.tensor(data['scaler_scale'], dtype=torch.float)
        self.seq_X_train = torch.tensor(data['X_train'], dtype=torch.float)
        self.seq_X_val = torch.tensor(data['X_val'], dtype=torch.float)

    def __call__(
            self,
            mu: torch.Tensor,
            cov: torch.Tensor,
            alpha: torch.Tensor,
            n: int = 1,
            debug_img=None,
            progress_bar=False,
    ) -> torch.Tensor:
        samples = []
        for i in range(n):
            first_instant = random.randint(0, 1)
            s = self.sample_two_contours(mu, cov, alpha, first_instant=first_instant, debug_img=debug_img)
            samples.append(s[None])
        return torch.cat(samples).permute(1,0,2,3)

    def sample_two_contours(
            self,
            mu: torch.Tensor,
            cov: torch.Tensor,
            alpha: torch.Tensor,
            first_sample=None,
            first_instant=0,
            debug_img=None
    ):
        second_instant = abs(1 - first_instant)

        self.seq_X_train = self.seq_X_train.to(mu.device)
        self.seq_mean = self.seq_mean.to(mu.device)
        self.seq_scale = self.seq_scale.to(mu.device)
        self.seq_mu = self.seq_mu.to(mu.device)
        self.seq_Q = self.seq_Q.to(mu.device)
        self.seq_mu, self.seq_Q = pca(self.seq_X_train, self.sequence_transform(mu).reshape(-1, 1))
        s = torch.zeros_like(mu)

        d_img = debug_img[first_instant] if debug_img is not None else None
        s[first_instant] = self.sample_one_instant(mu[first_instant], cov[first_instant], alpha[first_instant], 1, d_img, False)

        s_g = torch.zeros_like(mu, device=mu.device)
        s_g[first_instant] = s[first_instant].clone()
        s_g = self.sequence_transform(s_g).reshape(-1, 1)

        sampled_indices = list(range(21)) if first_instant == 0 else list(range(21, 42))

        mu_c, cov_c = posterior_shape_model(s_g, index_to_flat(sampled_indices), self.seq_mu, self.seq_Q, sigma2=1)
        mu_c = self.sequence_inverse_transform(mu_c.squeeze()).reshape((42, 2))
        cov_c *= self.seq_scale
        cov_c = torch.stack([cov_c[2 * i:(2 * i) + 2, 2 * i:(2 * i) + 2] for i in range(42)])

        mu_c = mu_c.reshape(2, 21, 2)
        cov_c = cov_c.reshape(2, 21, 2, 2)

        skew_pdfs = []
        psm_pdfs = []
        pdfs = []
        for i in range(mu.shape[1]):
            skew_p = BivariateSkewNormal.pdf(self.grid_points, mu[second_instant, i], cov[second_instant, i], alpha[second_instant, i])[None]
            psm_p = BivariateNormal.pdf(self.grid_points, mu_c[second_instant, i], cov_c[second_instant, i])[None]
            skew_pdfs.append(skew_p)
            psm_pdfs.append(psm_p)
            p = skew_p * psm_p
            pdfs.append(p / p.sum())

        skew_pdfs = torch.cat(skew_pdfs)
        psm_pdfs = torch.cat(psm_pdfs)
        pdfs = torch.cat(pdfs)

        d_img = debug_img[second_instant] if debug_img is not None else None
        s[second_instant] = self.sample_one_instant(mu[second_instant], cov[second_instant], alpha[second_instant], 1,
                                                   d_img, pdfs=pdfs, use_initial_pdf=True, progress_bar=False)

        if debug_img is not None:
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))

            ax1.set_title(f'First instant {first_instant}')

            ax1.imshow(debug_img[0].squeeze(), cmap='gray')
            ax2.imshow(debug_img[1].squeeze(), cmap='gray')

            ax1.scatter(mu.detach().cpu()[0, :, 0], mu.detach().cpu()[0, :, 1], c='r', s=5, label=r'$mu_p$')
            ax2.scatter(mu.detach().cpu()[1, :, 0], mu.detach().cpu()[1, :, 1], c='r', s=5, label=r'$mu_p$')

            ax1.scatter(mu_c.detach().cpu()[0, :, 0], mu_c.detach().cpu()[0, :, 1], c='b', s=5, label=r'$mu_c$')
            ax2.scatter(mu_c.detach().cpu()[1, :, 0], mu_c.detach().cpu()[1, :, 1], c='b', s=5, label=r'$mu_c$')

            ax1.scatter(s.detach().cpu()[0, :, 0], s.detach().cpu()[0, :, 1], c='m', s=5, label=r'$s$')
            ax2.scatter(s.detach().cpu()[1, :, 0], s.detach().cpu()[1, :, 1], c='m', s=5, label=r'$s$')

            for i in range(0, 21):
                plot_skewed_normals(
                    ax1, mu[0, i:i + 1].cpu(), cov[0, i:i + 1].cpu(), alpha[0, i:i + 1].cpu(), flip_y=True
                )
                plot_skewed_normals(
                    ax2, mu[1, i:i + 1].cpu(), cov[1, i:i + 1].cpu(), alpha[1, i:i + 1].cpu(), flip_y=True
                )

                # confidence_ellipse(mu[0, i, 0], mu[0, i, 1], cov[0, i], ax1, n_std=2, linewidth=1, edgecolor='red')
                # confidence_ellipse(mu[1, i, 0], mu[1, i, 1], cov[1, i], ax2, n_std=2, linewidth=1, edgecolor='red')

                confidence_ellipse(mu_c[0, i, 0].cpu(), mu_c[0, i, 1].cpu(), cov_c[0, i].cpu(), ax1, n_std=2, linewidth=1,
                                   edgecolor='blue')
                confidence_ellipse(
                    mu_c[1, i, 0].cpu(), mu_c[1, i, 1].cpu(), cov_c[1, i].cpu(), ax2, n_std=2, linewidth=1,
                                   edgecolor='blue')

                # confidence_ellipse(mu_f[0, i, 0], mu_f[0, i, 1], cov_f[0, i], ax1, n_std=2, linewidth=1,
                #                    edgecolor='green')
                # confidence_ellipse(mu_f[1, i, 0], mu_f[1, i, 1], cov_f[1, i], ax2, n_std=2, linewidth=1,
                #                    edgecolor='green')

            ax1.legend()
            ax2.legend()

            f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
            ax1.imshow(skew_pdfs.sum(0).cpu().T)
            ax2.imshow(psm_pdfs.sum(0).cpu().T)
            ax3.imshow(pdfs.sum(0).cpu().T)
            plt.show()

        return s

    def sequence_transform(self, s):
        shape = s.shape
        # print(s.shape)
        # print(self.seq_mean.shape)
        # print(self.seq_scale.shape)
        s = (s.reshape(1, -1) - self.seq_mean) / self.seq_scale
        return s.reshape(shape)

    def sequence_inverse_transform(self, s, ):
        shape = s.shape
        s = (s.reshape(1, -1) * self.seq_scale) + self.seq_mean
        return s.reshape(shape)
