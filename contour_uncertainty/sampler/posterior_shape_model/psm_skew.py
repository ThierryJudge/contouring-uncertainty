from functools import partial
import math
from pathlib import Path
from typing import Sequence, List

import numpy as np
from contour_uncertainty.data.ultromics.lv.dataset import LVDataset
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.distributions import MultivariateNormal
import multiprocessing
from tqdm import tqdm
from joblib import Parallel, delayed

from contour_uncertainty.distributions.bivariatenormal import BivariateNormal
from contour_uncertainty.distributions.bivariateskewnormal import BivariateSkewNormal
from contour_uncertainty.sampler.posterior_shape_model.posteriorshapemodel import posterior_shape_model, pca
from contour_uncertainty.sampler.posterior_shape_model.utils import index_to_flat
from contour_uncertainty.sampler.sampler import Sampler
from contour_uncertainty.utils.plotting import confidence_ellipse, crop_axis
from contour_uncertainty.utils.skew_normal import plot_skewed_normals
from vital.data.camus.config import Label
#
# def numerical_grid(start, end, size, device='cpu'):
#     x = torch.linspace(0,255, size, device=device)
#     y = torch.linspace(0,255, size, device=device)
#
#     X, Y = torch.meshgrid(x, y, indexing='ij')
#
#     # Stack the grid points along the last dimension
#     grid_points = torch.stack([X, Y], dim=-1)
#
#     return grid_points
#
#
# def numberical_sampling():
#     pass
#
# def numerical_product_sampling():
#     pass



def numerical_sampling(
        mu1,
        cov1,
        mu2,
        cov2,
        alpha1=None,
        alpha2=None,
        n_samples=1,
        grid_size = 256,
        p1 = None,
):
    # mu = torch.mean(torch.cat([mu1[None], mu2[None]]), dim=0).int()
    # x = torch.linspace(mu[0]-50, mu[0]+50, grid_size, device=mu1.device)
    # y = torch.linspace(mu[1]-50, mu[1]+50, grid_size, device=mu1.device)

    mu = torch.mean(torch.cat([mu1[None], mu2[None]]), dim=0).int()
    x = torch.linspace(0,255, grid_size, device=mu1.device)
    y = torch.linspace(0,255, grid_size, device=mu1.device)

    X, Y = torch.meshgrid(x, y, indexing='ij')

    # Stack the grid points along the last dimension
    grid_points = torch.stack([X, Y], dim=-1)

    # Evaluate the PDFs of the two distributions on the grid
    if p1 is None:
        if alpha1 is not None:
            p1 = BivariateSkewNormal.pdf(grid_points, mu1, cov1, alpha1)
        else:
            p1 = BivariateNormal.pdf(grid_points, mu1, cov1)
    if alpha2 is not None:
        p2 = BivariateSkewNormal.pdf(grid_points, mu2, cov2, alpha2)
    else:
        p2 = torch.exp(MultivariateNormal(mu2, cov2, validate_args=False).log_prob(grid_points))


    # print(p1.sum())
    # print(p2.sum())
    # Compute the product of the two distributions
    p = p1 * p2

    # print(p.sum())

    p /= torch.sum(p)

    # f, ax = plt.subplots(1,1)
    # # plot_skewed_normals(ax, mu1[None].cpu(), cov1[None].cpu(), alpha1[None].cpu(), flip_y=False)
    # confidence_ellipse(mu1[0].cpu(), mu1[1].cpu(), cov1.cpu(), ax, edgecolor='blue')
    # confidence_ellipse(mu2[0].cpu(), mu2[1].cpu(), cov2.cpu(), ax, edgecolor='blue')
    # plt.scatter(mu1[0].detach().cpu(), mu1[1].detach().cpu(), c='r')
    # plt.scatter(mu2[0].detach().cpu(), mu2[1].detach().cpu(), c='b')
    # plt.gca().set_xlim([0, 256])
    # plt.gca().set_ylim([256, 0])
    #
    # # f, (ax0, ax1, ax2, ax3) = plt.subplots(1,4, figsize=(20,10))
    # f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20,10))
    #
    # ax1.scatter(mu1[0].detach().cpu(), mu1[1].detach().cpu(), c='r', s=1)
    # ax2.scatter(mu2[0].detach().cpu(), mu2[1].detach().cpu(), c='b', s=1)
    #
    # ax1.imshow(p1.T.detach().cpu())
    # ax2.imshow(p2.T.detach().cpu())
    # ax3.imshow(p.T.detach().cpu())
    # # ax1.scatter(mu1[0].detach().cpu(), mu1[1].detach().cpu(), c='r', s=1)
    # # ax2.scatter(mu2[0].detach().cpu(), mu2[1].detach().cpu(), c='b', s=1)
    #
    # plot_skewed_normals(ax1, mu1[None].cpu(), cov1[None].cpu(), alpha1[None].cpu(), flip_y=False)
    # # confidence_ellipse(mu1[0].cpu(), mu1[1].cpu(), cov1.cpu(), ax1, edgecolor='blue')
    # confidence_ellipse(mu2[0].cpu(), mu2[1].cpu(), cov2.cpu(), ax2, edgecolor='blue')
    #
    # # ax0.set_xlim([50, 150])
    # # ax0.set_ylim([150, 50])
    # # ax1.set_xlim([50, 150])
    # # ax1.set_ylim([150, 50])
    # # ax2.set_xlim([50, 150])
    # # ax2.set_ylim([150, 50])
    # # ax3.set_xlim([50, 150])
    # # ax3.set_ylim([150, 50])
    #
    # plt.show()

    # Flatten the grid and the p for sampling
    flattened_pdf_torch = p.flatten()
    # print(flattened_pdf_torch)

    # Sample one point from the grid using the computed product PDF
    try:
        sample_index = torch.multinomial(flattened_pdf_torch, n_samples)
        sample_x = X.flatten()[sample_index]
        sample_y = Y.flatten()[sample_index]
    except:
        # f, ax = plt.subplots(1,1)
        # plot_skewed_normals(ax, mu1[None].cpu(), cov1[None].cpu(), alpha1[None].cpu(), flip_y=False)
        # confidence_ellipse(mu2[0].cpu(), mu2[1].cpu(), cov2.cpu(), ax, edgecolor='blue')
        # plt.scatter(mu1[0].detach().cpu(), mu1[1].detach().cpu(), c='r')
        # plt.scatter(mu2[0].detach().cpu(), mu2[1].detach().cpu(), c='b')
        # plt.gca().set_xlim([0, 256])
        # plt.gca().set_ylim([256, 0])
        #
        # f, (ax1, ax2, ax3) = plt.subplots(1,3)
        # ax1.imshow(p1.detach().cpu())
        # ax2.imshow(p2.detach().cpu())
        # ax3.imshow(p.detach().cpu())
        # ax1.scatter(mu1[0].detach().cpu(), mu1[1].detach().cpu(), c='r')
        # ax2.scatter(mu2[0].detach().cpu(), mu2[1].detach().cpu(), c='b')
        # plt.gca().set_xlim([0, 256])
        # plt.gca().set_ylim([256, 0])
        # plt.show()
        sample_x = mu2[0]
        sample_y = mu2[1]



    return torch.cat([sample_x[None],sample_y[None]], dim=0).squeeze()



class SkewPosteriorShapeModelSampler(Sampler):
    """Sampler using a posterior shape model for conditional distribution

        Posterior Shape Models : https://edoc.unibas.ch/30789/1/20140113141209_52d3e629d3417.pdf

    """

    def __init__(self, psm_path: Path, levels: int = 3, skew_indices: List[int] = None):

        data = np.load(str(psm_path), allow_pickle=True).item()
        self.mu, self.Q = torch.tensor(data['mu'], dtype=torch.float), torch.tensor(data['Q'], dtype=torch.float)
        self.mean = torch.tensor(data['scaler_mean'], dtype=torch.float)
        self.scale = torch.tensor(data['scaler_scale'], dtype=torch.float)
        self.X_train = torch.tensor(data['X_train'], dtype=torch.float)

        self.initial_points, self.points_order = self.get_points_order(self.mu.shape[0] // 2, levels=levels)

        self.skew_indices = list(range(self.mu.shape[0] // 2)) if skew_indices is None else skew_indices

        self.grid_size = 256  # Number of points in grid for numerical sampling
        x = torch.linspace(0, 255, self.grid_size)
        y = torch.linspace(0, 255, self.grid_size)
        self.X, self.Y = torch.meshgrid(x, y, indexing='ij')
        self.grid_points = torch.stack([self.X, self.Y], dim=-1)

    def __call__(
            self,
            mu: torch.Tensor,
            cov: torch.Tensor,
            alpha: torch.Tensor,
            n: int=1,
            debug_img=None,
            progress_bar=False,
    ) -> torch.Tensor:

        if mu.ndim == 2:
            mu = mu[None]
            cov = mu[None]
            alpha  = alpha[None] if alpha is not None else None


        results = []
        for idx in range(mu.shape[0]):
            debug_img = debug_img[idx] if debug_img is not None else None
            results.append(self.sample_one_instant(mu[idx], cov[idx], alpha[idx], n, debug_img, progress_bar))

        return torch.stack(results)

    def sample_one_instant(
            self,
            mu: torch.Tensor,
            cov: torch.Tensor,
            alpha: torch.Tensor,
            n: int = 1,
            debug_img=None,
            progress_bar=False,
            pdfs=None,
            use_initial_pdf: bool = False
    ):
        self.X_train = self.X_train.to(mu.device)
        self.mean = self.mean.to(mu.device)
        self.scale = self.scale.to(mu.device)
        self.grid_points = self.grid_points.to(mu.device)
        self.X = self.X.to(mu.device)
        self.Y = self.Y.to(mu.device)

        self.mu, self.Q = pca(self.X_train, self.transform(mu).reshape(-1, 1))



        alpha = torch.clone(alpha) * torch.tensor([1., -1.], device=mu.device)

        if pdfs is None:
            pdfs = []
            for i in range(mu.shape[0]):
                pdfs.append(BivariateSkewNormal.pdf(self.grid_points, mu[i], cov[i], alpha[i])[None])
            pdfs = torch.cat(pdfs)

        p_bar = tqdm(range(n)) if progress_bar else range(n)

        res = [self.sample_contour(mu, cov, alpha, pdfs=pdfs, use_initial_pdf=use_initial_pdf, debug_img=debug_img) for _ in p_bar]

        return torch.stack(res)


    def sample_contour(
            self,
            mu_p: torch.tensor,
            cov_p: torch.tensor,
            alpha_p: torch.Tensor = None,
            complete_shape: bool = True,
            debug_img: np.ndarray = None,
            pdfs: torch.tensor = None,
            use_initial_pdf: bool = False,
    ):
        """

        Args:
            mu_p:
            cov_p:
            alpha_p
            complete_shape:
            debug_img:

        Returns:

        """
        # sigmas = [.1, .1, .1, .1]
        sigmas = [1, 1, 1, 1]
        device = mu_p.device

        sampled_contour = torch.zeros(mu_p.shape, dtype=torch.float, device=device, requires_grad=debug_img is None)
        sampled_indices = []

        # Sample initial po ints
        sampled_indices.extend(self.initial_points)
        if use_initial_pdf:
            s = self.numerical_sample(pdfs, sample_indices=self.initial_points)
        else:
            s = self.sample_points(mu_p, cov_p, alpha_p, sample_indices=self.initial_points)

        mask = torch.zeros_like(sampled_contour, device=device)
        mask[self.initial_points] = 1

        sampled_contour = sampled_contour + s * mask

        if debug_img is not None:
            nb_subplots = len(self.points_order) + 2 if complete_shape else len(self.points_order) + 1
            f, axes = plt.subplots(1, nb_subplots, figsize=(24, 10))
            axes = axes.ravel()
            axes[0].set_title('Initial sampling')
            axes[0].imshow(debug_img.squeeze(), cmap="gray")
            axes[0].scatter(mu_p[:, 0].cpu(), mu_p[:, 1].cpu(), s=10, c='r', label='Initial shape')
            if alpha_p is not None:
                plot_skewed_normals(axes[0], mu_p.cpu(), cov_p.cpu(), alpha_p.cpu(), flip_y=False)
            else:
                for index in range(0, mu_p.shape[0], 1):
                    confidence_ellipse(mu_p[index, 0].cpu(), mu_p[index, 1].cpu(), cov_p[index].cpu(), axes[0])
            axes[0].scatter(s[:, 0].cpu(), s[:, 1].cpu(), s=10, c='g', label='Sampled points')

            axes[0].legend()
            axes[0].set_axis_off()
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)

        for i, points in enumerate(self.points_order):
            sampled_indices.sort()
            if len(sampled_indices) == mu_p.shape[0]:
                break

            # Posterior model
            mu_c, cov_c = self.compute_psm(sampled_contour, sampled_indices, sigmas[i], contour_shape=mu_p.shape)
            # Combine priors and Sample points from combined prior in one step using rejection sampling.
            s = torch.zeros((mu_p.shape[0], mu_p.shape[1]), device=device)
            rejects = np.zeros((len(mu_p)))
            for point in points:
                if point in self.skew_indices:
                    s[point] = numerical_sampling(
                        mu_p[point],
                        cov_p[point],
                        mu_c[point],
                        cov_c[point],
                        alpha1=alpha_p[point],
                        p1 = pdfs[point]
                    )
                else:
                    # Sample points from combined prior.
                    mu_f, cov_f = self.merge_gaussian_priors(mu_p, cov_p, mu_c, cov_c)
                    s[point] = self.sample_points(mu_f, cov_f, sample_indices=[point])[point]
                    # print(f'Level {i}, point {point} in {points}. Merge gaussian')


            if debug_img is not None:
                axes[i + 1].set_title(f'Level {i + 1} sampling')
                axes[i + 1].imshow(debug_img.squeeze(), cmap="gray")
                axes[i + 1].scatter(mu_p[:, 0].cpu(), mu_p[:, 1].cpu(), s=10, c='r', label='Initial shape')
                axes[i + 1].scatter(mu_c[:, 0].cpu(), mu_c[:, 1].cpu(), s=10, c='b', label='Posterior prediction')
                axes[i + 1].scatter(sampled_contour[sampled_indices, 0].cpu(),
                                    sampled_contour[sampled_indices, 1].cpu(), s=10, c='g', label='Previous points')
                for p in points:
                #     if p in self.skew_indices:
                #         confidence_ellipse(
                #             mu_env[0].cpu(), mu_env[1].cpu(), cov_env.cpu(), axes[i + 1], n_std=2, edgecolor='magenta'
                #         )
                #         axes[i + 1].scatter(mu_env[0].cpu(), mu_env[1].cpu(), s=10, c='m', label='Envelope')
                #
                #     else:
                #         axes[i + 1].scatter(
                #             mu_f[p, 0].cpu(), mu_f[p, 1].cpu(), s=10, c='g', label='Combined model'
                #         )
                #         confidence_ellipse(
                #             mu_f[p, 0].cpu(), mu_f[p, 1].cpu(), cov_f[p].cpu(), axes[i + 1], edgecolor='green'
                #         )


                    plot_skewed_normals(
                        axes[i + 1], mu_p[p:p+1].cpu(), cov_p[p:p+1].cpu(), alpha_p[p:p+1].cpu(), flip_y=False
                    )
                    confidence_ellipse(mu_p[p, 0].cpu(), mu_p[p, 1].cpu(), cov_p[p].cpu(), axes[i + 1], n_std=2,
                                       edgecolor='red', linestyle='--')
                    confidence_ellipse(mu_c[p, 0].cpu(), mu_c[p, 1].cpu(), cov_c[p].cpu(), axes[i + 1], n_std=2,
                                       edgecolor='blue')
                    axes[i + 1].text(mu_p[p, 0].cpu()+20, mu_p[p, 1].cpu(), f"{rejects[p]}", color='red')

                axes[i + 1].scatter(s[:, 0].cpu(), s[:, 1].cpu(), s=25, c='y', marker="*", label='Sampled')
                # axes[i + 1].legend()
                axes[i + 1].set_axis_off()
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                plt.margins(0, 0)



            # sampled_contour[points] = s
            mask = torch.zeros_like(sampled_contour, device=device)
            mask[points] = 1
            sampled_contour += s * mask

            sampled_indices.extend(points)

        sampled_indices.sort()

        #  Extract other points from posterior shape model
        if complete_shape and len(sampled_indices) != mu_p.shape[0]:
            # Posterior model
            s_g = self.transform(sampled_contour).reshape(-1, 1)
            # TODO Sigma = 0 because we do not want any extra noise.
            mu_c, cov_c = posterior_shape_model(s_g, index_to_flat(sampled_indices), self.mu, self.Q, sigma2=0.001)
            # Rescale mu_c
            mu_c = self.inverse_transform(mu_c.squeeze()).reshape(mu_p.shape)

            # Fill remainder of sampled_contour with mu_c
            mask = torch.ones(len(sampled_contour), dtype=torch.bool)
            mask[sampled_indices] = 0
            sampled_contour[mask] = mu_c[mask]

            if debug_img is not None:
                axes[-1].set_title(f'Filling last points')
                axes[-1].imshow(debug_img.squeeze(), cmap="gray")
                axes[-1].scatter(mu_p[:, 0].cpu(), mu_p[:, 1].cpu(), s=10, c='r', label='Initial shape')
                axes[-1].scatter(mu_c[mask, 0].cpu(), mu_c[mask, 1].cpu(), s=10, c='c', marker="*",
                                 label='Selected posterior predictions')
                axes[-1].scatter(mu_c[~mask, 0].cpu(), mu_c[~mask, 1].cpu(), s=10, c='b', label='Posterior prediction')
                axes[-1].scatter(sampled_contour[sampled_indices, 0].cpu(),
                                 sampled_contour[sampled_indices, 1].cpu(), s=10, c='g', label='Previous points')
        else:
            sampled_contour = sampled_contour[sampled_indices]
        if debug_img is not None:
            axes[-1].legend()
            plt.show()
        return sampled_contour

    @staticmethod
    def sample_points(
            mu: torch.tensor,
            cov: torch.tensor,
            alpha: torch.Tensor = None,
            n: int = 1,
            sample_indices: Sequence = None
    ):
        """ Sample each point distribution independently and combine in array.

        Args:
            mu: Point distribution means (K, 2)
            cov: Point distribution covariance matrices (K, 2, 2)
            alpha: Point distribution skewness vector (K, 2)
            n: Number of contours to sample
            sample_indices: Indices of points to sample

        Returns:
            Sampled contours (N, len(sampled_points), 2) or (N, K, 2)
        """
        sample_indices = sample_indices or range(mu.shape[0])
        if n > 1:
            points = torch.zeros((n, mu.shape[0], mu.shape[1]), device=mu.device)
        else:
            points = torch.zeros((mu.shape[0], mu.shape[1]), device=mu.device)
        for j in sample_indices:
            if alpha is None:
                x = MultivariateNormal(mu[j].squeeze(), cov[j], validate_args=False).rsample((n,)).T.squeeze()
            else:
                x = BivariateSkewNormal.rvs_fast(mu[j].squeeze(), cov[j], alpha[j].squeeze(), size=(n,))
            if n > 1:
                points[:, j] = x
            else:
                points[j] = x

        return points

    def numerical_sample(self, pdfs: torch.Tensor, n: int = 1, sample_indices: Sequence = None):
        sample_indices = sample_indices or range(pdfs.shape[0])
        if n > 1:
            points = torch.zeros((n, pdfs.shape[0], 2), device=pdfs.device)
        else:
            points = torch.zeros((pdfs.shape[0], 2), device=pdfs.device)
        for j in sample_indices:
            flattened_pdf_torch = pdfs[j].flatten()
            sample_index = torch.multinomial(flattened_pdf_torch, n)
            sample_x = self.X.flatten()[sample_index]
            sample_y = self.Y.flatten()[sample_index]
            if n > 1:
                points[:, j] = torch.cat([sample_x[None], sample_y[None]], dim=0).squeeze()
            else:
                points[j] = torch.cat([sample_x[None], sample_y[None]], dim=0).squeeze()

        return points


    def compute_psm(self, sampled_contour, sampled_indices, sigma, contour_shape=(21, 2)):
        """

        Args:
            sampled_contour:
            sampled_indices:
            sigma:
            contour_shape:

        Returns:

        """
        # Posterior model
        s_g = self.transform(sampled_contour).reshape(-1, 1)
        mu_c, cov_c = posterior_shape_model(s_g, index_to_flat(sampled_indices), self.mu, self.Q, sigma2=sigma)

        # Rescale mu_c and cov_c
        mu_c = self.inverse_transform(mu_c.squeeze()).reshape(contour_shape)
        cov_c *= self.scale
        # Extract 2x2 covariance matrices along diagonal of pxp matrix.
        cov_c = torch.stack([cov_c[2 * i:(2 * i) + 2, 2 * i:(2 * i) + 2] for i in range(contour_shape[0])])

        return mu_c, cov_c


    def transform(self, s):
        shape = s.shape
        s = (s.reshape(1, -1) - self.mean) / self.scale
        return s.reshape(shape)

    def inverse_transform(self, s, ):
        shape = s.shape
        s = (s.reshape(1, -1) * self.scale) + self.mean
        return s.reshape(shape)

