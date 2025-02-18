from functools import partial
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from contour_uncertainty.data.ultromics.lv.dataset import LVDataset
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.distributions import MultivariateNormal
import multiprocessing

from contour_uncertainty.distributions.bivariateskewnormal import BivariateSkewNormal
from contour_uncertainty.sampler.posterior_shape_model.posteriorshapemodel import posterior_shape_model, pca
from contour_uncertainty.sampler.posterior_shape_model.utils import index_to_flat
from contour_uncertainty.sampler.sampler import Sampler
from contour_uncertainty.utils.plotting import confidence_ellipse, crop_axis
from contour_uncertainty.utils.skew_normal import plot_skewed_normals
from vital.data.camus.config import Label


class PosteriorShapeModelSampler(Sampler):
    """Sampler using a posterior shape model for conditional distribution

        Posterior Shape Models : https://edoc.unibas.ch/30789/1/20140113141209_52d3e629d3417.pdf

    """

    def __init__(self, psm_path: Path, levels: int = 3):

        data = np.load(str(psm_path), allow_pickle=True).item()
        self.mu, self.Q = torch.tensor(data['mu'], dtype=torch.float), torch.tensor(data['Q'], dtype=torch.float)
        self.mean = torch.tensor(data['scaler_mean'], dtype=torch.float)
        self.scale = torch.tensor(data['scaler_scale'], dtype=torch.float)

        self.X_train = torch.tensor(data['X_train'], dtype=torch.float)
        self.X_val = torch.tensor(data['X_val'], dtype=torch.float)

        self.initial_points, self.points_order = self.get_points_order(self.mu.shape[0] // 2, levels=levels)

    @staticmethod
    def get_points_order(nb_points: int = 21, nb_initial_points: int = 3, levels: int = None):
        """Get point order by repeatedly splitting points in half.

        Args:
            nb_points: Number of points in contour
            nb_initial_points: number of initial points to sample freely.
            levels: Number of levels to split. If None, split until all points are included.

        Returns:
            List of list of point indices.
        """
        initial_points = np.round(np.linspace(0, nb_points - 1, nb_initial_points)).astype(int).tolist()
        levels = levels or int(math.log(nb_points, 2))
        all_points, point_order = [], []
        all_points.extend(initial_points)
        for i in range(levels):
            level_points = []
            for j in range(len(all_points) - 1):
                if all_points[j] + 1 != all_points[j + 1]:
                    point = (all_points[j] + all_points[j + 1]) / 2
                    point = math.ceil(point) if point > nb_points / 2 else math.floor(point)  # Round towards the base.
                    level_points.append(int(point))
            if len(level_points) == 0:
                break
            all_points.extend(level_points)
            all_points.sort()
            point_order.append(level_points)

        return initial_points, point_order

    def __call__(
            self,
            mu: torch.Tensor,
            cov: torch.Tensor,
            alpha: torch.Tensor=None,
            n: int=1,
            debug_img=None
    ) -> torch.Tensor:
        self.X_train = self.X_train.to(mu.device)
        self.mean = self.mean.to(mu.device)
        self.scale = self.scale.to(mu.device)

        # print(mu.shape)
        # print(cov.shape)
        # print(alpha.shape)
        # print(self.mean.shape)
        # print(self.scale.shape)

        self.mu, self.Q = pca(self.X_train, self.transform(mu).reshape(-1, 1))
        # return torch.stack([self.sample_endo_epi_contour(mu, cov, debug_img=debug_img) for _ in range(n)])
        return torch.stack([self.sample_endo_contour(mu, cov, alpha, debug_img=debug_img) for _ in range(n)])

    def pca_params(self, mu: torch.Tensor):
        mu, Q, cov, D, U = pca(self.X_train, self.transform(mu).reshape(-1, 1), return_all=True)
        return mu, Q, cov, D, U

    def sample_endo_epi_contour(
            self,
            mu_p: torch.tensor,
            cov_p: torch.tensor,
            complete_shape: bool = True,
            debug_img: np.ndarray = None):

        lv_indices = np.arange(21)
        myo_indices = np.arange(21, 42)

        samples = []

        sampled_contour = torch.zeros(mu_p.shape, dtype=torch.float, device=mu_p.device)

        lv_s = self.sample_points(mu_p, cov_p,
                                  sample_indices=list(lv_indices[self.initial_points]))
        myo_s = self.sample_points(mu_p, cov_p,
                                   sample_indices=list(myo_indices[self.initial_points]))

        sampled_indices = []
        sampled_indices.extend(self.initial_points)

        lv_mask = torch.zeros_like(sampled_contour, device=sampled_contour.device)
        lv_mask[lv_indices[self.initial_points]] = 1
        myo_mask = torch.zeros_like(sampled_contour, device=sampled_contour.device)
        myo_mask[myo_indices[self.initial_points]] = 1
        sampled_contour = sampled_contour + lv_s * lv_mask
        sampled_contour = sampled_contour + myo_s * myo_mask

        for i, points in enumerate(self.points_order):
            sampled_indices.sort()
            if len(sampled_indices) == mu_p.shape[0]:
                break

            s_g = self.transform(sampled_contour).reshape(-1, 1)
            indices = list(np.array([index_to_flat(lv_indices[sampled_indices]),
                                     index_to_flat(myo_indices[sampled_indices])]).flatten())
            mu_c, cov_c = posterior_shape_model(s_g.float(), indices, self.mu, self.Q, sigma2=1)

            mu_c = self.inverse_transform(mu_c.squeeze()).reshape(mu_p.shape)
            cov_c *= self.scale
            # Extract 2x2 covariance matrices along diagonal of pxp matrix.
            cov_c = torch.stack([cov_c[2 * i:(2 * i) + 2, 2 * i:(2 * i) + 2] for i in range(len(mu_p))])

            mu_f, cov_f = self.merge_priors(mu_p, cov_p, mu_c.float(), cov_c.float())

            lv_s = self.sample_points(mu_f.squeeze(), cov_f,
                                      sample_indices=list(lv_indices[points]))
            myo_s = self.sample_points(mu_f.squeeze(), cov_f,
                                       sample_indices=list(myo_indices[points]))

            lv_mask = torch.zeros_like(sampled_contour, device=sampled_contour.device)
            lv_mask[lv_indices[points]] = 1
            myo_mask = torch.zeros_like(sampled_contour, device=sampled_contour.device)
            myo_mask[myo_indices[points]] = 1
            sampled_contour = sampled_contour + lv_s * lv_mask
            sampled_contour = sampled_contour + myo_s * myo_mask

            sampled_indices.extend(points)

        sampled_indices.sort()

        #  Extract other points from posterior shape model
        if complete_shape and len(sampled_indices) != mu_p.shape[0]:
            s_g = self.transform(sampled_contour).reshape(-1, 1)
            # TODO Sigma = 0 because we do not want any extra noise.
            indices = list(np.array(
                [index_to_flat(lv_indices[sampled_indices]), index_to_flat(myo_indices[sampled_indices])]).flatten())
            mu_c, cov_c = posterior_shape_model(s_g.float(), indices, self.mu, self.Q, sigma2=0.001)
            # Rescale mu_c
            mu_c = self.inverse_transform(mu_c.squeeze()).reshape(mu_p.shape).float()

            # Fill remainder of sampled_contour with mu_c
            lv_mask = torch.ones(len(sampled_contour), dtype=torch.bool)
            lv_mask[lv_indices[sampled_indices]] = 0
            myo_mask = torch.ones(len(sampled_contour), dtype=torch.bool)
            myo_mask[myo_indices[sampled_indices]] = 1
            sampled_contour[lv_mask] = mu_c[lv_mask]
            sampled_contour[myo_mask] = mu_c[myo_mask]

        # lv_contour = sampled_contour[lv_indices[sampled_indices]]
        # myo_contour = sampled_contour[myo_indices[sampled_indices]]

        return sampled_contour


    def compute_psm(self, sampled_contour, sampled_indices, sigma, contour_shape=(21, 2)):
        # Posterior model
        s_g = self.transform(sampled_contour).reshape(-1, 1)
        mu_c, cov_c = posterior_shape_model(s_g, index_to_flat(sampled_indices), self.mu, self.Q, sigma2=sigma)

        # Rescale mu_c and cov_c
        mu_c = self.inverse_transform(mu_c.squeeze()).reshape(contour_shape)
        cov_c *= self.scale
        # Extract 2x2 covariance matrices along diagonal of pxp matrix.
        cov_c = torch.stack([cov_c[2 * i:(2 * i) + 2, 2 * i:(2 * i) + 2] for i in range(contour_shape[0])])

        return mu_c, cov_c


    def sample_endo_contour(
            self,
            mu_p: torch.tensor,
            cov_p: torch.tensor,
            alpha_p: torch.Tensor = None,
            complete_shape: bool = True,
            debug_img: np.ndarray = None):
        """

        Args:
            mu_p:
            cov_p:
            complete_shape:
            debug_img:

        Returns:

        # TODO initialize posterior shape model with mu_p points (or just 3).
        # TODO checklikelihood of sampled points using PCA.
        # TODO use average likelihood of all samples as uncertainty metric
        # TODO compare likelihood of mu_p vs likelihood of samples
        # TODO reduce sigma (slack) term during loop


        """
        # sigmas = [.1, .1, .1, .1]
        sigmas = [1, 1, 1, 1]
        ps = [0.5, 0.25, 0.25]
        device = mu_p.device

        sampled_contour = torch.zeros(mu_p.shape, dtype=torch.float, device=device, requires_grad=debug_img is None)
        sampled_indices = []

        # Sample initial points
        sampled_indices.extend(self.initial_points)
        if alpha_p is not None:
            ap = torch.clone(alpha_p)
            ap[..., 1] = -ap[..., 1]
        else:
            ap = None
        s = self.sample_points(mu_p, cov_p, ap, sample_indices=self.initial_points)

        mask = torch.zeros_like(sampled_contour, device=device)
        mask[self.initial_points] = 1

        sampled_contour = sampled_contour + s * mask

        if debug_img is not None:
            nb_subplots = len(self.points_order) + 2 if complete_shape else len(self.points_order) + 1
            f, axes = plt.subplots(1, nb_subplots, figsize=(24, 10))
            axes = axes.ravel()
            axes[0].set_title('Initial sampling')
            axes[0].imshow(debug_img.squeeze(), cmap="gray")
            axes[0].scatter(mu_p[:, 0], mu_p[:, 1], s=10, c='r', label='Initial shape')
            if alpha_p is not None:
                plot_skewed_normals(axes[0], mu_p, cov_p, alpha_p)
            else:
                for index in range(0, mu_p.shape[0], 1):
                    confidence_ellipse(mu_p[index, 0], mu_p[index, 1], cov_p[index], axes[0], n_std=2, edgecolor='red')
            axes[0].scatter(s[:, 0], s[:, 1], s=10, c='g', label='Sampled points')

            axes[0].legend()
            axes[0].set_axis_off()
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)

        for i, points in enumerate(self.points_order):
            sampled_indices.sort()
            if len(sampled_indices) == mu_p.shape[0]:
                break

            # Posterior model
            s_g = self.transform(sampled_contour).reshape(-1, 1)
            mu_c, cov_c = posterior_shape_model(s_g, index_to_flat(sampled_indices), self.mu, self.Q, sigma2=sigmas[i])

            # Rescale mu_c and cov_c
            mu_c = self.inverse_transform(mu_c.squeeze()).reshape(mu_p.shape)
            cov_c *= self.scale
            # Extract 2x2 covariance matrices along diagonal of pxp matrix.
            cov_c = torch.stack([cov_c[2 * i:(2 * i) + 2, 2 * i:(2 * i) + 2] for i in range(len(mu_p))])

            # Combine priors
            mu_f, cov_f = self.merge_priors(mu_p, cov_p, mu_c, cov_c, p=ps[i])

            # Sample points from combined prior.
            s = self.sample_points(mu_f.squeeze(), cov_f, sample_indices=points)

            if debug_img is not None:
                axes[i + 1].set_title(f'Level {i + 1} sampling')
                axes[i + 1].imshow(debug_img.squeeze(), cmap="gray")
                axes[i + 1].scatter(mu_p[:, 0], mu_p[:, 1], s=10, c='r', label='Initial shape')
                axes[i + 1].scatter(mu_c[:, 0], mu_c[:, 1], s=10, c='b', label='Posterior prediction')
                axes[i + 1].scatter(mu_f[points, 0], mu_f[points, 1], s=10, c='y', label='Combined model')
                axes[i + 1].scatter(sampled_contour[sampled_indices, 0],
                                    sampled_contour[sampled_indices, 1], s=10, c='g', label='Previous points')
                for point in points:
                    confidence_ellipse(mu_p[point, 0], mu_p[point, 1], cov_p[point], axes[i + 1], n_std=2,
                                       edgecolor='red')
                    confidence_ellipse(mu_c[point, 0], mu_c[point, 1], cov_c[point], axes[i + 1], n_std=2,
                                       edgecolor='blue')
                    confidence_ellipse(mu_f[point, 0], mu_f[point, 1], cov_f[point], axes[i + 1], n_std=2,
                                       edgecolor='green')
                axes[i + 1].scatter(s[:, 0], s[:, 1], s=10, c='y', marker="*", label='Sampled')
                # axes[i + 1].legend()
                axes[i + 1].set_axis_off()
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                plt.margins(0, 0)

                if i == 0:
                    f2, ax = plt.subplots(1, 1, figsize=(6,6))
                    x_plot_min, x_plot_max, y_plot_min, y_plot_max = crop_axis(mu_p, 40)
                    ax.set_xlim(x_plot_min, x_plot_max)
                    ax.set_ylim(y_plot_max, y_plot_min)


                    # ax.set_title(f'Level {i + 1} sampling')
                    ax.imshow(debug_img.squeeze(), cmap="gray")
                    ax.scatter(mu_p[:, 0], mu_p[:, 1], s=10, c='r', label=r'$\hat{\mu}, \hat{\Sigma}$')
                    ax.scatter(mu_c[:, 0], mu_c[:, 1], s=10, c='b', label=r'$\mu_c, \Sigma_c$')
                    ax.scatter(mu_f[points, 0], mu_f[points, 1], s=10, c='m', label=r'$\mu_f, \Sigma_f$')

                    # ax.scatter(mu_p[:, 0], mu_p[:, 1], s=10, c='r', label='Initial shape')
                    # ax.scatter(mu_c[:, 0], mu_c[:, 1], s=10, c='b', label='Posterior prediction')
                    # ax.scatter(mu_f[points, 0], mu_f[points, 1], s=10, c='y', label='Combined model')

                    print(sampled_contour[sampled_indices])
                    print(sampled_contour)
                    print(sampled_indices)
                    ax.scatter(sampled_contour[sampled_indices, 0], sampled_contour[sampled_indices, 1], s=100, c='g', label=r'$s^{({b}_1, {apex}, {b}_2)}$')

                    for point in sampled_indices:
                        confidence_ellipse(mu_p[point, 0], mu_p[point, 1], cov_p[point], ax, n_std=2,
                                           edgecolor='red')

                    for point in points:
                        confidence_ellipse(mu_p[point, 0], mu_p[point, 1], cov_p[point], ax, n_std=2,
                                           edgecolor='red')
                        confidence_ellipse(mu_c[point, 0], mu_c[point, 1], cov_c[point], ax, n_std=2,
                                           edgecolor='blue')
                        confidence_ellipse(mu_f[point, 0], mu_f[point, 1], cov_f[point], ax, n_std=2,
                                           edgecolor='magenta')
                    ax.scatter(s[:, 0], s[:, 1], s=100, c='g', marker="*", label=r'$s^{(level 2)}$')
                    ax.legend(fontsize=15)
                    ax.set_axis_off()
                    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                    plt.margins(0, 0)
                    plt.savefig(f"Sampling_level{i}.png",  dpi=300, bbox_inches='tight', pad_inches = 0)
                    plt.show()
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
                axes[-1].scatter(mu_p[:, 0], mu_p[:, 1], s=10, c='r', label='Initial shape')
                axes[-1].scatter(mu_c[mask, 0], mu_c[mask, 1], s=10, c='c', marker="*",
                                 label='Selected posterior predictions')
                axes[-1].scatter(mu_c[~mask, 0], mu_c[~mask, 1], s=10, c='b', label='Posterior prediction')
                axes[-1].scatter(sampled_contour[sampled_indices, 0],
                                 sampled_contour[sampled_indices, 1], s=10, c='g', label='Previous points')
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
            points = torch.zeros((n, mu.shape[0], mu.shape[1]))
        else:
            points = torch.zeros((mu.shape[0], mu.shape[1]))
        for j in sample_indices:
            if alpha is None:
                x = MultivariateNormal(mu[j].squeeze(), cov[j], validate_args=False).rsample((n,)).T.squeeze()
            else:
                x = torch.tensor(BivariateSkewNormal.rvs_fast(mu[j].squeeze(), cov[j], alpha[j].squeeze(), size=(n,)))
            if n > 1:
                points[:, j] = x
            else:
                points[j] = x

        return points

    @staticmethod
    def merge_priors(mu1, cov1, mu2, cov2, p: float = 0.5):
        """ Parallel merge two multivariate gaussian distributions.
        Args:
            mu1: means of the first distribution (N,2)
            cov1: covariance matrices of the first distribution (N, 2,2)
            mu2: means of the second distribution (N,2)
            cov2: covariance matrices of the second distribution (N, 2,2)
            p:

        Returns:

        """
        sigma_f = cov1 @ torch.inverse(cov1 + cov2) @ cov2
        mu_f = cov1 @ torch.inverse(cov1 + cov2) @ mu2[..., None] + cov2 @ torch.inverse(cov1 + cov2) @ mu1[..., None]
        # mu_f = p * mu1 + (1 - p) * mu2
        # sigma_f = p ** 2 * cov1 + (1 - p) ** 2 * cov2
        return mu_f, sigma_f

    def transform(self, s):
        shape = s.shape
        s = (s.reshape(1, -1) - self.mean) / self.scale
        return s.reshape(shape)

    def inverse_transform(self, s, ):
        shape = s.shape
        s = (s.reshape(1, -1) * self.scale) + self.mean
        return s.reshape(shape)


if __name__ == "__main__":
    import random
    from argparse import ArgumentParser
    from contour_uncertainty.data.camus.dataset import CamusContour
    from vital.data.config import Subset
    from tqdm import tqdm

    np.random.seed(0)

    args = ArgumentParser(add_help=False)
    args.add_argument(
        '--ds', default='camus', choices=['camus', 'lv'], help='Name of the dataset to generate'
    )
    args.add_argument(
        "--path", type=Path, default=None
    )
    args.add_argument(
        "--nb", type=int, default=11
    )
    args.add_argument(
        '--no_mean', action='store_false'
    )
    args.add_argument(
        '--no_std', action='store_false'
    )
    args.add_argument(
        '--sequence', action='store_true', help='Generate sequence (TWO instant PCA)'
    )
    params = args.parse_args()


    # labels = "" #"-".join([str(label) for label in train_ds.labels if str(label) not in ['bg', 'atrium']])
    filename = f'psm_{params.nb}{"" if params.no_mean else "_no_mean"}{"" if params.no_std else "_no_std"}.npy'

    predict = False
    if params.sequence:
        filename = f'sequence_{filename}'
        predict = True

    if params.ds == 'camus':
        train_ds = CamusContour(
            params.path, image_set=Subset.TRAIN, fold=5, predict=predict,
            points_per_side=params.nb, labels=[Label.LV]
        )
        val_ds = CamusContour(
            params.path, image_set=Subset.VAL, fold=5, predict=predict,
            points_per_side=params.nb, labels=[Label.LV]
        )
        filename = f'camus-cont_{filename}'
    elif params.ds == 'lv':
        train_ds = LVDataset(Path(params.path), image_set=Subset.TRAIN, predict=predict)
        val_ds = LVDataset(Path(params.path), image_set=Subset.VAL, predict=predict)
        filename = f'lv-cont_{filename}'
    else:
        raise ValueError("Wrong dataset")

    train_points = []
    for i in tqdm(range(len(train_ds)), desc="Extracting train shapes"):
        try:
            sample = train_ds[i]
            contour = sample['contour'].numpy() if isinstance(sample['contour'], torch.Tensor) else sample['contour']
            train_points.append(contour)
        except Exception as e:
            pass # For cases where only ES on ED are available

    val_points = []
    for i in tqdm(range(len(val_ds)), desc="Extracting validation shapes"):
        try:
            sample = val_ds[i]
            contour = sample['contour'].numpy() if isinstance(sample['contour'], torch.Tensor) else sample['contour']
            val_points.append(contour)
        except Exception as e:
            pass # For cases where only ES on ED are available

    X_train = np.array(train_points).reshape(len(train_points), -1)
    X_val = np.array(val_points).reshape(len(val_points), -1)

    print("Training set shape", X_train.shape)
    print("Validation set shape", X_val.shape)

    scaler = StandardScaler(with_mean=params.no_mean, with_std=params.no_std)
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    mu, Q = pca(torch.tensor(X_train_scaled))

    print('Scaler mean shape', scaler.mean_.shape if scaler.mean_ is not None else 'No mean')
    print('Scaler scale shape', scaler.scale_.shape if scaler.scale_ is not None else 'No scale')

    print('PCA mu shape', mu.shape)
    print('PCA Q shape', Q.shape)

    pca_dict = {
        'mu': mu.numpy(),
        'Q': Q.numpy(),
        'scaler_mean': scaler.mean_ if params.no_mean else np.ones_like(X_train[0]).squeeze(),
        'scaler_scale': scaler.scale_ if params.no_std else np.ones_like(scaler.mean_),
        'X_train': X_train_scaled,
        'X_val': X_val_scaled
    }

    np.save(filename, pca_dict)
