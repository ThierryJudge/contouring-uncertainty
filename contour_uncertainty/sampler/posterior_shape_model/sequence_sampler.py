from pathlib import Path
import numpy as np
from contour_uncertainty.data.ultromics.lv.dataset import LVDataset
import torch
from matplotlib import pyplot as plt
import random
from contour_uncertainty.sampler.posterior_shape_model.posteriorshapemodel import pca, posterior_shape_model
from contour_uncertainty.sampler.posterior_shape_model.psm import PosteriorShapeModelSampler
from contour_uncertainty.sampler.posterior_shape_model.utils import index_to_flat
from contour_uncertainty.utils.plotting import confidence_ellipse


class SequencePSMSampler(PosteriorShapeModelSampler):

    def __init__(self, psm_path: Path, sequence_psm_path: Path, levels: int = 3):
        super().__init__(psm_path, levels)

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
            alpha: torch.Tensor=None,
            n: int=1,
            debug_img=None
    ) -> torch.Tensor:
        """

        Args:
            mu: (2, K, 2)
            cov: (2, K, 2, 2)
            n: Number of samples to sample
            debug_img: (2, H, W)

        Returns:

        """
        samples = []
        for i in range(n):
            first_instant = random.randint(0, 1)
            s = self.sample_two_contours(mu, cov, first_instant=first_instant, debug_img=debug_img)['s']
            samples.append(s[None])
        return torch.cat(samples)

    def sample_two_contours(
            self,
            mu: torch.Tensor,
            cov: torch.Tensor,
            alpha: torch.Tensor = None,
            first_sample=None,
            first_instant=0,
            debug_img=None
    ):
        assert first_instant in [0, 1]
        second_instant = abs(1 - first_instant)
        s = torch.zeros_like(mu)

        if first_sample is None:
            # s[first_instant] = self.sample_contour(mu[first_instant], cov[first_instant], 1, debug_img=debug_img[first_instant]).squeeze()
            alpha_instant1 = alpha[first_instant] if alpha is not None else None
            s[first_instant] = super().__call__(mu[first_instant], cov[first_instant], alpha_instant1, 1).squeeze()
        else:
            s[first_instant] = first_sample

        s_g = torch.zeros_like(mu)
        s_g[first_instant] = s[first_instant].clone()
        s_g = self.sequence_transform(s_g).reshape(-1, 1)

        sampled_indices = list(range(21)) if first_instant == 0 else list(range(21, 42))

        # self.seq_X_train = self.seq_X_train.to(mu.device)
        # self.seq_mean = self.seq_mean.to(mu.device)
        # self.seq_scale = self.seq_scale.to(mu.device)
        # self.seq_mu, self.seq_Q = pca(self.seq_X_train, self.sequence_transform(mu).reshape(-1, 1))

        mu_c, cov_c = posterior_shape_model(s_g, index_to_flat(sampled_indices), self.seq_mu, self.seq_Q, sigma2=1)
        mu_c = self.sequence_inverse_transform(mu_c.squeeze()).reshape((42, 2))
        cov_c *= self.seq_scale
        cov_c = torch.stack([cov_c[2 * i:(2 * i) + 2, 2 * i:(2 * i) + 2] for i in range(42)])

        mu_f, cov_f = self.merge_priors(mu.reshape(42, 2), cov.reshape(42, 2, 2), mu_c.float(), cov_c.float())

        mu_f = mu_f.reshape(2, 21, 2)
        cov_f = cov_f.reshape(2, 21, 2, 2)

        mu_c = mu_c.reshape(2, 21, 2)
        cov_c = cov_c.reshape(2, 21, 2, 2)

        alpha_instant2 = alpha[second_instant] if alpha is not None else None
        s[second_instant] =  super().__call__(mu_f[second_instant], cov_f[second_instant], alpha_instant2, 1).squeeze()

        if debug_img is not None:
            f, (ax1, ax2) = plt.subplots(1, 2)

            ax1.set_title(f'First instant {first_instant}')

            ax1.imshow(debug_img[0].squeeze(), cmap='gray')
            ax2.imshow(debug_img[1].squeeze(), cmap='gray')

            ax1.scatter(mu[0, :, 0], mu[0, :, 1], c='r', s=5, label=r'$mu_p$')
            ax2.scatter(mu[1, :, 0], mu[1, :, 1], c='r', s=5, label=r'$mu_p$')

            ax1.scatter(mu_c[0, :, 0], mu_c[0, :, 1], c='b', s=5, label=r'$mu_c$')
            ax2.scatter(mu_c[1, :, 0], mu_c[1, :, 1], c='b', s=5, label=r'$mu_c$')

            ax1.scatter(mu_f[0, :, 0], mu_f[0, :, 1], c='g', s=5, label=r'$mu_f$')
            ax2.scatter(mu_f[1, :, 0], mu_f[1, :, 1], c='g', s=5, label=r'$mu_f$')

            ax1.scatter(s[0, :, 0], s[0, :, 1], c='m', s=5, label=r'$s$')
            ax2.scatter(s[1, :, 0], s[1, :, 1], c='m', s=5, label=r'$s$')

            for i in range(0, 21):
                # TODO add alpha for skewed ellipse plot
                confidence_ellipse(mu[0, i, 0], mu[0, i, 1], cov[0, i], ax1, n_std=2, linewidth=1, edgecolor='red')
                confidence_ellipse(mu[1, i, 0], mu[1, i, 1], cov[1, i], ax2, n_std=2, linewidth=1, edgecolor='red')

                confidence_ellipse(mu_c[0, i, 0], mu_c[0, i, 1], cov_c[0, i], ax1, n_std=2, linewidth=1,
                                   edgecolor='blue')
                confidence_ellipse(mu_c[1, i, 0], mu_c[1, i, 1], cov_c[1, i], ax2, n_std=2, linewidth=1,
                                   edgecolor='blue')

                confidence_ellipse(mu_f[0, i, 0], mu_f[0, i, 1], cov_f[0, i], ax1, n_std=2, linewidth=1,
                                   edgecolor='green')
                confidence_ellipse(mu_f[1, i, 0], mu_f[1, i, 1], cov_f[1, i], ax2, n_std=2, linewidth=1,
                                   edgecolor='green')

            ax1.legend()
            ax2.legend()
            plt.show()

        return {
            'mu_c': mu_c,
            'cov_c': cov_c,
            'mu_f': mu_f,
            'cov_f': cov_f,
            's': s
        }

    def sample_contour(self, mu, cov, n, debug_img=None):
        return super().__call__(mu, cov, n=n, debug_img=debug_img).squeeze()

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






if __name__ == '__main__':
    from argparse import ArgumentParser
    from contour_uncertainty.data.camus.dataset import CamusContour
    from vital.data.config import Subset
    from tqdm import tqdm
    from vital.data.camus.config import Label
    from sklearn.preprocessing import StandardScaler

    np.random.seed(0)

    args = ArgumentParser(add_help=False)
    args.add_argument("--path", type=Path, default=None)
    args.add_argument("--nb", type=int, default=11)
    args.add_argument('--no_mean', action='store_false')
    args.add_argument('--no_std', action='store_false', )
    params = args.parse_args()

    filename =  f'sequence_psm_{params.nb}{"" if params.no_mean else "_no_mean"}{"" if params.no_std else "_no_std"}.npy'

    # USE PREDICT = TRUE TO GET ES AND ED
    if params.ds == 'camus':
        train_ds = CamusContour(
            params.path, image_set=Subset.TRAIN, fold=5, predict=True,
            points_per_side=params.nb, labels=[Label.LV]
        )
        val_ds = CamusContour(
            params.path, image_set=Subset.VAL, fold=5, predict=True,
            points_per_side=params.nb, labels=[Label.LV]
        )
        filename = f'camus-cont_{filename}'
    elif params.ds == 'lv':
        train_ds = LVDataset(Path(params.path), image_set=Subset.TRAIN, predict=True)
        val_ds = LVDataset(Path(params.path), image_set=Subset.VAL, predict=True)
        filename = f'lv-cont_{filename}'
    else:
        raise ValueError("Wrong dataset")


    train_points = []
    for i in tqdm(range(len(train_ds)), desc="Extracting train shapes"):
        train_points.append(train_ds[i]['contour'].numpy())

    val_points = []
    for sample in tqdm(val_ds, desc="Extracting validation shapes"):
        val_points.append(sample['contour'].numpy())

    X_train = np.array(train_points).reshape(len(train_points), -1)
    X_val = np.array(val_points).reshape(len(val_points), -1)

    print("Training set shape", X_train.shape)
    print("Validation set shape", X_val.shape)

    scaler = StandardScaler(with_mean=params.no_mean, with_std=params.no_std)
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    mu, Q = pca(torch.tensor(X_train_scaled))

    print('Scaler mean shape', scaler.mean_)
    print('Scaler scale shape', scaler.scale_)


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
