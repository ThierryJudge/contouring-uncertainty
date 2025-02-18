from pathlib import Path

from contour_uncertainty.sampler.posterior_shape_model.posteriorshapemodel import posterior_shape_model
from contour_uncertainty.sampler.posterior_shape_model.psm import PosteriorShapeModelSampler
import numpy as np
from matplotlib import pyplot as plt

from contour_uncertainty.sampler.posterior_shape_model.sequence_sampler import SequencePSMSampler
from contour_uncertainty.sampler.posterior_shape_model.utils import index_to_flat
import torch

from contour_uncertainty.utils.plotting import confidence_ellipse
from vital.data.camus.config import Label

if __name__ == "__main__":
    import random
    from argparse import ArgumentParser
    from contour_uncertainty.data.camus.dataset import CamusContour
    from vital.data.config import Subset
    from tqdm import tqdm

    # np.random.seed(0)

    args = ArgumentParser(add_help=False)
    args.add_argument("--path", type=Path, default='/home/thierry/data/camus.h5')
    params = args.parse_args()

    # params.path = Path("C://Users//ThierryJudge//data//camus.h5")
    train_ds = CamusContour(
        params.path, image_set=Subset.VAL, fold=5, predict=True, labels=[Label.LV]
    )

    sampler = SequencePSMSampler(sequence_psm_path=Path('/home/thierry/contour-uncertainty/camus-cont_sequence_psm_11_no_std.npy'),
                                 psm_path=Path('/home/thierry/contour-uncertainty/camus-cont_psm_11_no_std.npy'))

    index = 10
    index = 684
    index = np.random.choice(len(train_ds)-1)
    print("Index: ", index)
    dataset_sample = train_ds[index]
    img = dataset_sample['img']
    gt = dataset_sample['contour']

    s_g = sampler.sequence_transform(torch.tensor(gt)).reshape(-1, 1)

    indices = list(range(21))
    indices = [0, 10, 20]


    f, ax = plt.subplots(5, 2)

    sigmas = [0.1, 1, 2, 5, 100]
    for i in range(5):

        mu_c, cov_c = posterior_shape_model(s_g, index_to_flat(indices), sampler.seq_mu, sampler.seq_Q, sigma2=sigmas[i])

        cov_c *= sampler.seq_scale

        mu_c = sampler.sequence_inverse_transform(mu_c).reshape(-1, 2).numpy()
        cov_c = torch.stack([cov_c[2 * i:(2 * i) + 2, 2 * i:(2 * i) + 2] for i in range(gt.shape[1] * 2)]).numpy()

        mu_c = mu_c.reshape(2, 21, 2)
        cov_c = cov_c.reshape(2, 21, 2, 2)

        ax[i, 0].imshow(img[0].squeeze(), cmap='gray')
        ax[i, 1].imshow(img[1].squeeze(), cmap='gray')

        ax[i, 0].scatter(gt[0, :, 0], gt[0, :, 1], c='r', s=5)
        ax[i, 1].scatter(gt[1, :, 0], gt[1, :, 1], c='r', s=5)

        ax[i, 0].scatter(mu_c[0, :, 0], mu_c[0, :, 1], s=5, c='b', label=r'Initial shape')
        ax[i, 1].scatter(mu_c[1, :, 0], mu_c[1, :, 1], s=5, c='b', label=r'$\mu_c, \Sigma_c$ (Posterior distribution)')

        for index in range(0, mu_c.shape[1], 1):
            confidence_ellipse(mu_c[0, index, 0], mu_c[0, index, 1], cov_c[0, index], ax[i, 0], n_std=2, edgecolor='blue', linewidth=2)
            confidence_ellipse(mu_c[1, index, 0], mu_c[1, index, 1], cov_c[1, index], ax[i, 1], n_std=2, edgecolor='blue', linewidth=2)

        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.show()
