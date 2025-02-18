from pathlib import Path

from contour_uncertainty.sampler.posterior_shape_model.posteriorshapemodel import posterior_shape_model
from contour_uncertainty.sampler.posterior_shape_model.psm import PosteriorShapeModelSampler
import numpy as np
from matplotlib import pyplot as plt

from contour_uncertainty.sampler.posterior_shape_model.utils import index_to_flat
import torch

from contour_uncertainty.utils.plotting import confidence_ellipse
from vital.data.camus.config import Label

# data = np.load('data.npy', allow_pickle=True).item()
# # patient = data['patient0197-4CH_0']
# # patient = data['patient0051-2CH_1']  # Slanted
# patient = data['patient0047-2CH_0']
# # patient = data['patient0047-4CH_0']
# # patient = data['patient0275-4CH_1']
# # patient = data['patient0052-2CH_1']
# # patient = data['patient0208-2CH_0']
# # patient = data['patient0273-2CH_0']
# img = patient['img'].squeeze()
# gt = np.array(patient['gt'])

if __name__ == "__main__":
    import random
    from argparse import ArgumentParser
    from contour_uncertainty.data.camus.dataset import CamusContour
    from vital.data.config import Subset
    from tqdm import tqdm

    np.random.seed(0)

    args = ArgumentParser(add_help=False)
    args.add_argument("--path", type=Path, default=None)
    params = args.parse_args()

    # params.path = Path("C://Users//ThierryJudge//data//camus.h5")
    train_ds = CamusContour(
        params.path, image_set=Subset.TRAIN, fold=5, predict=False, labels=[Label.LV]
    )

    # sampler = PosteriorShapeModelSampler(psm_path=Path('psm_11_lv.npy'))
    # sampler = PosteriorShapeModelSampler(psm_path=Path('psm_11_lv_no_mean.npy'))
    sampler = PosteriorShapeModelSampler(psm_path=Path('/home/thierry/contour-uncertainty/camus-cont_psm_11_no_std.npy'))
    # sampler = PosteriorShapeModelSampler(psm_path=Path('psm_11_lv_no_mean_no_std.npy'))

    print(sampler.mean)
    print(sampler.scale)
    indices = [[5],
               [0, -1],
               [0, 10, -1],
               [0, 5, 10, -5, -1],
               list(range(21))]

    # indices = [[10],
    #            [0, -1],
    #            [0, 10, -1],
    #            [0, 5, 10, -5, -1]]
               # list(range(21))]

    indices = [
               [0, 20],
               [0, 10, 20],
               [0, 5, 10, 16, 20]]

    dataset_sample = train_ds[10]
    img = dataset_sample['img']
    gt = dataset_sample['contour']


    f, axes = plt.subplots(1, len(indices), figsize=(18,8))
    # f, axes = plt.subplots(2, 2, figsize=(10,10))
    axes = axes.ravel()
    for i, idx in enumerate(indices):
        axes[i].set_title(f"{len(idx)} input points", fontsize=25)
        # axes[i].imshow(img.squeeze(), cmap="gray")
        axes[i].set_xlim([0, 256])
        axes[i].set_ylim([256, 0])
        axes[i].scatter(gt[:, 0], gt[:, 1], s=50, c='r', label=r'Initial shape')
        axes[i].scatter(gt[idx, 0], gt[idx, 1], s=250, c='r', marker="*", label=r'$s^{(g)}$ (Partial input)')

    for step, (sigma, color) in enumerate(zip([0.1, 1, 5], ['cyan', 'blue', 'green'])):
        for i, idx in enumerate(indices):

            s_g = sampler.transform(torch.tensor(gt)).reshape(-1, 1)

            mu_c, cov_c = posterior_shape_model(s_g, index_to_flat(idx), sampler.mu, sampler.Q, sigma2=sigma)

            cov_c *= sampler.scale

            mu_c = sampler.inverse_transform(mu_c).reshape(-1, 2).numpy()
            cov_c = torch.stack([cov_c[2 * i:(2 * i) + 2, 2 * i:(2 * i) + 2] for i in range(len(gt))]).numpy()

            # axes[i].set_title(f"{len(idx)} input points", fontsize=25)
            # # axes[i].imshow(img.squeeze(), cmap="gray")
            # axes[i].set_xlim([0, 256])
            # axes[i].set_ylim([256, 0])
            # axes[i].scatter(gt[:, 0], gt[:, 1], s=50, c='r', label=(r'Initial shape' if step == 0 else None))

            for index in range(0, mu_c.shape[0], 1):
                if index not in idx:
                    confidence_ellipse(mu_c[index, 0], mu_c[index, 1], cov_c[index], axes[i], n_std=2, edgecolor=color, linewidth=2)
            axes[i].set_axis_off()
            # axes[i].scatter(gt[idx, 0], gt[idx, 1], s=250, c='r', marker="*", label=(r'$s^{(g)}$ (Partial input)' if step == 0 else None))

            p = np.ones(21, dtype=int)
            p[idx] = 0
            print(idx)
            print(p)
            z = np.where(p)[0]
            print(z)
            # axes[i].scatter(mu_c[z, 0], mu_c[z, 1], s=50, c=color, label=r'$\mu_c, \Sigma_c$ (Posterior distribution), $\sigma^2$=' + f'{sigma}')
            axes[i].scatter(mu_c[z, 0], mu_c[z, 1], s=50, c=color, label=r'$\mu_c, \Sigma_c$ ($\sigma^2$=' + f'{sigma})')

            # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            # plt.margins(0, 0)

            min_x = gt[:, 0].min()
            max_x = gt[:, 0].max()
            min_y = gt[:, 1].min()
            max_y = gt[:, 1].max()

            mid_x = (min_x + max_x) / 2
            mid_y = (min_y + max_y) / 2

            w = max_x - min_x
            h = max_y - min_y

            margin_x = 20
            margin_y = 60
            # Make a square by taking the longuest lenght
            crop_w = max(h, w) + margin_x
            crop_h = max(h, w) + margin_y

            x_plot_min = max(mid_x - crop_w // 2, 0)
            x_plot_max = min(mid_x + crop_w // 2, 255)

            y_plot_min = max(mid_y - crop_h // 2, 0)
            y_plot_max = min(mid_y + crop_h // 2, 255)

            axes[i].set_xlim(x_plot_min, x_plot_max)
            axes[i].set_ylim(y_plot_max, y_plot_min)

    # axes[2].set_zorder(100)
    # axes[2].legend(prop={'size': 20}, loc='lower center', ncol=3)

    lines_labels = [axes[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    f.legend(lines[:2], labels[:2], prop={'size': 25}, loc='lower center', ncol=2, bbox_to_anchor=(0.5,0.09))
    f.legend(lines[2:], labels[2:], prop={'size': 25}, loc='lower center', ncol=3, bbox_to_anchor=(0.5,-0.01))


    plt.subplots_adjust(top=None, bottom=None, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig('posterior_shape_model.png', dpi=300, bbox_inches='tight')
    plt.show()
