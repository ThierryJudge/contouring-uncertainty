import random
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from vital.data.camus.config import Label
from vital.data.config import Subset

from contour_uncertainty.data.camus.dataset import CamusContour
from contour_uncertainty.sampler.posterior_shape_model.sequence_sampler import SequencePSMSampler
from contour_uncertainty.task.regression.dsnt.dsnt_al import DSNTAleatoric
from contour_uncertainty.utils.clinical import lv_FAC
from contour_uncertainty.utils.contour import contour_spline, reconstruction
from contour_uncertainty.utils.plotting import confidence_ellipse, crop_axis
from contour_uncertainty.utils.uncertainty_projection import projected_uncertainty

# import addcopyfighandler  # noqa

if __name__ == "__main__":

    seed = 709  # np.random.randint(0, 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(seed)

    path = "/home/thierry/data/camus.h5"
    COLORMAP = 'brg'

    ds_val = CamusContour(
        Path(path),
        image_set=Subset.TEST,
        fold=5,
        predict=True,
        points_per_side=11,
        use_sequence=False,
        labels=[Label.LV]
    )

    print(len(ds_val))

    index = random.randint(0, len(ds_val) - 1)
    index = 3  # CURVE
    # index = 99
    # index = 79
    # index = 84
    # index = 46
    # index = 94
    print("INDEX", index)

    sample = ds_val[index]
    print(sample['id'])
    img = sample['img']
    gt = sample['contour']
    seg_gt = sample['gt']

    model = DSNTAleatoric.load_from_checkpoint("/home/thierry/data/tmi-models-FINAL/1/camus-cont-lv_dsnt-al-unet2-False_1.ckpt", strict=False,
                                               psm_path='/home/thierry/contour-uncertainty/camus-cont_psm_11_no_std.npy',
                                               seq_psm_path='/home/thierry/contour-uncertainty/camus-cont_sequence_psm_11_no_std.npy'
                                               )
    mu_p, cov_p = model.predict(img)
    mu_p = mu_p.detach().float().squeeze()
    cov_p = cov_p.detach().float().squeeze()

    seg_p = np.zeros((2, 256, 256))
    seg_p[0] = reconstruction(mu_p[0].numpy(), 256, 256)
    seg_p[1] = reconstruction(mu_p[1].numpy(), 256, 256)

    mu_w = torch.zeros_like(mu_p)
    seg_w = np.zeros((2, 256, 256))

    cov_width = 1.5
    u, v, alpha_proj = projected_uncertainty(mu_p[0].numpy(), cov_p[0].numpy(), np.zeros_like(mu_p[0]), all=True)
    mu_w[0] = mu_p[0] - v * u[..., None] * cov_width
    c_w_1 = contour_spline(mu_w[0].numpy())
    seg_w[0] = reconstruction(mu_w[0].numpy(), 256, 256)

    u, v, alpha_proj = projected_uncertainty(mu_p[1].numpy(), cov_p[1].numpy(), np.zeros_like(mu_p[1]), all=True)
    mu_w[1] = mu_p[1] + v * u[..., None] * cov_width
    c_w_2 = contour_spline(mu_w[1].numpy())
    seg_w[1] = reconstruction(mu_w[1].numpy(), 256, 256)

    FAC_gt = lv_FAC(seg_gt[0], seg_gt[1])
    FAC_w = lv_FAC(seg_w[0], seg_w[1])
    FAC_p = lv_FAC(seg_p[0], seg_p[1])

    print("FAC GT", FAC_gt)
    print("FAC mean", FAC_p)
    print("FAC W", FAC_w)

    sampler = SequencePSMSampler(psm_path=Path('/home/thierry/contour-uncertainty/camus-cont_psm_11_no_std.npy'),
                                 sequence_psm_path=Path('/home/thierry/contour-uncertainty/camus-cont_sequence_psm_11_no_std.npy'))

    sample_dict = sampler.sample_two_contours(mu_p, cov_p, first_sample=mu_w[0], first_instant=0)

    s = sample_dict['s'][1].detach().numpy()
    mu_c = sample_dict['mu_c']
    cov_c = sample_dict['cov_c']
    mu_f = sample_dict['mu_f']
    cov_f = sample_dict['cov_f']

    s_contour = contour_spline(s)
    seg_s = reconstruction(s, 256, 256)

    FAC_s = lv_FAC(seg_w[0], seg_s)
    print("FAC GT", FAC_gt)
    print("FAC W", FAC_w)
    print("FAC S", FAC_s)

    f, ax = plt.subplots(1, 2, figsize=(15, 7.5))
    ax = ax.ravel()

    x_plot_min, x_plot_max, y_plot_min, y_plot_max = crop_axis(gt[0], 50)
    ax[0].set_xlim(x_plot_min, x_plot_max)
    ax[0].set_ylim(y_plot_max, y_plot_min)
    ax[1].set_xlim(x_plot_min, x_plot_max)
    ax[1].set_ylim(y_plot_max, y_plot_min)

    ########## PLOT PREDICTION ###################
    ax[0].imshow(img[0].squeeze(), cmap="gray")
    ax[0].scatter(mu_p[0, :, 0], mu_p[0, :, 1], s=10, c='r')
    # ax[0].scatter(mu_w[0, :, 0], mu_w[0, :, 1], s=10, c='m')
    # ax[0].scatter(gt[0, :, 0], gt[0, :, 1], s=10, c='b', label='Gt')
    mean_contour = contour_spline(mu_p[0].numpy())
    ax[0].plot(mean_contour[:, 0], mean_contour[:, 1], c='r')
    ax[0].plot(c_w_1[:, 0], c_w_1[:, 1], c='m', linewidth=2)
    # ax[0].plot(s_contour[:, 0], s_contour[:, 1], c='y')

    # ax[0].scatter(mu_c[0, :, 0], mu_c[0, :, 1], c='b', s=5, label=r'$mu_c$')
    # ax[0].scatter(mu_f[0, :, 0], mu_f[0, :, 1], c='g', s=5, label=r'$mu_f$')

    for k in range(0, mu_p.shape[1], 1):
        confidence_ellipse(mu_p[0, k, 0], mu_p[0, k, 1], cov_p[0, k], ax[0], edgecolor='red', linewidth=1.5)
        # confidence_ellipse(mu_c[0, k, 0], mu_c[0, k, 1], cov_c[0, k], ax[0], edgecolor='blue', linewidth=1.5)
        # confidence_ellipse(mu_f[0, k, 0], mu_f[0, k, 1], cov_f[0, k], ax[0], edgecolor='green', linewidth=1.5)
    ax[0].set_axis_off()

    ax[1].imshow(img[1].squeeze(), cmap="gray")
    # ax[1].scatter(mu_p[1, :, 0], mu_p[1, :, 1], s=10, c='r')
    ax[1].scatter(mu_p[1, :, 0], mu_p[1, :, 1], s=10, c='r', label=r'$\hat{\mu}, \hat{\Sigma}$' + f'  [FAC = {FAC_p * 100:.0f}%]')
    # ax[1].scatter(mu_w[1, :, 0], mu_w[1, :, 1], s=10, c='m')
    # ax[1].scatter(gt[1, :, 0], gt[1, :, 1], s=10, c='b', label='Gt')
    mean_contour = contour_spline(mu_p[1].numpy())

    # ax[1].scatter(mu_c[1, :, 0], mu_c[1, :, 1], c='b', s=5, label=r'$\mathcal{N}_c$')
    # ax[1].scatter(mu_f[1, :, 0], mu_f[1, :, 1], c='g', s=5, label=r'$mu_f$')
    # ax[1].scatter(s[:, 0], s[:, 1], c='y', s=5, label=r'$s$')

    ax[1].plot(mean_contour[:, 0], mean_contour[:, 1], c='r')
    # ax[1].plot(mean_contour[:, 0], mean_contour[:, 1], c='r', label=f'Mean (FAC = {FAC_p * 100:.0f}%)')
    # ax[1].plot(c_w_2[:, 0], c_w_2[:, 1], c='m', linewidth=2, label=f'Failure Case (FAC = {FAC_w * 100:.0f}%)')
    ax[1].plot(c_w_2[:, 0], c_w_2[:, 1], c='m', linewidth=2, label=r'$s \sim \mathcal{N}(\hat{\mu}, \hat{\Sigma})$' + f'  [FAC = {FAC_w * 100:.0f}%]')
    ax[1].scatter(mu_f[1, :, 0], mu_f[1, :, 1], c='g', s=20, label=r'$\widetilde{\mu}_f, \widetilde{\Sigma}_f$')
    ax[1].plot(s_contour[:, 0], s_contour[:, 1], c='c', linewidth=2,
               label=r'$s \sim \mathcal{N}(\widetilde{\mu}_f, \widetilde{\Sigma}_f)$' + f'  [FAC = {FAC_s * 100:.0f}%]')
    for k in range(0, mu_p.shape[1], 1):
        confidence_ellipse(mu_p[1, k, 0], mu_p[1, k, 1], cov_p[1, k], ax[1], edgecolor='red', linewidth=1.5)
        # confidence_ellipse(mu_c[1, k, 0], mu_c[1, k, 1], cov_c[1, k], ax[1], edgecolor='blue', linewidth=1.5)
        confidence_ellipse(mu_f[1, k, 0], mu_f[1, k, 1], cov_f[1, k], ax[1], edgecolor='green', linewidth=1.5)
    ax[1].set_axis_off()
    # ax[1].legend(fontsize=20, ncol=2, loc = "lower right")

    lines_labels = [ax.get_legend_handles_labels() for ax in f.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    # f.legend(lines, labels)
    f.legend(lines, labels, fontsize=20, ncol=4, loc="lower center", columnspacing=2, handletextpad=0.15, handlelength=1)

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)

    plt.savefig('sequence_sampler.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()
