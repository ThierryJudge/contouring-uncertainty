from pathlib import Path

import numpy as np
import torch
import scipy
from matplotlib.lines import Line2D
from scipy.spatial.distance import cdist
from skimage import measure
from skimage.morphology import convex_hull_image

from contour_uncertainty.data.camus.dataset import CamusContour
from contour_uncertainty.sampler.naive import NaiveSampler
from contour_uncertainty.sampler.posterior_shape_model.psm import PosteriorShapeModelSampler
from contour_uncertainty.task.regression.dsnt.dsnt_al import DSNTAleatoric
from contour_uncertainty.utils.plotting import confidence_ellipse, colorline
from matplotlib import pyplot as plt
import random
from contour_uncertainty.utils.contour import contour_spline, reconstruction
from contour_uncertainty.utils.uncertainty_projection import projected_uncertainty
from vital.data.config import Subset
from matplotlib import cm
# import addcopyfighandler  # noqa

from vital.metrics.camus.anatomical.utils import check_segmentation_validity
from vital.data.camus.config import Label

if __name__ == "__main__":

    np.random.seed(5)
    torch.manual_seed(5)

    # data = np.load('data.npy', allow_pickle=True).item()
    # # patient = data['patient0273-2CH_0']
    # patient = data['patient0051-2CH_1']  # Slanted
    # # patient = data['patient0047-2CH_0']
    # # patient = data['patient0047-4CH_0']
    # # patient = data['patient0275-4CH_1']
    # # patient = data['patient0052-2CH_1']
    # # patient = data['patient0208-2CH_0']
    # # patient = data['patient0199-4CH_0']
    # # patient = data['patient0189-2CH_1']
    # img = patient['img'].squeeze()
    # mu_p = np.array(patient['pred'])
    # cov_p = np.array(patient['sigma'])
    # gt = np.array(patient['gt'])

    path = '/home/thierry/data/camus.h5'
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

    index = 99
    # index = 79
    # index = 84
    sample = ds_val[index]
    print(sample['id'])
    img = sample['img']
    gt = sample['contour']

    model = DSNTAleatoric.load_from_checkpoint("/home/thierry/data/tmi-models-FINAL/1/camus-cont-lv_dsnt-al-unet2-False_1.ckpt",
                                               strict=False,
                                               psm_path='/home/thierry/contour-uncertainty/camus-cont_psm_11_no_std.npy',
                                               seq_psm_path='/home/thierry/contour-uncertainty/camus-cont_sequence_psm_11_no_std.npy')
    mu_p, cov_p = model.predict(img)
    mu_p = mu_p.detach().float().squeeze()
    cov_p = cov_p.detach().float().squeeze()

    print(mu_p.shape)
    print(cov_p.shape)

    patient_index = 1
    img = img[patient_index]
    gt = gt[patient_index]
    mu_p = mu_p[patient_index]
    cov_p = cov_p[patient_index]

    min_x = gt[:, 0].min()
    max_x = gt[:, 0].max()
    min_y = gt[:, 1].min()
    max_y = gt[:, 1].max()

    mid_x = (min_x + max_x) / 2
    mid_y = (min_y + max_y) / 2

    w = max_x - min_x
    h = max_y - min_y

    margin = 30
    # Make a square by taking the longuest lenght
    crop_w = max(h, w) + margin
    crop_h = max(h, w) + margin

    x_plot_min = max(mid_x - crop_w // 2, 0)
    x_plot_max = min(mid_x + crop_w // 2, 255)

    y_plot_min = max(mid_y - crop_h // 2, 0)
    y_plot_max = min(mid_y + crop_h // 2, 255)

    # f, ax = plt.subplots(1, 2, figsize=(24, 10))
    # f, ax = plt.subplots(1, 3, figsize=(24, 10))
    f, ax = plt.subplots(1, 1, squeeze=False)
    ax = ax.ravel()

    ########## PLOT PREDICTION ###################
    ax[0].imshow(img.squeeze(), cmap="gray")
    # ax[0].scatter(mu_p[:, 0], mu_p[:, 1], s=50, c='r', label='Initial shape')
    ax[0].scatter(mu_p[:, 0], mu_p[:, 1], s=50, c='r')
    # # ax[0].scatter(gt[:, 0], gt[:, 1], s=10, c='b', label='Gt')
    mean_contour = contour_spline(mu_p.numpy())
    ax[0].plot(mean_contour[:, 0], mean_contour[:, 1], c='r', linewidth=3)

    # gt_contour = contour_spline(gt)
    # ax[0].plot(gt_contour[:, 0], gt_contour[:, 1], c='b', linewidth=3)


    sampler = PosteriorShapeModelSampler(psm_path=Path('/home/thierry/contour-uncertainty/camus-cont_psm_11_no_std.npy'))

    initial_points, points_order = sampler.get_points_order(21)
    print(initial_points)
    print(points_order)
    #
    colors = ['red', 'blue', 'green', 'cyan', 'magenta']
    # colors = cm.get_cmap("Spectral", 6)

    for k in initial_points:
        confidence_ellipse(mu_p[k, 0],
                           mu_p[k, 1],
                           cov_p[k],
                           ax[0],
                           # edgecolor=colors(0),
                           edgecolor=colors[0],
                           linewidth=4)

    for i, order in enumerate(points_order):
        for k in order:
            confidence_ellipse(mu_p[k, 0],
                               mu_p[k, 1],
                               cov_p[k],
                               ax[0],
                               edgecolor=colors[i+1],
                               linewidth=4)

    # line1 = Line2D([0], [0], linewidth=4, label='Level 1', color=colors[0])
    # line2 = Line2D([0], [0], linewidth=4, label='Level 2', color=colors[1])
    # line3 = Line2D([0], [0], linewidth=4, label='Level 3', color=colors[2])
    # line4 = Line2D([0], [0], linewidth=4, label='Level 4', color=colors[3])
    # line5 = Line2D([0], [0], linewidth=4, label='Level 5', color=colors[4])

    line1 = Line2D([0], [0], linewidth=4, label='Lvl. 1', color=colors[0])
    line2 = Line2D([0], [0], linewidth=4, label='Lvl. 2', color=colors[1])
    line3 = Line2D([0], [0], linewidth=4, label='Lvl. 3', color=colors[2])
    line4 = Line2D([0], [0], linewidth=4, label='Lvl. 4', color=colors[3])
    line5 = Line2D([0], [0], linewidth=4, label='Lvl. 5', color=colors[4])

    handles, labels = ax[0].get_legend_handles_labels()
    handles.extend([line1, line2, line3, line4, line5])
    # ax[0].legend(handles=handles, fontsize=28, loc='upper right')
    ax[0].legend(handles=handles, fontsize=40, loc='upper right', handlelength=1)

    # for k in range(0, mu_p.shape[0], 1):
    #     confidence_ellipse(mu_p[k, 0],
    #                        mu_p[k, 1],
    #                        cov_p[k],
    #                        ax[0],
    #                        edgecolor='red',
    #                        linewidth=4)
    # for i in range(mu_p.shape[0]):
    #     ax[0].annotate(str(i), (mu_p[i, 0], mu_p[i, 1]), c='r')
    # ax[0].legend()
    ax[0].set_axis_off()
    ax[0].set_ylim([255, 0])
    ax[0].set_xlim([0, 255])
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    ax[0].set_xlim(x_plot_min, x_plot_max)
    ax[0].set_ylim(y_plot_max, y_plot_min)
    plt.show()

    # exit(0)

    n = 100

    ########## NAIVE SAMPLING ###################
    naive_sampler = NaiveSampler()  # sampled_points=[0, 1, 2, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15, 16, 18, 19, 20])
    naive_contours = naive_sampler(mu_p, cov_p, n)

    ax[1].imshow(img.squeeze(), cmap="gray")
    # ax[1].scatter(mu_p[:, 0], mu_p[:, 1], s=10, c='r', label='Initial shape')
    # ax[1].scatter(gt[:, 0], gt[:, 1], s=10, c='b', label='Gt')
    # for i, contour in enumerate(naive_contours):
    for i in range(min(10, len(naive_contours))):
        sample = contour_spline(naive_contours[i])
        # distance_map = np.min(cdist(sample, mean_contour), axis=0)
        # distance_map = (distance_map - np.min(distance_map)) / (np.max(distance_map) - np.min(distance_map))
        # colorline(sample[:, 0], sample[:, 1], z=distance_map, cmap="plasma", ax=ax[1])
        ax[1].plot(sample[:, 0], sample[:, 1], linewidth=3)
        # ax[1].scatter(contour[:, 0], contour[:, 1], s=10)  # , label=f'Sampled contour {i}')
    # ax[1].legend()
    ax[1].set_axis_off()
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.margins(0, 0)
    ax[1].set_xlim(x_plot_min, x_plot_max)
    ax[1].set_ylim(y_plot_max, y_plot_min)

    rec = []
    for i in range(n):
        segmap = reconstruction(naive_contours[i], 256, 256)
        rec.append(segmap)
    rec = np.array(rec)

    rec_mean = rec.mean(0)
    rec_mean = np.concatenate([rec_mean[None], 1 - rec_mean[None]], axis=0)
    print(rec_mean.shape)
    umap = scipy.stats.entropy(rec_mean, axis=0)

    ins = ax[1].inset_axes([0.7, 0.7, 0.3, 0.3])
    ins.set_axis_off()
    ins.imshow(umap)
    ins.set_xlim(x_plot_min, x_plot_max)
    ins.set_ylim(y_plot_max, y_plot_min)

    ########## PSM SAMPLING ###################

    # cov_width = 7
    # u, v, alpha_proj = projected_uncertainty(mu_p.numpy(), cov_p.numpy(), np.zeros_like(mu_p), all=True)
    # mu_text1 = mu_p + v * u[..., None] * cov_width
    #
    # cov_width = 3
    # u, v, alpha_proj = projected_uncertainty(mu_p.numpy(), cov_p.numpy(), np.zeros_like(mu_p), all=True)
    # mu_text2 = mu_p + v * u[..., None] * cov_width
    #
    #
    # mu_text = np.zeros_like(mu_text2)
    # mu_text[:10] = mu_text1[:10]
    # mu_text[10:] = mu_text2[10:]
    #
    # size=40
    # for i in initial_points:
    #     ax[0].annotate(str(1), (mu_text[i, 0], mu_text[i, 1]), c='r', fontsize=size)
    #
    # for k, order in enumerate(points_order):
    #     for i in order:
    #         ax[0].annotate(str(k + 2), (mu_text[i, 0], mu_text[i, 1]), c='r', fontsize=size)

    debug_img = img
    debug_img = None
    n = 1

    print(mu_p.shape)
    print(cov_p.shape)
    contours = sampler(torch.tensor(mu_p), torch.tensor(cov_p), n=n, debug_img=debug_img).detach().numpy()

    ax[2].imshow(img.squeeze(), cmap="gray")
    ax[2].set_ylim([255, 0])
    ax[2].set_xlim([0, 255])
    # ax.plot(sample_out[0], sample_out[1])
    # ax.scatter(mu_c[:, 0], mu_c[:, 1], s=10, c='b', label='Posterior prediction')
    cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
    cmap = cm.get_cmap(COLORMAP, n)

    for i in range(min(10, len(contours))):
        sample_contour = contour_spline(contours[i])

        # distance_map = np.min(cdist(sample_contour, mean_contour), axis=1)
        # distance_map = (distance_map - np.min(distance_map)) / (np.max(distance_map) - np.min(distance_map))

        # s = sample_contour.round().astype(int)
        # distance_map = prob_map[s[:, 0], s[:, 1]]
        # distance_map = (distance_map - np.min(distance_map)) / (np.max(distance_map) - np.min(distance_map))

        # distance_map = np.sqrt((mean_contour[:, 0] - sample_contour[:, 0]) ** 2 + (mean_contour[:, 1] - sample_contour[:, 1]) ** 2)

        # colorline(sample_contour[:, 0], sample_contour[:, 1], z=distance_map, cmap="plasma", ax=ax[2])
        ax[2].plot(sample_contour[:, 0], sample_contour[:, 1], linewidth=4)
        # ax[2].plot(sample_contour[:, 0], sample_contour[:, 1], c=cmap(i))
        # ax[2].scatter(contour[:, 0], contour[:, 1], s=10)

    # ax[2].scatter(mu_p[:, 0], mu_p[:, 1], s=10, c='r', label='Initial shape')
    # ax[2].scatter(gt[:, 0], gt[:, 1], s=10, c='k', label='Gt')
    # for k in range(0, mu_p.shape[0], 1):
    #     confidence_ellipse(mu_p[k, 0], mu_p[k, 1], cov_p[k], ax[2])
    # ax[2].legend()
    ax[2].set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    ax[2].set_xlim(x_plot_min, x_plot_max)
    ax[2].set_ylim(y_plot_max, y_plot_min)






    #####################################
    reconstructed_mu = np.mean(contours, axis=0)
    print(reconstructed_mu.shape)
    reconstructed_cov = np.zeros((contours.shape[1], 2, 2))
    for i in range(contours.shape[1]):
        reconstructed_cov[i] = np.cov(contours[:, i].T)
    print(reconstructed_cov.shape)

    ax[0].scatter(reconstructed_mu[:, 0], reconstructed_mu[:, 1], s=10, c='b', label='Sampled points')
    for k in range(0, reconstructed_mu.shape[0], 1):
        confidence_ellipse(reconstructed_mu[k, 0],
                           reconstructed_mu[k, 1],
                           reconstructed_cov[k],
                           ax[0],
                           edgecolor='blue',
                           linewidth=1.5)

    rec = []
    for i in range(n):
        segmap = reconstruction(contours[i], 256, 256)
        rec.append(segmap)
    rec = np.array(rec)

    rec_mean = rec.mean(0)
    rec_mean = np.concatenate([rec_mean[None], 1 - rec_mean[None]], axis=0)
    print(rec_mean.shape)
    umap = scipy.stats.entropy(rec_mean, axis=0)

    ins = ax[2].inset_axes([0.7, 0.7, 0.3, 0.3])
    ins.set_axis_off()
    ins.imshow(umap)
    ins.set_xlim(x_plot_min, x_plot_max)
    ins.set_ylim(y_plot_max, y_plot_min)

    plt.show()

    plt.figure()
    plt.imshow(img.squeeze(), cmap="gray")
    plt.gca().set_axis_off()
    # plt.gca().set_xlim(x_plot_min, x_plot_max)
    # plt.gca().set_ylim(y_plot_max, y_plot_min)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.show()

    plt.figure()
    plt.imshow(umap, cmap='gray')
    plt.gca().set_axis_off()
    plt.gca().set_xlim(x_plot_min, x_plot_max)
    plt.gca().set_ylim(y_plot_max, y_plot_min)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.show()

    ax[3].imshow(umap)
    ax[3].set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    ax[3].set_xlim(x_plot_min, x_plot_max)
    ax[3].set_ylim(y_plot_max, y_plot_min)

    plt.savefig('sampling_example.png', dpi=300, bbox_inches='tight')

    ######################################

    f, ax = plt.subplots(1, 1)
    ax.imshow(img.squeeze(), cmap="gray")
    ax.scatter(reconstructed_mu[:, 0], reconstructed_mu[:, 1], s=10, c='b', label='Sampled points')
    for k in range(0, reconstructed_mu.shape[0], 1):
        confidence_ellipse(reconstructed_mu[k, 0],
                           reconstructed_mu[k, 1],
                           reconstructed_cov[k],
                           ax,
                           edgecolor='blue')
    ax.scatter(mu_p[:, 0], mu_p[:, 1], s=10, c='r', label='Initial prediction')
    for k in range(0, mu_p.shape[0], 1):
        confidence_ellipse(mu_p[k, 0],
                           mu_p[k, 1],
                           cov_p[k],
                           ax)

    ax.legend()

    ####################################
    f, ax = plt.subplots(1, n + 1)
    ax = ax.ravel()
    for i in range(n):
        ax[i].imshow(rec[i])
        ax[i].scatter(contours[i, :, 0], contours[i, :, 1], s=10, c='r')

    print(rec.shape)
    validity = [check_segmentation_validity(rec[i], (1, 1), labels=[Label.LV]) for i in range(n)]
    props = [measure.regionprops(measure.label(rec[i]))[0].solidity for i in range(n)]
    hull = [np.sum(convex_hull_image(rec[i])) for i in range(n)]

    print(validity)
    print(props)
    print(hull)
    rec = rec.mean(0)
    rec = np.concatenate([rec[None], 1 - rec[None]], axis=0)
    print(rec.shape)
    umap = scipy.stats.entropy(rec, axis=0)
    ax[-1].imshow(umap)

    plt.show()
