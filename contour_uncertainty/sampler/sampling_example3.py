import time
from pathlib import Path

import numpy as np
import scipy
import torch
from matplotlib import pyplot as plt

from contour_uncertainty.data.camus.dataset import CamusContour
from contour_uncertainty.sampler.posterior_shape_model.psm_skew import SkewPosteriorShapeModelSampler
from contour_uncertainty.sampler.posterior_shape_model.psm_skew_sequence import SequenceSkewPSMSampler
from contour_uncertainty.task.regression.dsnt.dsnt_skew import DSNTSkew
from contour_uncertainty.utils.clinical import lv_FAC
from contour_uncertainty.utils.contour import contour_spline, reconstruction
from contour_uncertainty.utils.skew_normal import plot_skewed_normals
from vital.data.camus.config import Label, CamusTags
from vital.data.config import Subset

# import addcopyfighandler  # noqa

if __name__ == "__main__":

    np.random.seed(5)
    torch.manual_seed(5)

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
    # index = 58
    # index = 44
    sample = ds_val[index]
    print(sample['id'])
    img = sample['img']
    gt = sample['contour']

    model_skew = DSNTSkew.load_from_checkpoint("/home/thierry/data/tmi-models-FINAL/1/camus-cont-lv_dsnt-skew-all-unet2-False_1.ckpt",
                                               strict=False,
                                               psm_path='/home/thierry/contour-uncertainty/camus-cont_psm_11_no_std.npy',
                                               seq_psm_path='/home/thierry/contour-uncertainty/camus-cont_sequence_psm_11_no_std.npy')
    print(model_skew.skew_indices)
    sampler = SkewPosteriorShapeModelSampler(psm_path=Path('/home/thierry/contour-uncertainty/camus-cont_psm_11_no_std.npy'), skew_indices=None) #[0, 2, 5, 7, 10, 13, 15, 18, 20])
    sampler = SequenceSkewPSMSampler(psm_path=Path('/home/thierry/contour-uncertainty/camus-cont_psm_11_no_std.npy'),
                                     sequence_psm_path=Path('/home/thierry/contour-uncertainty/camus-cont_sequence_psm_11_no_std.npy'), skew_indices=None) #[0, 2, 5, 7, 10, 13, 15, 18, 20])

    mu_p, cov_p, alpha_p = model_skew.predict(img)
    mu_p = mu_p.detach().float().squeeze()
    cov_p = cov_p.detach().float().squeeze()
    alpha_p = alpha_p.detach().float().squeeze()

    print(mu_p.shape)
    print(cov_p.shape)
    print(alpha_p.shape)

    patient_index = [0, 1]
    img = img[patient_index]
    gt = gt[patient_index]
    mu_p = mu_p[patient_index]
    cov_p = cov_p[patient_index]
    alpha_p = alpha_p[patient_index]

    print(img.shape)
    print(mu_p.shape)
    print(cov_p.shape)
    print(alpha_p.shape)


    n=25
    t0 = time.time()
    s = sampler(mu=mu_p.cuda(), cov=cov_p.cuda(), alpha=alpha_p.cuda(), n=n, debug_img=None, progress_bar=True).detach().cpu()
    t1 = time.time()
    total_n = t1 - t0

    print(f'Generated {n} samples ({s.shape}) in {total_n} sec. ({total_n / n} per sample)')

    average_contour = np.mean(s.numpy(), axis=1)

    print(average_contour.shape)

    ########## PLOT PREDICTION ###################
    f, ax = plt.subplots(2, 2, figsize=(20, 10), squeeze=False)

    reconstructions = []
    for i in range(mu_p.shape[0]):
        mean_contour = contour_spline(mu_p[i].numpy())
        average_c = contour_spline(average_contour[i])

        ax[0, i].imshow(img[i].squeeze(), cmap="gray")
        ax[0, i].scatter(mu_p[i, :, 0], mu_p[i, :, 1], s=10, c='r')
        ax[0, i].scatter(gt[i, :, 0], gt[i, :, 1], s=30, c='b', marker='x', zorder=100)
        ax[0, i].plot(mean_contour[:, 0], mean_contour[:, 1], c='r', linewidth=1)
        plot_skewed_normals(ax[0, i], mu_p[i], cov_p[i], alpha_p[i], flip_y=True)

        for j in range(n):
            contour = contour_spline(s[i, j].numpy())
            ax[0, i].plot(contour[:, 0], contour[:, 1], linewidth=1)


        ax[1, i].imshow(img[i].squeeze(), cmap="gray")
        ax[1, i].scatter(mu_p[i, :, 0], mu_p[i, :, 1], s=10, c='r')
        ax[1, i].scatter(gt[i, :, 0], gt[i, :, 1],  s=30, c='b', marker='x', zorder=100)
        ax[1, i].plot(mean_contour[:, 0], mean_contour[:, 1], c='r', linewidth=1)
        plot_skewed_normals(ax[1, i], mu_p[i], cov_p[i], alpha_p[i], flip_y=True)

        ax[1, i].plot(average_c[:, 0], average_c[:, 1], c='g', linewidth=1)

        for j in range(21):
            ax[1, i].scatter(s[i, :, j, 0], s[i, :, j, 1], s=5)


        rec = []
        for j in range(n):
            segmap = reconstruction(s[i, j].numpy(), 256, 256)
            rec.append(segmap)
        rec = np.array(rec)
        reconstructions.append(rec)

        rec_mean = rec.mean(0)
        rec_mean = np.concatenate([rec_mean[None], 1 - rec_mean[None]], axis=0)
        umap = scipy.stats.entropy(rec_mean, axis=0)

        ins = ax[1, i].inset_axes([0.7, 0.7, 0.3, 0.3])
        ins.set_axis_off()
        ins.imshow(umap)


    masks = np.array(reconstructions)
    print(masks.shape)

    areas = np.sum(masks, axis=(-1, -2))
    print(areas.shape)

    instants = sample[CamusTags.metadata].instants
    print(instants)
    facs = np.array([lv_FAC(masks[0, i], masks[1, i]) for i in range(n)])
    print(facs.shape)

    fig = plt.figure(constrained_layout=True, figsize=(20, 12))
    spec = fig.add_gridspec(ncols=2, nrows=2, width_ratios=[1, 1], height_ratios=[1, 1], hspace=0, wspace=0)


    ax_ed_area = fig.add_subplot(spec[0, 0])
    ax_es_area = fig.add_subplot(spec[0, 1])
    ax_fac = fig.add_subplot(spec[1, :2])

    ax_ed_area.hist(areas[0])
    ax_es_area.hist(areas[1])
    ax_fac.hist(facs)
    ax_ed_area.axvline(x=sample['gt'].sum((-1, -2))[0], c='r')
    ax_es_area.axvline(x=sample['gt'].sum((-1, -2))[1], c='r')
    ax_fac.axvline(x=lv_FAC(sample['gt'][0], sample['gt'][1]), c='r')


    plt.show()

    # # Sample Basal and apex
    # indices = [0, 10, 20]
    # alpha_p[:, 1] = - alpha_p[:, 1]
    # s = model_skew.sampler.sample_points(mu_p, cov_p, alpha_p, sample_indices=indices, n=1)
    # print('s', s.shape)
    # ax[0].scatter(s[indices, 0], s[indices, 1], c='g')
    #
    # # PSM on basal and apex points
    # mu_c, cov_c = model_skew.sampler.compute_psm(s, indices, 1, contour_shape=(21, 2))
    # ax[0].scatter(mu_c[:, 0], mu_c[:, 1], s=10, c='b', label='Posterior prediction')
    #
    # for point in range(21):
    #     confidence_ellipse(mu_c[point, 0], mu_c[point, 1], cov_c[point], ax[0], n_std=2, edgecolor='blue')
    # # Sample with rejection sampling on mid points
    #
    # indices = [5, 15]
    # for ind in indices:
    #     mu_env = torch.mean(torch.cat([mu_p[ind][None], mu_c[ind][None]]), dim=0)
    #     cov_env = torch.tensor([[30., 0.], [0., 30.]])
    #     confidence_ellipse(mu_env[0], mu_env[1], cov_env, ax[0], n_std=2, edgecolor='magenta')
    #
    #     samples = rejection_sampling(
    #         1, mu_p[ind], cov_p[ind], mu_c[ind], cov_c[ind], mu_env, cov_env, 10, alpha1=alpha_p[ind]
    #     )
    #     ax[0].scatter(samples[:, 0], samples[:, 1], c='g', s=10)
    #
    # # Plot samples
