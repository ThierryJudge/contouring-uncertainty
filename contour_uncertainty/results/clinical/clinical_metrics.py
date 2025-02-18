import linecache
import os
from collections import defaultdict
from itertools import cycle
from pathlib import Path
from pprint import pprint
from typing import List, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from pytorch_lightning.loggers import CometLogger
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

from contour_uncertainty.task.segmentation.utils import get_contour_from_mask
from vital.data.camus.config import Instant

from contour_uncertainty.data.config import BatchResult
from contour_uncertainty.results.clinical.instant import InstantMetric, AreaError
from contour_uncertainty.results.clinical.patient import Volume, PatientMetric
from contour_uncertainty.results.clinical.view import GLS, FAC, ViewMetric
from contour_uncertainty.results.metrics import Metrics
from contour_uncertainty.results.utils.calibration import compute_calibration, compute_adaptive_calibration
from contour_uncertainty.utils.contour import contour_spline
from contour_uncertainty.utils.plotting import str2tex, confidence_ellipse


class ClinicalMetrics(Metrics):

    def __init__(
            self,
            instant_metrics: List[InstantMetric] = (AreaError(),),
            view_metrics: List[ViewMetric] = (FAC(), GLS(),),
            # view_metrics: List[ViewMetric] = (FAC(),),
            patient_metrics: List[PatientMetric] = (Volume(),),
            output_dir: Path = Path('clinical')
    ):
        super().__init__()
        self.instant_metrics = instant_metrics or []
        self.view_metrics = view_metrics or []
        self.patient_metrics = patient_metrics or []

        self.metric_results = {}

        self.output_dir = output_dir

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_predict_epoch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: List[List[BatchResult]]
    ) -> None:

        self.metric_results = {}

        result_dict = defaultdict(lambda: defaultdict(BatchResult))
        for res in outputs[0]:
            patient_id, view_id = res.id.split('/')
            result_dict[patient_id][view_id] = res


        # print([(patient_id, patient_results.keys()) for patient_id, patient_results in result_dict.items()])

        instant_df = pd.concat([metric(outputs[0]) for metric in self.instant_metrics])
        instant_df.to_csv('instant_df.csv')
        print(instant_df)
        self.plot_correlation(instant_df, 'Area')
        self.plot_correlation(instant_df, 'Area', x_str='pred', y_str='mean', c_str=None)
        self.plot_calibration(instant_df, 'Area', trainer=trainer)

        view_df = pd.concat([metric(outputs[0]) for metric in self.view_metrics], axis=1)
        view_df.to_csv('view_df.csv')

        for metric in ['FAC', 'GLS']:
            reject = np.array(view_df[f'{metric}_reject'], dtype=bool)
                # print(reject)
                # print(reject.sum())
                # print(np.size(reject))
                # print(np.size(reject))
                # print(reject.sum() / np.size(reject))
            self.metric_results[f'{metric}_reject'] = reject.sum() / np.size(reject)

        self.plot_correlation(view_df, 'FAC')
        self.plot_correlation(view_df, 'FAC', x_str='pred', y_str='mean', c_str=None)
        self.plot_calibration(view_df, 'FAC', trainer=trainer)
        self.plot_correlation(view_df, 'GLS')
        self.plot_correlation(view_df, 'GLS', x_str='pred', y_str='mean', c_str=None)

        if 'GLS_contour_gt' in view_df.columns:
            print("CONTOUR")
            self.plot_correlation(view_df, 'GLS', x_str='pred', y_str='contour_pred', c_str=None)
            self.plot_correlation(view_df, 'GLS', x_str='gt', y_str='contour_gt', c_str=None)
            self.plot_correlation(view_df, 'GLS', x_str='gt', y_str='contour_gt', c_str=None)
            self.plot_calibration(view_df, 'GLS_contour', trainer=trainer)

        self.plot_calibration(view_df, 'GLS', trainer=trainer)
        print(view_df)

        patient_df = pd.concat([metric(result_dict) for metric in self.patient_metrics])
        patient_df.to_csv('patient_df.csv')
        self.plot_correlation(patient_df, 'EF')
        self.plot_correlation(patient_df, 'EF', x_str='pred', y_str='mean', c_str=None)
        self.plot_calibration(patient_df, 'EF', trainer=trainer)
        self.plot_correlation(patient_df, 'ESV')
        self.plot_correlation(patient_df, 'ESV', x_str='pred', y_str='mean', c_str=None)
        self.plot_calibration(patient_df, 'ESV')
        self.plot_correlation(patient_df, 'EDV')
        self.plot_correlation(patient_df, 'EDV', x_str='pred', y_str='mean', c_str=None)
        self.plot_calibration(patient_df, 'EDV')
        print(patient_df)

        reject = np.array(patient_df[f'EF_reject'], dtype=bool)
        self.metric_results[f'EF_reject'] = reject.sum() / np.size(reject)

        volume_df = self.merge_volume_df(patient_df)
        volume_df.to_csv('volume_df.csv')
        print(volume_df)

        self.plot_correlation(volume_df, 'Volume')
        self.plot_correlation(volume_df, 'Volume', x_str='pred', y_str='mean', c_str=None)
        self.plot_calibration(volume_df, 'Volume', trainer=trainer)

        metric_results = pd.DataFrame([self.metric_results])
        metric_results.to_csv('clinical_metrics.csv')

        pprint(self.metric_results)

        trainer.logger.log_metrics(self.metric_results)

        metric_figure_dir = self.output_dir / 'metric_figures'
        metric_figure_dir.mkdir(exist_ok=True)

        metric_figure_dir2 = self.output_dir / 'metric_figures2'
        metric_figure_dir2.mkdir(exist_ok=True)

        for patient_id, patient_results in result_dict.items():
            for view_id, view_results in patient_results.items():
                metric_plot(view_results, view_df, instant_df, metric_figure_dir)
                metric_plot(view_results, view_df, instant_df, metric_figure_dir2, use_contour=False)

    @staticmethod
    def merge_volume_df(patient_df: pd.DataFrame):
        esv_patient = patient_df.filter(regex='ESV')
        esv_patient.index = [f'{patient}/ES' for patient in esv_patient.index]
        esv_patient.columns = [f'Volume_{col.replace("ESV_", "")}' for col in esv_patient.columns]

        edv_patient = patient_df.filter(regex='EDV')
        edv_patient.index = [f'{patient}/ED' for patient in edv_patient.index]
        edv_patient.columns = [f'Volume_{col.replace("EDV_", "")}' for col in edv_patient.columns]

        return pd.concat([esv_patient, edv_patient])

    def plot_correlation(
        self,
        df: pd.DataFrame,
        metric_name: str,
        x_str: str = 'gt',
        y_str: str = 'pred',
        c_str: Optional[str] = 'std'
    ):
        gt = np.array(df[f'{metric_name}_{x_str}'], dtype=float)
        pred = np.array(df[f'{metric_name}_{y_str}'], dtype=float)
        color = np.array(df[f'{metric_name}_{c_str}']) if c_str is not None else None

        plt.figure()
        cc = plt.scatter(gt, pred, c=color)
        if c_str is not None:
            plt.colorbar(cc, ax=plt.gca())

        # plt.errorbar(area_gt, area_pred, yerr=2 * area_std, fmt="o")

        # plt.xlabel(f'GT {metric_name}')
        # plt.ylabel(f'Predicted {metric_name}')

        plt.xlabel(f'{x_str} {metric_name}')
        plt.ylabel(f'{y_str} {metric_name}')

        nas = np.logical_or(np.isnan(pred), np.isnan(gt))
        inf = np.logical_or(np.isinf(pred), np.isinf(gt))
        invalid = np.logical_or(nas, inf)
        corr, _ = pearsonr(gt[~invalid], pred[~invalid])
        mae = mean_absolute_error(gt[~invalid], pred[~invalid])

        plt.axline((0, 0), slope=1, color='black')
        plt.title(f"correlation={corr:.3f}, MAE={mae:.3f}")
        plt.savefig(self.output_dir / f'{metric_name}_{x_str}_{y_str}_correlation.png')
        plt.close()

        self.metric_results[f'{metric_name}_{x_str}_{y_str}_correlation'] = corr
        self.metric_results[f'{metric_name}_{x_str}_{y_str}_mae'] = mae

    # def crps(self, df: pd.DataFrame, metric_name: str):
    #     gt = np.array(df[f'{metric_name}_gt'])
    #     mc = np.array(df[f'{metric_name}_mc'])
    #     std = np.array(df[f'{metric_name}_std'])
    #     mu = np.array(df[f'{metric_name}_mean'])
    #
    #
    #
    #     normal_crps = (mu - gt) / gt
    #
    #     print(gt.shape)
    #     print(mc.shape)

    def plot_calibration(
            self,
            df: pd.DataFrame,
            metric_name: str,
            trainer=None
    ):
        std = np.array(df[f'{metric_name}_std'], dtype=float)
        error = np.array(df[f'{metric_name}_error'], dtype=float)

        nas = np.logical_or(np.isnan(std), np.isnan(error))
        std = std[~nas]
        error = error[~nas]

        if f'{metric_name}_reject' in df.columns:
            filters = ~np.array(df[f'{metric_name}_reject'], dtype=bool)
            filters = filters[~nas]
        else:
            filters = None

        uce, bins_avg_conf, bins_avg_acc, bins_size = compute_calibration(error, std, filters=filters)
        a_uce, a_bins_avg_conf, a_bins_avg_acc, a_bins_size = compute_adaptive_calibration(error, std, filters=filters)

        plt.figure()
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        # ax1[i].scatter(bins_avg_conf, bins_avg_acc, c=bins_size, s=bins_size)
        ax1.plot(bins_avg_conf, bins_avg_acc, marker='o')
        ax2.plot(a_bins_avg_conf, a_bins_avg_acc, marker='o')

        ax12 = ax1.twinx()
        try:
            ax12.bar(bins_avg_conf, bins_size, alpha=0.7, width=np.min(np.diff(bins_avg_conf)) / 2)
        except:
            ax12.bar(bins_avg_conf, bins_size, alpha=0.7)

        # data_range = [std.min(), std.max()]
        # ax1.plot(data_range, data_range, "--", c="k", label="Perfect calibration")

        ax1.plot(ax1.get_xlim(), ax1.get_xlim(), "--", c="k", label="Perfect calibration")
        ax2.plot(ax2.get_xlim(), ax2.get_xlim(), "--", c="k", label="Perfect calibration")
        # plt.axline((0, 0), slope=1, color='black', label="Perfect calibration")
        ax1.set_title(f"UCE={uce:.3f}")
        ax1.set_ylabel(f'{metric_name} error')
        ax1.set_xlabel(str2tex(f'sigma_{metric_name}'))

        ax2.set_title(f"A-UCE={a_uce:.3f}")
        ax2.set_ylabel(f'{metric_name} error')
        ax2.set_xlabel(str2tex(f'sigma_{metric_name}'))

        plt.savefig(self.output_dir / f'{metric_name}_calibration.png')
        plt.close()

        self.metric_results[f'{metric_name}_uce'] = uce
        self.metric_results[f'{metric_name}_a-uce'] = a_uce

        if trainer is not None:
            if isinstance(trainer.logger, CometLogger):
                trainer.logger.experiment.log_curve(f'{metric_name}_uce', bins_avg_conf, bins_avg_acc)
                trainer.logger.experiment.log_curve(f'{metric_name}_auce', a_bins_avg_conf, a_bins_avg_acc)


def metric_plot(view_result: BatchResult, view_df: pd.DataFrame, instant_df: pd.DataFrame, output_dir: Path,
                use_contour=True):
    # if view_result.contour is None:
    #     return

    def get_df_items(df: pd.DataFrame, key, metric_name):
        pred = df.at[key, f'{metric_name}_pred']
        std = df.at[key, f'{metric_name}_std']
        gt = df.at[key, f'{metric_name}_gt']
        mc = np.array(df.at[key, f'{metric_name}_mc'], dtype=float)
        mean = df.at[key, f'{metric_name}_mean']

        mc = mc[~np.isnan(mc)]
        al = df.at[key, f'{metric_name}_aleatoric_std']
        ep = df.at[key, f'{metric_name}_epistemic_std']

        return pred, std, gt, mc, mean, al, ep

    def plot_metric_axis(ax, df: pd.DataFrame, key, metric_name, vertical=False):
        pred, std, gt, mc, mean, al, ep = get_df_items(df, key, metric_name)
        try:
            reject = df.at[key, f'{metric_name}_reject']
        except:
            reject = None

        if vertical:
            ax.hist(mc, alpha=0.5,  orientation='horizontal')
            x = ax.get_xlim()[0] + np.mean(ax.get_xlim())
            # ax.errorbar(pred, y, xerr=al+ep, fmt='o', capsize=2, c='b')
            # ax.errorbar(pred, y, xerr=al, fmt='o', capsize=2, c='r')
            # ax.scatter(mean, y, c='g')

            ax.errorbar(x, mean, yerr=al + ep, fmt='o', capsize=2, c='b', linewidth=2)
            ax.errorbar(x, mean, yerr=al, fmt='o', capsize=2, c='r', linewidth=2)
            # ax.scatter(x, pred, c='g')

            ax.scatter(x, gt, c='k', marker='x', s=100)

            return

        # ax.hist(mc, alpha=0.5)
        # y = ax.get_ylim()[0] + np.mean(ax.get_ylim())
        y = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.75
        # ax.errorbar(pred, y, xerr=al+ep, fmt='o', capsize=2, c='b')
        # ax.errorbar(pred, y, xerr=al, fmt='o', capsize=2, c='r')
        # ax.scatter(mean, y, c='g')

        fmt = 'x' if reject else 'o'
        ax.errorbar(mean, y, xerr=al+ep, fmt=fmt, capsize=2, c='b', elinewidth=2)
        ax.errorbar(mean, y, xerr=al, fmt=fmt, capsize=2, c='r', elinewidth=2)
        ax.scatter(mean, y, c='r', s=100)
        # ax.scatter(pred, y, c='g')

        ax.scatter(gt, y, c='k', s=100, zorder=2)
        # ax.scatter(gt, y, c='k', marker='x', s=100)
        #
        return reject


    ES = view_result.instants[Instant.ES]
    ED = view_result.instants[Instant.ED]

    fig = plt.figure(constrained_layout=True, figsize=(20, 12))
    spec = fig.add_gridspec(ncols=2, nrows=4, width_ratios=[1, 1], height_ratios=[1, 0.05, 0.05, 0.05], hspace=0,
                            wspace=0)

    # fig = plt.figure(constrained_layout=True, figsize=(20, 10))
    # spec = fig.add_gridspec(ncols=2, nrows=4, width_ratios=[1, 1], height_ratios=[1, 0.1, 0.1, 0.1])
    # spec = fig.add_gridspec(ncols=4, nrows=3, width_ratios=[0.1, 1, 1, 0.1], height_ratios=[1, 0.1, 0.1], hspace=0, wspace=0)

    ax_ed_img = fig.add_subplot(spec[0, 0])
    ax_es_img = fig.add_subplot(spec[0, 1])
    ax_ed_area = fig.add_subplot(spec[1, 0])
    ax_es_area = fig.add_subplot(spec[1, 1])
    ax_fac = fig.add_subplot(spec[2, :2])
    ax_gls = fig.add_subplot(spec[3, :2], sharex=ax_fac)

    # ax_ed_area = fig.add_subplot(spec[0, 0])
    # ax_ed_img = fig.add_subplot(spec[0, 1])
    # ax_es_img = fig.add_subplot(spec[0, 2])
    # ax_es_area = fig.add_subplot(spec[0, 3], sharey=ax_ed_area)
    # ax_fac = fig.add_subplot(spec[1, :])
    # ax_gls = fig.add_subplot(spec[2, :], sharex=ax_fac)

    ax_ed_img.set_axis_off()
    ax_es_img.set_axis_off()

    ax_ed_img.set_title("ED")
    ax_es_img.set_title("ES")

    ax_ed_img.imshow(view_result.img[ED].squeeze(), cmap='gray')
    ax_es_img.imshow(view_result.img[ES].squeeze(), cmap='gray')

    ins = ax_ed_img.inset_axes([0.7, 0.7, 0.3, 0.3])
    ins.set_axis_off()
    rmin, rmax, cmin, cmax = bbox(view_result.entropy_map[ED])
    ins.imshow(view_result.entropy_map[ED, rmin:rmax, cmin:cmax])

    ins = ax_es_img.inset_axes([0.7, 0.7, 0.3, 0.3])
    ins.set_axis_off()
    rmin, rmax, cmin, cmax = bbox(view_result.entropy_map[ES])
    ins.imshow(view_result.entropy_map[ES, rmin:rmax, cmin:cmax])



    if view_result.contour is not None and use_contour:
        ax_ed_img.scatter(view_result.mu[ED, :, 0], view_result.mu[ED, :, 1], c='r', s=5)
        ax_es_img.scatter(view_result.mu[ES, :, 0], view_result.mu[ES, :, 1], c='r', s=5)

        ax_ed_img.scatter(view_result.contour[ED, :, 0], view_result.contour[ED, :, 1], c='b', s=5)
        ax_es_img.scatter(view_result.contour[ES, :, 0], view_result.contour[ES, :, 1], c='b', s=5)

        for index in range(0, 21, 1):
            confidence_ellipse(
                view_result.mu[ED, index, 0], view_result.mu[ED, index, 1], view_result.cov[ED, index],
                ax_ed_img, n_std=2, #alpha=0.5
            )
            confidence_ellipse(
                view_result.mu[ES, index, 0], view_result.mu[ES, index, 1], view_result.cov[ES, index],
                ax_es_img, n_std=2, #alpha=0.5
            )
            # confidence_ellipse(
            #     view_result.mu[ED, index, 0], view_result.mu[ED, index, 1], view_result.pca_cov[ED, index],
            #     ax_ed_img, n_std=2, alpha=0.5, edgecolor='blue'
            # )
            # confidence_ellipse(
            #     view_result.mu[ES, index, 0], view_result.mu[ES, index, 1], view_result.pca_cov[ES, index],
            #     ax_es_img, n_std=2, alpha=0.5, edgecolor='blue'
            # )


    # lines = ["-", "--", "-.", ":"]
    lines = ["-"]
    linecycler = cycle(lines)
    for j in range(min(2, view_result.pred_samples.shape[1])):
        linestyle = next(linecycler)
        for k in range(min(5, view_result.pred_samples.shape[2])):
            # if len(view_result.contour_samples[ED, j]) > 21:
            #     endo_contour = contour_spline(view_result.contour_samples[ED, j, k, :21])
            #     epi_contour = contour_spline(view_result.contour_samples[ED, j, k, 21:])
            #     color = next(ax_ed_img._get_lines.prop_cycler)['color']
            #     ax_ed_img.plot(endo_contour[:, 0], endo_contour[:, 1], alpha=0.7, c=color)
            #     ax_ed_img.plot(epi_contour[:, 0], epi_contour[:, 1], alpha=0.7, c=color)
            #     # ax_ed_img.plot(sample_contour[:, 0], sample_contour[:, 1], alpha=0.7)
            # else:
            color = next(ax_ed_img._get_lines.prop_cycler)['color']
            if view_result.contour is not None and use_contour:
                endo_contour = contour_spline(view_result.contour_samples[ED, j, k, :21])
                # ax_ed_img.plot(endo_contour[:, 0], endo_contour[:, 1], linestyle, alpha=0.7, c=color)
                ax_ed_img.plot(endo_contour[:, 0], endo_contour[:, 1], linestyle, linewidth=2, c=color)
            else:
                contour = get_contour_from_mask(view_result.pred_samples[ED, j, k].squeeze().round())
                ax_ed_img.plot(contour[:, 1], contour[:, 0], linestyle, linewidth=2, c=color)

    lines = ["-", "--", "-.", ":"]
    lines = ["-"]
    linecycler = cycle(lines)
    for j in range(min(2, view_result.pred_samples.shape[1])):
        linestyle = next(linecycler)
        for k in range(min(5, view_result.pred_samples.shape[2])):
            # if len(view_result.contour_samples[ES, j]) > 21:
            #     # sample_contour = contour_spline(view_result.contour_samples[ES, j])
            #     # ax_es_img.plot(sample_contour[:, 0], sample_contour[:, 1], alpha=0.7)
            #     # ax_es_img.plot(sample_contour[:, 0], sample_contour[:, 1], alpha=0.7)
            #
            #     endo_contour = contour_spline(view_result.contour_samples[ES, j, k, :21])
            #     epi_contour = contour_spline(view_result.contour_samples[ES, j, k, 21:])
            #     color = next(ax_es_img._get_lines.prop_cycler)['color']
            #     ax_es_img.plot(endo_contour[:, 0], endo_contour[:, 1], alpha=0.7, c=color)
            #     ax_es_img.plot(epi_contour[:, 0], epi_contour[:, 1], alpha=0.7, c=color)
            # else:
            color = next(ax_es_img._get_lines.prop_cycler)['color']
            if view_result.contour is not None and use_contour:
                endo_contour = contour_spline(view_result.contour_samples[ES, j, k, :21])
                # ax_es_img.plot(endo_contour[:, 0], endo_contour[:, 1], linestyle, alpha=0.7, c=color)
                ax_es_img.plot(endo_contour[:, 0], endo_contour[:, 1], linestyle, linewidth=2, c=color)
            else:
                contour = get_contour_from_mask(view_result.pred_samples[ES, j, k].squeeze().round())
                ax_es_img.plot(contour[:, 1], contour[:, 0], linestyle, linewidth=2, c=color)

    # ax_ed_area.set_xlabel("Area")
    # ax_es_area.set_xlabel("Area")
    # ax_ed_area.xaxis.set_label_position('top')
    # # ax_ed_area.set_yticks([])  # Command for hiding y-axis
    # ax_es_area.set_yticks([])  # Command for hiding y-axis
    #
    # ax_ed_area.set_xticks([])  # Command for hiding y-axis
    # ax_es_area.set_xticks([])  # Command for hiding y-axis
    #
    # ax_fac.set_ylabel("FAC")
    # ax_fac.set_yticks([])  # Command for hiding y-axis
    # # ax_fac.set_xticks([])  # Command for hiding y-axis
    #
    # ax_gls.set_ylabel("GLS")
    # ax_gls.set_yticks([])  # Command for hiding y-axis

    ax_ed_area.set_ylabel("Area", fontsize=15)
    # ax_es_area.set_ylabel("Area (ml)")
    # ax_ed_area.xaxis.set_label_position('top')
    ax_ed_area.set_yticks([])  # Command for hiding y-axis
    ax_es_area.set_yticks([])  # Command for hiding y-axis
    ax_ed_area.tick_params(axis="x", direction="in", pad=-15)
    ax_es_area.tick_params(axis="x", direction="in", pad=-15)
    ax_ed_area.yaxis.set_label_coords(0.02, 0.5)
    ax_ed_area.xaxis.get_major_ticks()[0].label1.set_visible(False)
    ax_es_area.xaxis.get_major_ticks()[0].label1.set_visible(False)
    # ax_ed_area.set_xticks([])  # Command for hiding y-axis
    # ax_es_area.set_xticks([])  # Command for hiding y-axis

    ax_fac.set_ylabel("FAC", fontsize=15)
    ax_fac.set_yticks([])  # Command for hiding y-axis
    # ax_fac.set_xticks([])  # Command for hiding y-axis
    ax_fac.tick_params(axis="x", direction="in", pad=-15)
    ax_fac.yaxis.set_label_coords(0.01, 0.5)
    ax_fac.xaxis.get_major_ticks()[0].label1.set_visible(False)

    ax_gls.set_ylabel("GLS", fontsize=15)
    ax_gls.set_yticks([])  # Command for hiding y-axis
    ax_gls.tick_params(axis="x", direction="in", pad=-15)
    ax_gls.yaxis.set_label_coords(0.01, 0.5)
    ax_gls.xaxis.get_major_ticks()[0].label1.set_visible(False)


    reject = False
    reject = reject or plot_metric_axis(ax_ed_area, instant_df, f'{view_result.id}/{Instant.ED}', 'Area', vertical=False)
    reject = reject or plot_metric_axis(ax_es_area, instant_df, f'{view_result.id}/{Instant.ES}', 'Area', vertical=False)
    reject = reject or plot_metric_axis(ax_fac, view_df, f'{view_result.id}', 'FAC')
    reject = reject or plot_metric_axis(ax_gls, view_df, f'{view_result.id}', 'GLS')

    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.margins(0, 0)

    # plt.subplot_tool()

    # plt.show()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.1, wspace=0.1)

    name = f"{view_result.id.replace('/', '-')}_reject.png" if reject else f"{view_result.id.replace('/', '-')}.png"
    plt.savefig(output_dir / name, dpi=300)
    plt.close()


def bbox(img, pad=20):
    img = (img > 0)
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.argmax(rows), img.shape[0] - 1 - np.argmax(np.flipud(rows))
    cmin, cmax = np.argmax(cols), img.shape[1] - 1 - np.argmax(np.flipud(cols))

    h = (rmax - rmin)
    w = (cmax - cmin)

    y = rmin + h // 2
    x = cmin + w // 2

    s = max(h, w)

    rmin = y - s // 2 - pad
    rmax = y + s // 2 + pad
    cmin = x - s // 2 - pad
    cmax = x + s // 2 + pad

    return rmin, rmax, cmin, cmax