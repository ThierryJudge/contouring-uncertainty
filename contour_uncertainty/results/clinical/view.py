from random import sample
from typing import List

import numpy as np
import pandas as pd

from contour_uncertainty.results.clinical.utils import aleatoric_epistemic_uncertainty
from vital.data.camus.config import Label, Instant
from vital.utils.format.native import prefix
from vital.utils.image.us.measure import EchoMeasure

from contour_uncertainty.data.config import BatchResult
from contour_uncertainty.utils.clinical import metric_error, lv_FAC, global_longitudinal_strain


class ViewMetric:
    PREFIX: str = ""

    def compute(self, view: BatchResult):
        raise NotImplementedError

    def __call__(self, view_results: List[BatchResult]) -> pd.DataFrame:
        res = {}
        for view in view_results:
            res[view.id] = prefix(self.compute(view), self.PREFIX)

        return pd.DataFrame(res).T


class FAC(ViewMetric):
    PREFIX = 'FAC_'
    MIN_VALUE = 0
    MAX_VALUE = 1

    def compute(self, view: BatchResult):
        instants = view.instants
        T = view.pred_samples.shape[1]
        pred = lv_FAC(view.pred[instants[Instant.ED]], view.pred[instants[Instant.ES]])
        gt = lv_FAC(view.gt[instants[Instant.ED]], view.gt[instants[Instant.ES]])

        Te = view.pred_samples.shape[1]
        Ta = view.pred_samples.shape[2]

        ED = instants[Instant.ED]
        ES = instants[Instant.ES]

        mc = [[lv_FAC(view.pred_samples[ED, j, i], view.pred_samples[ES, j, i]) for i in range(Ta)] for j in range(Te)]
        mc = np.array(mc)  # (Te, Ta)

        sample_reject = np.logical_or(mc < self.MIN_VALUE, mc > self.MAX_VALUE)
        mc[sample_reject] = np.nan

        metric_mean, aleatoric_var, epistemic_var, metric_variance = aleatoric_epistemic_uncertainty(mc)

        # reject = not (self.MIN_VALUE < metric_mean <= self.MAX_VALUE)
        reject = not (self.MIN_VALUE < pred <= self.MAX_VALUE)
        # print('FAC', pred, metric_mean)
        if np.sum(sample_reject) / np.size(sample_reject) > 0.5:
            reject = True
        # print('reject', reject,     np.sum(sample_reject) / np.size(sample_reject))

        # error = metric_error(pred, gt)
        error = metric_error(metric_mean, gt)

        # reject = False

        return {
            'pred': pred,
            'gt': gt,
            'error': error,
            'mc': mc.tolist(),
            'std': metric_variance,
            'mean': metric_mean,
            'aleatoric_std': aleatoric_var,
            'epistemic_std': epistemic_var,
            'reject': reject,
            'sample_reject': sample_reject
        }


class GLS(ViewMetric):
    PREFIX = "GLS_"
    MIN_VALUE = 0
    MAX_VALUE = 1

    @classmethod
    def gls(cls, segmentation, view: BatchResult):
        # print(segmentation.shape)
        instants = view.instants
        myo_label = Label.MYO if Label.MYO in view.labels else None
        try:
            # print(EchoMeasure.gls(segmentation, Label.LV, myo_label))
            # print(EchoMeasure.gls(segmentation, Label.LV, myo_label)[instants[Instant.ES]])
            return EchoMeasure.gls(segmentation, Label.LV, myo_label)[instants[Instant.ES]]
        except Exception as e:
            return np.nan

    def compute(self, view: BatchResult):
        instants = view.instants
        # T = view.pred_samples.shape[1]

        Te = view.pred_samples.shape[1]
        Ta = view.pred_samples.shape[2]

        ED = instants[Instant.ED]
        ES = instants[Instant.ES]

        if view.contour is not None:
            contour_pred = global_longitudinal_strain(view.mu[ED], view.mu[ES])
            coutour_gt = global_longitudinal_strain(view.contour[ED], view.contour[ES])

            # coutour_mc = np.array(
            #     [global_longitudinal_strain(view.contour_samples[instants[Instant.ED], t],
            #                                 view.contour_samples[instants[Instant.ES], t]) for t in
            #      range(T)])

            mc = [[
                global_longitudinal_strain(view.pred_samples[ED, j, i], view.pred_samples[ES, j, i])
                for i in range(Ta)]
                for j in range(Te)]
            mc = np.array(mc)  # (Te, Ta)

            metric_mean, aleatoric_var, epistemic_var, metric_variance = aleatoric_epistemic_uncertainty(mc)

            # error = metric_error(contour_pred, coutour_gt)
            error = metric_error(metric_mean, coutour_gt)

            output = {
                'contour_pred': contour_pred,
                'contour_gt': coutour_gt,
                'contour_error': error,
                'contour_mc': mc.tolist(),
                'contour_std': metric_variance,
                'contour_mean': metric_mean,
                'contour_aleatoric_std': aleatoric_var,
                'contour_epistemic_std': epistemic_var,
            }
        else:
            output = {}

        pred = self.gls(view.pred, view) / -100
        gt = self.gls(view.gt, view) / -100

        # mc = [[
        #     global_longitudinal_strain(view.pred_samples[ED, j, i], view.pred_samples[ES, j, i])
        #     for i in range(Ta)]
        #     for j in range(Te)]
        # mc = np.array(mc)  # (Te, Ta)

        # mc = np.array([self.gls(view.pred_samples[:, t], view) for t in range(T)])

        mc = [[self.gls(view.pred_samples[:, j, i], view) for i in range(Ta)] for j in range(Te)]
        mc = np.array(mc) / -100  # (Te, Ta)
        mc[np.isinf(mc)] = np.nan

        sample_reject = np.logical_or(mc < self.MIN_VALUE, mc > self.MAX_VALUE)
        mc[sample_reject] = np.nan

        metric_mean, aleatoric_var, epistemic_var, metric_variance = aleatoric_epistemic_uncertainty(mc)
        # print('GLS', pred, metric_mean)

        # mc = np.array([[global_longitudinal_strain(view.pred_samples[instants[Instant.ED], i], view.pred_samples[instants[Instant.ES], j]) for i in range(T) if i!= j] for j in range(T)]).flatten()

        reject = not (self.MIN_VALUE < pred <= self.MAX_VALUE)
        # reject = not (self.MIN_VALUE < metric_mean <= self.MAX_VALUE)
        if np.sum(sample_reject) / np.size(sample_reject) > 0.5:
            reject = True
        # print('reject', reject,     np.sum(sample_reject) / np.size(sample_reject))

        # error = metric_error(pred, gt)
        error = metric_error(metric_mean, gt)

        # reject = False

        output.update({
            'pred': pred,
            'gt': gt,
            'error': error,
            'mc': mc.tolist(),
            'std': np.nanstd(mc),
            'mean': np.nanmean(mc),
            'mc_nan_count': np.count_nonzero(np.isnan(mc)),
            'aleatoric_std': aleatoric_var,
            'epistemic_std': epistemic_var,
            'reject': reject,
            'sample_reject': sample_reject
        })

        return output
