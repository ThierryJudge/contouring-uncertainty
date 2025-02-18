from typing import List

import numpy as np
import pandas as pd

from contour_uncertainty.results.clinical.utils import aleatoric_epistemic_uncertainty
from vital.utils.format.native import prefix

from contour_uncertainty.data.config import BatchResult
from contour_uncertainty.utils.clinical import metric_error, lv_area


class InstantMetric:
    PREFIX: str = ""

    def compute(self, view: BatchResult, instant_key: str, instant: int):
        raise NotImplementedError

    def __call__(self, view_results: List[BatchResult]) -> pd.DataFrame:
        res = {}
        for view in view_results:
            for instant_key, instant in view.instants.items():
                res[f'{view.id}/{instant_key}'] = prefix(self.compute(view, instant_key, instant), self.PREFIX)

        return pd.DataFrame(res).T


class AreaError(InstantMetric):
    PREFIX = 'Area_'

    def compute(self, view: BatchResult, instant_key: str, instant: int):
        voxelspacing = np.prod(view.voxelspacing)

        area_pred = lv_area(view.pred[instant]) * voxelspacing
        area_gt = lv_area(view.gt[instant]) * voxelspacing
        Te = view.pred_samples.shape[1]
        Ta = view.pred_samples.shape[2]

        # print(view.pred_samples.shape)

        area_mc = [[lv_area(view.pred_samples[instant, j, i]) for i in range(Ta)] for j in range(Te)]
        area_mc = np.array(area_mc) * voxelspacing

        metric_mean, aleatoric_var, epistemic_var, metric_variance = aleatoric_epistemic_uncertainty(area_mc)
        # error = metric_error(area_pred, area_gt)
        error = metric_error(metric_mean, area_gt)

        return {
            'pred': area_pred,
            'gt': area_gt,
            'error': error,
            'mc': area_mc.tolist(),
            'std': metric_variance,
            'mean': metric_mean,
            'aleatoric_std': aleatoric_var,
            'epistemic_std': epistemic_var,
        }