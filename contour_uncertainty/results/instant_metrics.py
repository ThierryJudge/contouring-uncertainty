from collections import defaultdict
from typing import List, Any

import numpy as np
from medpy.metric import dc, jc, hd
from scipy.stats import pearsonr
from vital.data.camus.config import Label
from vital.data.config import Tags

from contour_uncertainty.data.config import ContourTags, BatchResult
from contour_uncertainty.results.metrics import Metrics
from contour_uncertainty.results.utils.calibration import calibration
from contour_uncertainty.results.utils.correlation import compute_correlations
from contour_uncertainty.results.utils.segmentation import dice
from contour_uncertainty.results.utils.thresholds import thresholded_metrics
from contour_uncertainty.utils.clinical import metric_error
import pandas as pd
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import CometLogger


class InstantMetrics(Metrics):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_predict_epoch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: List[List[BatchResult]]
    ) -> None:

        metrics = defaultdict(list)
        uncertainties = defaultdict(list)
        ids = []
        # area, area_mean, area_std, area_gt = [], [], [], []
        filters = []
        for res in outputs[0]:
            instant_metrics = res.instant_metrics
            # filters.extend((view[ContourTags.sample_validity]).tolist())
            for i in range(res.img.shape[0]):
                ids.append(f'{res.id}-{i}')
                dices = dice(res.pred[i], res.gt[i], res.labels, all_classes=True)
                for k, v in dices.items():
                    metrics[k].append(v)
                # metrics['Jaccard'].append(jc(res.pred[i], gt))
                # metrics['Hausdorff'].append(hd(res.pred[i], gt))
                # print(instant_metrics['area'][i], instant_metrics['area_gt'][i])

                if res.mu is not None:
                    metrics['mu_L2'].append(np.linalg.norm(res.mu[i] - res.contour[i]))

                if res.mode is not None:
                    metrics['mode_L2'].append(np.linalg.norm(res.mode[i] - res.contour[i]))

                if instant_metrics is not None:
                    metrics['Area Error'].append(
                        metric_error(instant_metrics['area'][i], instant_metrics['area_gt'][i], type='absolute'))

                # area.append(instant_metrics['area'][i])
                # area_mean.append(instant_metrics['area_mean'][i])
                # area_std.append(view[ContourTags.instant_uncertainty]['sigma_area'][i])
                # area_gt.append(instant_metrics['area_gt'][i])

                for key, unc in res.instant_uncertainty.items():
                    uncertainties[key].append(unc[i])

        if isinstance(trainer.logger, CometLogger):
            data = {"metrics": metrics, "uncertainty": uncertainties, 'ids': ids, 'filters': filters}
            np.save('data_instant', data)
            trainer.logger.experiment.log_asset('data_instant.npy')

        metrics_mean = pd.DataFrame(metrics).mean().T
        trainer.logger.log_metrics(metrics_mean.to_dict())
        print("Error")
        print(metrics_mean)

        dict = {'id': ids}
        dict.update(metrics)
        dict.update(uncertainties)
        dict = pd.DataFrame(dict)
        dict.to_csv('instant_metrics.csv')

        # filters = np.array(filters)
        # print(f"Number of filtered instants: {np.sum(1 * ~filters)} on {len(filters)} samples")

        correlations = compute_correlations(uncertainties, metrics, title='Instant Metrics Correlation',
                                            ids=np.array(ids), filename='correlation_instant.png')

        # correlation_filtered = compute_correlations(uncertainties, metrics, title='Instant Metrics Correlation',
        #                                             ids=np.array(ids), filename='correlation_instant_filtered.png',
        #                                             filters=filters)

        correlations = self.dataframe_to_dict(correlations, 'corr-')
        # correlation_filtered = self.dataframe_to_dict(correlation_filtered, 'corr-')
        # try:
        #     correlations['corr-sigma_area-Area_Error'] = correlation_filtered['corr-sigma_area-Area_Error']
        # except Exception as e:
        #     print("COPY of sigma_area-Area_Error failed")
        #     print(e)

        trainer.logger.log_metrics(correlations)

        # results = thresholded_metrics(uncertainties, metrics,
        #                               ['cov_projection', 'cov_det_mean', 'cov_projection', 'cov_det_mean'],
        #                               ['Dice', 'Dice', 'Average L2', 'Average L2'],
        #                               filename='thresholds_instants.png')
        # trainer.logger.log_metrics(results)

        # results = calibration(uncertainties, metrics,
        #                       ['sigma_area'],
        #                       ['Area Error'], filename='calibration_instants.png', filters=filters)
        # trainer.logger.log_metrics(results)
