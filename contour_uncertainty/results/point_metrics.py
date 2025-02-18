from collections import defaultdict
from typing import List, Any

import numpy as np
from scipy.stats import pearsonr
from vital.data.config import Tags
import seaborn as sns
from contour_uncertainty.data.config import ContourTags, LV_example_shape
from contour_uncertainty.results.metrics import Metrics
import pandas as pd
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import CometLogger

from contour_uncertainty.results.utils.calibration import calibration
from contour_uncertainty.results.utils.correlation import compute_correlations
from contour_uncertainty.results.utils.thresholds import thresholded_metrics
from contour_uncertainty.data.config import ContourTags, BatchResult

class PointMetrics(Metrics):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_predict_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: List[List[BatchResult]]) -> None:
        metrics = defaultdict(list)
        uncertainties = defaultdict(list)

        errors = []
        determinants = []

        for res in outputs[0]:
            for i in range(res.img.shape[0]):

                contour_pred = res.mu[i]
                contour_mode = res.mode[i]
                post_contour_pred = res.post_mu[i]
                contour_gt = res.contour[i]

                metrics['X-Error'].extend(np.abs(contour_pred[:, 0] - contour_gt[:, 0]).flatten().tolist())
                metrics['Y-Error'].extend(np.abs(contour_pred[:, 1] - contour_gt[:, 1]).flatten().tolist())
                metrics['Error'].extend(np.sqrt(np.sum((contour_pred - contour_gt) ** 2, axis=1)).flatten().tolist())

                metrics['mode_X-Error'].extend(np.abs(contour_mode[:, 0] - contour_gt[:, 0]).flatten().tolist())
                metrics['mode_Y-Error'].extend(np.abs(contour_mode[:, 1] - contour_gt[:, 1]).flatten().tolist())
                metrics['mode_Error'].extend(np.sqrt(np.sum((contour_mode - contour_gt) ** 2, axis=1)).flatten().tolist())

                metrics['post_X-Error'].extend(np.abs(post_contour_pred[:, 0] - contour_gt[:, 0]).flatten().tolist())
                metrics['post_Y-Error'].extend(np.abs(post_contour_pred[:, 1] - contour_gt[:, 1]).flatten().tolist())
                metrics['post_Error'].extend(np.sqrt(np.sum((post_contour_pred - contour_gt) ** 2, axis=1)).flatten().tolist())

                errors.append(np.sqrt(np.sum((contour_pred - contour_gt) ** 2, axis=1)))
                determinants.append(res.point_uncertainty['cov_det'][i])


                for key, unc in res.point_uncertainty.items():
                    uncertainties[key].extend(unc[i].flatten().tolist())

        if isinstance(trainer.logger, CometLogger):
            data = {"metrics": metrics, "uncertainty": uncertainties}
            np.save('data_point', data)
            trainer.logger.experiment.log_asset('data_point.npy')

        correlations = compute_correlations(uncertainties, metrics, title='Point Metrics Correlation',
                                                 filename='correlation_point.png')

        trainer.logger.log_metrics(self.dataframe_to_dict(correlations, 'corr-'))

        metrics_mean = pd.DataFrame(metrics).mean().T
        trainer.logger.log_metrics(metrics_mean.to_dict())
        print("Error")
        print(metrics_mean)

        errors = np.array(errors).mean(0)
        determinants = np.array(determinants).mean(0)

        corr, _ = pearsonr(determinants, errors)
        print('Average distance vs average determinant correlation: ', corr)
        trainer.logger.log_metrics({'avg_cov-avg_det': corr})

        f, ax = plt.subplots(1, 1)
        ax.scatter(errors, determinants)
        ax.set_xlabel('Distance', fontsize=20)
        ax.set_ylabel('Average determinant', fontsize=20)
        ax.set_title(f"R={corr:.3f}", fontsize=20)
        plt.savefig("distance-error-plot.png", dpi=100)
        plt.close()

        f, ax = plt.subplots(1, 1)
        ax.scatter(LV_example_shape[:, 0], LV_example_shape[:, 1], s=10, c="r")
        ax.set_title(f"Average Error / Average determinant, R={corr:.3f}")
        ax.set_xlim([0, 256])
        ax.set_ylim([256, 0])
        for k in range(0, LV_example_shape.shape[0], 1):
            ax.text(LV_example_shape[k, 0], LV_example_shape[k, 1], f"{errors[k]:.2f}/{determinants[k]:.2f}")
        plt.savefig("average_distance.png", dpi=100)
        plt.close()

        # self.calibration(uncertainties, metrics, 'cov_xx', 'X-Error')
        # self.calibration(uncertainties, metrics, 'cov_yy', 'Y-Error')
        # self.calibration(uncertainties, metrics, 'cov_det', 'Error')

        results = calibration(uncertainties, metrics, ['cov_xx', 'cov_yy', 'cov_det', 'cov_eigval_sum'],
                              ['X-Error', 'Y-Error', 'Error', 'Error'], filename='calibration_points.png',
                              adaptive=True)
        trainer.logger.log_metrics(results)
        results = calibration(uncertainties, metrics, ['post_cov_xx', 'post_cov_yy', 'post_cov_det', 'post_cov_eigval_sum'],
                              ['X-Error', 'Y-Error', 'Error', 'Error'], filename='post_calibration_points1.png',
                              adaptive=True)
        results = calibration(uncertainties, metrics, ['post_cov_xx', 'post_cov_yy', 'post_cov_det', 'post_cov_eigval_sum'],
                              ['post_X-Error', 'post_Y-Error', 'post_Error', 'post_Error'], filename='post_calibration_points2.png',
                              adaptive=True)


        results = thresholded_metrics(uncertainties, metrics, ['cov_xx', 'cov_yy', 'cov_det'],
                                      ['X-Error', 'Y-Error', 'Error'], filename='thresholds_points.png')
        trainer.logger.log_metrics(results)

        # self.thresholded_metrics(uncertainties, metrics, 'cov_xx', 'X-Error')
        # self.thresholded_metrics(uncertainties, metrics, 'cov_yy', 'Y-Error')
        # self.thresholded_metrics(uncertainties, metrics, 'cov_det', 'Error')

        # self.thresholded_correlation(uncertainties, metrics, 'cov_xx', 'X-Error')
        # self.thresholded_correlation(uncertainties, metrics, 'cov_yy', 'Y-Error')
        # self.thresholded_correlation(uncertainties, metrics, 'cov_det', 'Error')
