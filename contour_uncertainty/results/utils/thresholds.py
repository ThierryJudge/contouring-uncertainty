from unittest import result
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pytorch_lightning import Callback
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import OLSInfluence
import pymannkendall as mk
from contour_uncertainty.utils.plotting import str2tex


def compute_thresholds(error_values: np.ndarray, uncertainty_values: np.ndarray, nb_bins: int = 10, filters: np.ndarray = None, ):

    if filters is not None:
        idx = np.where(filters.astype(bool))[0]
        uncertainty = uncertainty_values[idx]
        error = error_values[idx]
    else:
        uncertainty = uncertainty_values
        error = error_values

    # print(uncertainty_key, metric_key)
    # print(uncertainty.shape, error.shape)

    threshold_errors = []
    threshold_stds = []
    num_samples = []
    # thresholds = np.linspace(uncertainty.min(), uncertainty.max(), 10)

    uncertainty_sorted = np.sort(uncertainty)
    x = np.linspace(1, len(uncertainty_sorted) - 1, nb_bins, dtype=np.int)
    thresholds = uncertainty_sorted[x]
    x = x / len(uncertainty_sorted) * 100

    # x = thresholds

    for t in thresholds:
        idx = np.where(uncertainty < t)[0]
        # print(idx)
        num_samples.append(len(idx))
        if len(idx) > 1:
            threshold_errors.append(np.mean(error[idx]))
            threshold_stds.append(np.std(error[idx]))
        else:
            threshold_errors.append(np.nan)
            threshold_stds.append(np.nan)

    threshold_errors = np.array(threshold_errors)
    threshold_stds = np.array(threshold_stds)

    return x, threshold_errors, threshold_stds


def thresholded_metrics(uncertainties, metrics, uncertainty_keys, metric_keys, filename, filters=None):
    assert len(uncertainty_keys) == len(metric_keys)
    f, ax = plt.subplots(1, len(uncertainty_keys), squeeze=False)
    ax = ax.ravel()
    results = {}

    for i, (uncertainty_key, metric_key) in enumerate(zip(uncertainty_keys, metric_keys)):

        uncertainty_values = np.array(uncertainties[uncertainty_key])
        error_values = np.array(metrics[metric_key])

        x, threshold_errors, threshold_stds = compute_thresholds(error_values, uncertainty_values, filters=filters)

        monoticity = stats.spearmanr(x, threshold_errors, nan_policy='omit')[0]

        results[f"monoticity_{metric_key}-{uncertainty_key}"] = monoticity

        ax[i].plot(x, threshold_errors, marker='o')
        # ax[i].fill_between(thresholds, y1=threshold_errors - 2*threshold_stds, y2=threshold_errors + 2*threshold_stds, alpha=.5)

        # ax[i].errorbar(thresholds, threshold_errors, yerr=threshold_stds)
        # ax[i].scatter(x, threshold_errors, marker="o", c=num_samples)
        # for j in [*list(range(0, len(x), 3)), -1]:
        # if j == len(thresholds)-1 or np.round(num_samples[j], 2) != np.round(num_samples[j+1], 2):
        # ax[i].text(x[j], threshold_errors[j], f'{num_samples[j]/len(uncertainty)*100:.2f}%', fontsize=5)
        ax[i].set_title(str2tex(f"{metric_key}-{uncertainty_key}") + f'{monoticity:02f}')
        ax[i].set_ylabel(str2tex(metric_key))
        # ax[i].set_xlabel(str2tex(f'{uncertainty_key} thresholds'))
        ax[i].set_xlabel("Percentage of remaining samples")
        ax[i].invert_xaxis()

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

    return results