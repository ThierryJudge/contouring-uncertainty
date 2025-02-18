import numpy as np

def aleatoric_epistemic_uncertainty(metric_mc):

    assert metric_mc.ndim == 2, "metric_mc should have 2 dimensions, current shape is {}".format(metric_mc.shape)

    metric_means = np.nanmean(metric_mc, axis=-1)
    metric_vars = np.nanstd(metric_mc, axis=-1)

    # print(metric_means.shape)
    # print(metric_vars.shape)

    metric_mean = np.nanmean(metric_means)
    epistemic_var = np.nanstd(metric_means)
    aleatoric_var = np.nanmean(metric_vars)
    metric_variance = epistemic_var + aleatoric_var

    return metric_mean, aleatoric_var, epistemic_var, metric_variance
