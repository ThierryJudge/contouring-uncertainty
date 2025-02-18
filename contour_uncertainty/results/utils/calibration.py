import numpy as np
from matplotlib import pyplot as plt
from contour_uncertainty.utils.plotting import str2tex


def compute_calibration(error: np.ndarray, uncertainty: np.ndarray, nb_bins: int = 10, filters: np.ndarray = None, ):
    if filters is not None:
        idx = np.where(filters.astype(bool))[0]
        uncertainty = uncertainty[idx]
        error = error[idx]

    bin_boundaries = np.linspace(uncertainty.min(), uncertainty.max(), nb_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = np.zeros(1)
    bins_avg_conf = []
    bins_avg_acc = []
    bins_size = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = np.greater(uncertainty, bin_lower) * np.less(uncertainty, bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = error[in_bin].mean()
            avg_confidence_in_bin = uncertainty[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            bins_avg_conf.append(avg_confidence_in_bin)
            bins_avg_acc.append(accuracy_in_bin)
            bins_size.append(in_bin.sum())

    return ece.item(), bins_avg_conf, bins_avg_acc, bins_size


def compute_adaptive_calibration(error: np.ndarray, uncertainty: np.ndarray, nb_bins: int = 10, filters: np.ndarray = None, ):
    if filters is not None:
        idx = np.where(filters.astype(bool))[0]
        uncertainty = uncertainty[idx]
        error = error[idx]

    idx = np.argsort(uncertainty)
    uncertainty = uncertainty[idx]
    error = error[idx]

    uncertainty = np.array_split(uncertainty, nb_bins)
    error = np.array_split(error, nb_bins)

    ece = np.zeros(1)
    bins_avg_conf = []
    bins_avg_acc = []
    bins_size = []
    for u, e in zip(uncertainty, error):
        # Calculated |confidence - accuracy| in each bin
        prop_in_bin = len(u) / nb_bins
        if prop_in_bin > 0:
            accuracy_in_bin = e.mean()
            avg_confidence_in_bin = u.mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            bins_avg_conf.append(avg_confidence_in_bin)
            bins_avg_acc.append(accuracy_in_bin)
            bins_size.append(len(u))

    return ece.item(), bins_avg_conf, bins_avg_acc, bins_size


def calibration(uncertainties, metrics, uncertainty_keys, metric_keys, filename, filters=None, adaptive=False):
    assert len(uncertainty_keys) == len(metric_keys)
    f, ax = plt.subplots(1, len(uncertainty_keys), squeeze=False)
    f.set_figheight(8)
    f.set_figwidth(15)
    ax = ax.ravel()

    results = {}

    for i, (uncertainty_key, metric_key) in enumerate(zip(uncertainty_keys, metric_keys)):
        uncertainty = np.array(uncertainties[uncertainty_key])
        error = np.array(metrics[metric_key])

        if adaptive:
            ece, bins_avg_conf, bins_avg_acc, bins_size = compute_adaptive_calibration(error, uncertainty, filters=filters)
        else:
            ece, bins_avg_conf, bins_avg_acc, bins_size = compute_calibration(error, uncertainty, filters=filters)
        print(bins_size)

        print(f'calibration-{metric_key}-{uncertainty_key}: {ece} ({len(uncertainty)} samples)')

        results[f'calibration-{metric_key}-{uncertainty_key}'] = ece

        data_range = [uncertainty.min(), uncertainty.max()]
        # ax[i].scatter(bins_avg_conf, bins_avg_acc, c=bins_size, s=bins_size)
        ax[i].plot(bins_avg_conf, bins_avg_acc, marker='o')

        ax2 = ax[i].twinx()
        ax2.bar(bins_avg_conf, bins_size, alpha=0.7, width=np.min(np.diff(bins_avg_conf)) / 2)

        ax[i].plot(data_range, data_range, "--", c="k", label="Perfect calibration")
        ax[i].set_title(f"ECE={ece:.3f}")
        ax[i].set_ylabel(str2tex(metric_key))
        ax[i].set_xlabel(str2tex(uncertainty_key))

    plt.suptitle('Calibration')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    return results
