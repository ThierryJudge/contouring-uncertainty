import numpy as np
from matplotlib import pyplot as plt
from pytorch_lightning import Callback
from scipy.stats import pearsonr


class Metrics(Callback):

    def __init__(self, nb_bins: int = 10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nb_bins = nb_bins

    @classmethod
    def dataframe_to_dict(cls, df, prefix:str=''):
        dict = {}
        for index, row in df.iterrows():
            for column in df:
                dict[f"{prefix}{index}-{column}"] = row[column]
        return dict

    @classmethod
    def thresholded_correlation(cls, uncertainties, metrics, uncertainty_key, metric_key):
        uncertainty = np.array(uncertainties[uncertainty_key])
        error = np.array(metrics[metric_key])

        n = 20
        threshold_corr_u = []
        threshold_corr_e = []
        thresholds_u = np.linspace(uncertainty.min(), uncertainty.max(), n)
        thresholds_e = np.linspace(error.min(), error.max(), n)
        for i in range(n):
            # Find correlation with uncertainty threshold
            idx = np.where(uncertainty > thresholds_u[i])[0]
            if len(idx) > 1:
                corr, _ = pearsonr(uncertainty[idx], error[idx])
            else:
                corr = np.nan
            threshold_corr_u.append(corr)

            # Find correlation with metric threshold
            idx = np.where(error > thresholds_e[i])[0]
            if len(idx) > 1:
                corr, _ = pearsonr(uncertainty[idx], error[idx])
            else:
                corr = np.nan
            threshold_corr_e.append(corr)

        fig = plt.figure()
        plt.suptitle(f"{metric_key}-{uncertainty_key}")
        ax1 = fig.add_subplot(1, 1, 1)
        lns1 = ax1.plot(thresholds_u, threshold_corr_u, marker="o", label='Uncertainty threshold')
        ax1.set_ylabel("Correlation")
        ax1.set_xlabel(f'Uncertainty thresholds')
        ax2 = ax1.twiny()
        ax2.yaxis.tick_right()
        lns2 = ax2.plot(thresholds_e, threshold_corr_e, marker="o", color='r', label='Metric threshold')
        ax2.set_xlabel(f'Metric thresholds')

        leg = lns1 + lns2
        labs = [l.get_label() for l in leg]
        ax1.legend(leg, labs)

        plt.savefig(f'corr_thresholds-{metric_key}-{uncertainty_key}.png')
        plt.tight_layout()
        plt.close()

    def bland_altman_plot(self, ax, data1, data2, errorbars=None, color=None, *args, **kwargs):
        data1 = np.asarray(data1)
        data2 = np.asarray(data2)
        mean = np.mean([data1, data2], axis=0)
        diff = data1 - data2  # Difference between data1 and data2
        md = np.mean(diff)  # Mean of the difference
        sd = np.std(diff, axis=0)  # Standard deviation of the difference

        ax.scatter(mean, diff, c=color, *args, **kwargs)
        if errorbars is not None:
            ax.errorbar(mean, diff, yerr=errorbars, fmt="o", c=color)

        ax.axhline(md, color=color, linestyle='--')
        ax.axhline(md + 1.96 * sd, color=color, linestyle='--')
        ax.axhline(md - 1.96 * sd, color=color, linestyle='--')
        ax.set_xlabel("Means")
        ax.set_ylabel("Difference")
