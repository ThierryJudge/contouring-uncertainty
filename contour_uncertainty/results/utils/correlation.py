from glob import escape
import numpy as np
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import OLSInfluence

from contour_uncertainty.utils.plotting import str2tex


def compute_correlations(uncertainties, metrics, title=None, ids=None, filename=None, filters=None):
    # print(metrics.keys())
    # print(uncertainties.keys())
    f, ax = plt.subplots(len(metrics.keys()), len(uncertainties), squeeze=False)
    f.set_figheight(15)
    f.set_figwidth(15)
    correlations = {}
    if title:
        f.suptitle(title)

    for i, (uncertainty_key, uncertainty_values) in enumerate(uncertainties.items()):
        corrs = {}
        for j, (metric_key, metric_values) in enumerate(metrics.items()):
            # print(uncertainty_key, metric_key, len(uncertainty), len(metrics))
            metric_key = metric_key.replace('-', '_')
            metric_key = metric_key.replace(' ', '_')

            uncertainty_values = np.array(uncertainty_values, copy=True)
            metric_values = np.array(metric_values, copy=True)
            if filters is not None:
                idx = np.where(filters.astype(bool))[0]
                uncertainty = uncertainty_values[idx]
                metric = metric_values[idx]
                id = ids[idx]
            else:
                uncertainty = uncertainty_values
                metric = metric_values
                id = ids

            try: 
                corr, _ = pearsonr(uncertainty, metric)
            except Exception as e:
                print("ERROR WITH: pearsonr(uncertainty, metric)")
                print(metric_key, uncertainty_key)
                print(uncertainty.shape)
                print(metric.shape)
                print(uncertainty)
                print(metric)
                print(e)
                exit(1)

            # corr, _ = spearmanr(uncertainty, metric)
            data = {metric_key: metric, uncertainty_key: uncertainty}

            f = f'{metric_key} ~ {uncertainty_key}'
            model = ols(formula=f, data=data).fit()
            cook_distance = OLSInfluence(model).cooks_distance
            (distance, p_value) = cook_distance

            threshold = 4 / len(metric)
            # the observations with Cook's distances higher than the threshold value are labeled in the plot
            # influencial_data = distance[distance > threshold]
            # print(influencial_data.keys())
            # print(influencial_data[list(influencial_data.keys())])
            # print(influencial_data[list(influencial_data.keys())])
            if ids is not None:
                # print(f"{metric_key}, {uncertainty_key} influencial data {ids[list(influencial_data.keys())]}")
                indices = (-distance).argsort()[:8]
                for idx in indices:
                    ax[j, i].text(metric[idx], uncertainty[idx], str(id[idx]), fontsize=5)

            corrs[metric_key] = corr
            sns.regplot(x=metric_key, y=uncertainty_key, data=data, ax=ax[j, i])
            sns.scatterplot(x=metric_key, y=uncertainty_key, data=data, ax=ax[j, i], hue=distance, size=distance,
                            sizes=(50, 200), edgecolor='black', linewidth=1, legend=None)
            # sns.scatterplot(data[metric_key], data[uncertainty_key], hue=distance, size=distance, sizes=(50, 200),
            #                 edgecolor='black', linewidth=1, ax=ax[j, i])

            ax[j, i].set_xlabel(str2tex(metric_key), fontsize=20)
            ax[j, i].set_ylabel(str2tex(uncertainty_key), fontsize=20)
            ax[j, i].set_title(f"R={corr:.3f}", fontsize=20)

            if filters is not None:
                idx = np.where(~filters.astype(bool))[0]
                rejected_uncertainty_values = uncertainty_values[idx]
                rejected_metric_values = metric_values[idx]
                ax[j, i].scatter(rejected_metric_values, rejected_uncertainty_values, marker='*')
                # if ids is not None:
                #     for k in idx:
                #         ax[j, i].text(metric_values[k], uncertainty_values[k], str(ids[k]), fontsize=5)

        correlations[uncertainty_key] = corrs

    # print(correlations)
    correlations = pd.DataFrame(data=correlations).T
    print(correlations)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300)

    return correlations