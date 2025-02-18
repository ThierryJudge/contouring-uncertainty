from typing import List

import numpy as np
from matplotlib import pyplot as plt

from contour_uncertainty.data.config import BatchResult, LV_example_shape, LV_MYO_example_shape, XRAY_example_shape
from contour_uncertainty.results.metrics import Metrics


class Skewness(Metrics):
    def on_predict_epoch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: List[List[BatchResult]]
    ) -> None:

        if outputs[0][0].mu is None:
            print(f"NO POINT SKIPPING {self.__class__.__name__}")
            return

        point_errors = []
        average_skewness = []

        for res in outputs[0]:
            for i in range(res.img.shape[0]):
                point_errors.append(res.contour[i] - res.mu[i])
                if res.alpha is not None:
                    average_skewness.append(res.alpha[i])


        point_errors = np.array(point_errors)
        average_skewness = np.array(average_skewness)

        np.save('skewness.npy', {'errors': point_errors,
                                 'average_skew': average_skewness})

        if point_errors.shape[1] == 21:
            example_shape = LV_example_shape
        elif point_errors.shape[1] == 42:
            example_shape = LV_MYO_example_shape
        else:
            example_shape = XRAY_example_shape

        ### Error skewness ### 
        f, ax = plt.subplots()
        ax.set_xlim([0, 256])
        ax.set_ylim([256, 0])
    
        for i in range(point_errors.shape[1]):
            ax.scatter(example_shape[i, 0] + point_errors[:, i, 0],
                       example_shape[i, 1] + point_errors[:, i, 1], alpha=0.5, s=5)

        ax.scatter(example_shape[:, 0], example_shape[:, 1], c='k')
        plt.savefig(f'skewness_error.png')
        plt.close()

        ### Error skewness ### 
        if len(average_skewness) > 0:
            average_skewness = np.array(average_skewness).mean(0)
            # average_skewness = average_skewness / np.linalg.norm(average_skewness, axis=-1, keepdims=True)
            scale = 25
            print(average_skewness.shape)
            f, ax = plt.subplots()
            ax.set_xlim([0, 256])
            ax.set_ylim([256, 0])
        
            plt.quiver(example_shape[:, 0], example_shape[:, 1], average_skewness[:,0], average_skewness[:,1], scale=scale)
            ax.scatter(example_shape[:, 0], example_shape[:, 1], c='k', s=5, marker='x')
            plt.savefig(f'skewness_pred.png')
            plt.close()

            f, ax = plt.subplots()
            ax.set_xlim([0, 256])
            ax.set_ylim([256, 0])
        
            for i in range(point_errors.shape[1]):
                ax.scatter(example_shape[i, 0] + point_errors[:, i, 0],
                        example_shape[i, 1] + point_errors[:, i, 1], alpha=0.5, s=5)
                
            plt.quiver(example_shape[:, 0], example_shape[:, 1], average_skewness[:,0], average_skewness[:,1], scale=scale)
            ax.scatter(example_shape[:, 0], example_shape[:, 1], c='k', marker='x')
            plt.savefig(f'skewness_error_pred.png')
            plt.close()



