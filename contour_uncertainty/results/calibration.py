from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from pytorch_lightning.loggers import CometLogger

from contour_uncertainty.data.config import BatchResult
from vital.data.camus.config import Label
from contour_uncertainty.results.metrics import Metrics


class Calibration(Metrics):
    """Abstract calibration evaluator.

    Args:
        nb_bins: number of bin for the calibration computation.
    """

    CALIBRATION_FILE_NAME: str = None

    def __init__(self, nb_bins: int = 20, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nb_bins = nb_bins
        bin_boundaries = np.linspace(0, 1, nb_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def on_predict_epoch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: List[List[BatchResult]]
    ) -> None:

        try:
            umap_confidences, umap_accuracies = self.get_confidences_and_accuracies(outputs[0],
                                                                                    prob_string='uncertainty_map')
            umap_ece, umap_mce = self.ece(umap_confidences, umap_accuracies, name='uncertainty_map')
        except Exception as e:
            print(e)
            umap_ece = -1
            umap_mce = -1

        try:
            confidences, accuracies, preds, gts = self.get_pixel_confidences_and_accuracies(outputs[0], prob_string='entropy_map')
            fg = preds + gts
            not_bg = fg != 0  # Background class is always 0
            entropy_confidences = confidences[not_bg]
            entropy_accuracies = accuracies[not_bg]
            entropy_ece, entropy_mce = self.ece(entropy_confidences, entropy_accuracies, name='entropy_map', trainer=trainer)

            samples_eces = []
            samples_mces = []
            for i in range(confidences.shape[0]):
                fg = preds[i] + gts[i]
                not_bg = fg != 0  # Background class is always 0
                entropy_confidences = confidences[i][not_bg]
                entropy_accuracies = accuracies[i][not_bg]

                if i < 10:
                    sample_ece, sample_mce = self.ece(entropy_confidences, entropy_accuracies, f'entropy_map_{i}')
                else:
                    sample_ece, sample_mce = self.ece(entropy_confidences, entropy_accuracies, name=None)
                samples_eces.append(sample_ece)
                samples_mces.append(sample_mce)
        except Exception as e:
            print(e)
            entropy_ece = -1
            entropy_mce = -1
        try:
            entropy_confidences, entropy_accuracies = self.get_confidences_and_accuracies(outputs[0],
                                                                                          prob_string='entropy_map')
            entropy_ace, entropy_amce = self.ace(entropy_confidences, entropy_accuracies, name='entropy_map')

        except Exception as e:
            print(e)
            entropy_ace = -1
            entropy_amce = -1

        results = {f"ece": float(umap_ece),
                   'entropy_ece': float(entropy_ece),
                   'entropy_aece': float(entropy_ace),
                   f"mce": float(umap_mce),
                   'entropy_mce': float(entropy_mce),
                   'entropy_amce': float(entropy_amce),
                   'sample_entropy_ece': float(np.mean(samples_eces)),
                   'sample_entropy_mce': float(np.mean(samples_mces)),
                   }
        trainer.logger.log_metrics(results)
        print(results)

    def ece(self, confidences, accuracies, name, trainer=None):
        ece = np.zeros(1)
        bins_avg_conf = []
        bins_avg_acc = []
        prob_in_bins = []
        bins_size = []
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = np.greater(confidences, bin_lower) * np.less(confidences, bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin.item() > 0:
                prob_in_bins.append(prop_in_bin)
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

                bins_avg_conf.append(avg_confidence_in_bin)
                bins_avg_acc.append(accuracy_in_bin)
                bins_size.append(in_bin.sum())

        bins_avg_conf = np.array(bins_avg_conf)
        bins_avg_acc = np.array(bins_avg_acc)

        mce = np.max(np.abs(bins_avg_conf - bins_avg_acc))

        if name is not None:
            np.save(f"{self.__class__.__name__}-{name}.npy", {"conf": bins_avg_conf, "acc": bins_avg_acc})

            plt.figure()
            plt.hist(confidences)
            plt.savefig(f"hist.png", dpi=100)
            plt.close()


            f, ax = plt.subplots(1,1)
            ax.plot(bins_avg_conf, bins_avg_acc)
            ax.plot([0, 1], [0, 1], "--", c="k", label="Perfect calibration")
            ax.set_title(f"{self.__class__.__name__}")
            ax.set_xlabel("Confidence")
            ax.set_ylabel("Accuracy")

            ax2 = ax.twinx()
            # ax2.set_yscale('log')
            try:
                ax2.bar(bins_avg_conf, bins_size, alpha=0.7, width=np.min(np.diff(bins_avg_conf)) / 2)
            except:
                ax2.bar(bins_avg_conf, bins_size, alpha=0.7)


            plt.savefig(f"{self.__class__.__name__}-{name}-ece.png", dpi=100)
            plt.close()

        if trainer is not None:
            if isinstance(trainer.logger, CometLogger):
                trainer.logger.experiment.log_curve(f"entropy-ece", bins_avg_conf.tolist(), bins_avg_acc.tolist())

        return ece, mce

    def ace(self, confidences, accuracies, name):

        idx = np.argsort(confidences)
        confidences = confidences[idx]
        accuracies = accuracies[idx]

        confidences = np.array_split(confidences, self.nb_bins)
        accuracies = np.array_split(accuracies, self.nb_bins)

        ece = np.zeros(1)
        bins_avg_conf = []
        bins_avg_acc = []
        prob_in_bins = []
        for c, a in zip(confidences, accuracies):
            # Calculated |confidence - accuracy| in each bin
            prop_in_bin = len(c) / self.nb_bins

            if prop_in_bin > 0:
                prob_in_bins.append(prop_in_bin)
                accuracy_in_bin = a.mean()
                avg_confidence_in_bin = c.mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

                bins_avg_conf.append(avg_confidence_in_bin)
                bins_avg_acc.append(accuracy_in_bin)

        bins_avg_conf = np.array(bins_avg_conf)
        bins_avg_acc = np.array(bins_avg_acc)

        mce = np.max(np.abs(bins_avg_conf - bins_avg_acc))

        np.save(f"{self.__class__.__name__}-{name}-ace.npy", {"conf": bins_avg_conf, "acc": bins_avg_acc})

        plt.figure()
        plt.plot(bins_avg_conf, bins_avg_acc)
        plt.plot([0, 1], [0, 1], "--", c="k", label="Perfect calibration")
        plt.title(f"{self.__class__.__name__}")
        plt.xlabel("Confidence")
        plt.ylabel("Accuracy")
        plt.savefig(f"{self.__class__.__name__}-{name}-ace.png", dpi=100)
        plt.close()

        return ece, mce

    def get_confidences_and_accuracies(self, results: List[BatchResult], prob_string) -> Tuple[np.ndarray, np.ndarray]:
        """Computes confidences and accuracies for all patients with respect pixels.

        Args:
            results: List of patient results including image, prediction, groundtruth and uncertainty prediction.

        Returns:
            average confidences and accuracies for each bin.
        """
        confidences, accuracies, preds, gts = self.get_pixel_confidences_and_accuracies(results, prob_string)

        fg = preds + gts
        not_bg = fg != 0  # Background class is always 0
        confidences = confidences[not_bg]
        accuracies = accuracies[not_bg]

        return confidences, accuracies

    @staticmethod
    def get_pixel_confidences_and_accuracies(results: List[BatchResult], prob_string) -> Tuple:
        """Computes confidences and accuracies for all patients with respect pixels.

        Args:
            results: List of patient results including image, prediction, groundtruth and uncertainty prediction.

        Returns:
            average confidences and accuracies for each bin.
        """
        confidences = []
        accuracies = []
        preds = []
        gts = []
        for res in results:
            correct_map = np.equal(res.pred, res.gt).astype(int)
            # Flip uncertainties to get confidence scores
            confidences.append(1 - getattr(res, prob_string))
            accuracies.append(correct_map)
            preds.append(res.pred)
            gts.append(res.gt)

        confidences = np.array(confidences)
        accuracies = np.array(accuracies)
        preds = np.array(preds)
        gts = np.array(gts)

        confidences = confidences.reshape(-1, *confidences.shape[-2:])
        accuracies = accuracies.reshape(-1, *accuracies.shape[-2:])
        preds = preds.reshape(-1, *preds.shape[-2:])
        gts = gts.reshape(-1, *gts.shape[-2:])

        return confidences, accuracies, preds, gts
