import random
from pathlib import Path
from typing import Dict, Any, Union, List

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.utils.data as torchdata
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger
from torch import Tensor

from contour_uncertainty.data.config import BatchResult
from contour_uncertainty.utils.mcdropout import patch_module
from contour_uncertainty.utils.metrics import Dice
from vital.data.camus.config import CamusTags
from vital.data.config import Subset, Tags
from vital.tasks.generic import SharedStepsTask
from vital.utils.format.native import prefix
from vital.utils.saving import resolve_model_checkpoint_path


def identity(x):
    return x


class UncertaintyTask(SharedStepsTask):
    """ Default uncertainty task.
    This class handles boilerplate code common to all methods.
    """
    def __init__(
            self,
            t_a: int = 1,
            t_e: int = 1,
            train_ensemble: bool = False,
            ensemble_ckpt: Union[Path, List] = None,
            *args, **kwargs):
        """Initializes the metric objects used repeatedly in the train/eval loop.

        Args:
            t_a (int): Number of aleatoric samples to draw
            t_e (int): Number of epistemic samples to draw. Will be overridden by len(ensemble_ckpt)
            train_ensemble (bool): Sets the train set size to random split of 90%.
            ensemble_ckpt: Path to the ensemble checkpoint.
            *args: Positional arguments to pass to the parent's constructor.
            **kwargs: Keyword arguments to pass to the parent's constructor.
        """
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.dice = Dice(labels=self.hparams.data_params.labels)
        self.model = self.configure_model()
        self.is_val_step = False

        if ensemble_ckpt is not None:
            print("Loading ensemble checkpoint")
            self.ensembling = True

            if isinstance(ensemble_ckpt, List):
                self.model = nn.ModuleList()
                for w in ensemble_ckpt:
                    self.model.append(self.load_from_checkpoint(resolve_model_checkpoint_path(w)).model)
            elif Path(ensemble_ckpt).is_dir():
                self.model = nn.ModuleList()
                for filename in (Path(ensemble_ckpt).glob('*.ckpt')):
                    self.model.append(self.load_from_checkpoint(resolve_model_checkpoint_path(filename)).model)
            else:
                raise ValueError("ENSEMBLE not valid")

            print("Len(ensemble_ckpt): ", len(self.model))
            self.hparams.t_e = len(self.model)
        else:
            self.ensembling = False
            if self.hparams.t_e > 1:
                # Keep dropout at test time.
                self.model = patch_module(self.model)

    def on_fit_start(self) -> None:
        if self.hparams.train_ensemble:
            train_set = self.trainer.datamodule._dataset[Subset.TRAIN]
            indices = random.sample(range(len(train_set)), int(0.9 * len(train_set)))
            self.trainer.datamodule._dataset[Subset.TRAIN] = torchdata.Subset(train_set, indices)

    def on_fit_end(self) -> None:
        self.trainer.logger.log_hyperparams({'train_complete': True})

    def forward(self, *args, **kwargs):  # noqa: D102
        return self.model(*args, **kwargs)

    def validation_step(self, *args, **kwargs) -> Dict[str, Tensor]:  # noqa: D102
        self.is_val_step = True
        result = prefix(self._shared_step(*args, **kwargs), "val/")
        self.is_val_step = False
        self.log_dict(result, **self.hparams.val_log_kwargs)
        self.log('val_loss', result['val/loss'], on_step=True, on_epoch=True, prog_bar=True, logger=False)
        return result

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> BatchResult:
        raise NotImplementedError

    def upload_fig(self, fig, title: str):
        if isinstance(self.trainer.logger, TensorBoardLogger):
            self.trainer.logger.experiment.add_figure(title, fig, self.current_epoch)
        if isinstance(self.trainer.logger, CometLogger):
            self.trainer.logger.experiment.log_figure(title, fig, step=self.current_epoch)

    @staticmethod
    def sample_entropy(samples):
        """
            samples: (T_e, T_a, C, H, W)
        """
        samples = torch.tensor(samples)

        activate_fn = identity

        if samples.ndim == 5:
            samples = samples.reshape(-1, *samples.shape[2:])

        probs = [activate_fn(samples[i]).detach() for i in range(samples.shape[0])]
        probs = torch.stack(probs, dim=0)
        y_hat = probs.mean(0)

        if samples.shape[1] == 1:
            y_hat = torch.cat([y_hat, 1 - y_hat], dim=0)
            base = 2
        else:
            base = samples.shape[1]

        uncertainty_map = scipy.stats.entropy(y_hat.cpu().numpy(), axis=0, base=base)

        uncertainty_map[~np.isfinite(uncertainty_map)] = 0


        return uncertainty_map


    @staticmethod
    def get_voxelspacing_and_instants(batch):
        if CamusTags.metadata in batch.keys():
            instants = batch[CamusTags.metadata].instants
            voxelspacing = np.array(batch[CamusTags.metadata].voxelspacing)
            if batch[CamusTags.metadata].gt is not None:
                voxelspacing = voxelspacing * batch[CamusTags.metadata].gt.shape / batch[Tags.gt].shape
            voxelspacing = voxelspacing[1:]
        else:
            instants = None
            voxelspacing = None
        return voxelspacing, instants

