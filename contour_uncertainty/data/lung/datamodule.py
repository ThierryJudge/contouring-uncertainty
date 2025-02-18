from pathlib import Path
from pathlib import Path
from typing import Callable, Tuple
from typing import Literal, Union, Optional
import albumentations as A
from pytorch_lightning.trainer.states import TrainerFn
from torch import Tensor
from torch.utils.data import DataLoader

from contour_uncertainty.augmentations.affine import RandomRotation, RandomTranslation
from contour_uncertainty.augmentations.augmentation import Compose
from contour_uncertainty.augmentations.brightnesscontrast import RandomBrightnessContrast
from contour_uncertainty.augmentations.gamma import RandomGamma
from contour_uncertainty.data.lung.config import Label
from contour_uncertainty.data.lung.dataset import JSRT
from vital.data.config import DataParameters, Subset
from vital.data.data_module import VitalDataModule

from contour_uncertainty.data.lung.utils import LungContourToMask, LungSkewUmap, LungUMap


class JSRTDataModule(VitalDataModule):
    """Implementation of the ``VitalDataModule`` for the Lung X-ray dataset."""

    def __init__(
            self,
            dataset_path: Union[str, Path],
            da: bool = False,
            **kwargs,
    ):
        """Initializes class instance.

        Args:
            dataset_path: Path to the HDF5 dataset.
            **kwargs: Keyword arguments to pass to the parent's constructor.
        """
        dataset_path = Path(dataset_path)
        self.data_augmentation = da

        self.tta_transforms = Compose([RandomRotation(3),
                                      RandomBrightnessContrast(0.2, 0.2),
                                      RandomGamma((0.8, 1.2)),
                                      RandomTranslation(5, 5)])

        self.da_transforms = Compose([RandomRotation(3),
                                      RandomBrightnessContrast(0.2, 0.2),
                                      RandomGamma((0.8, 1.2)),
                                      RandomTranslation(5, 5)])

        self.transforms = self.da_transforms if da else None

        image_shape = (256, 256)
        in_channels = 1

        super().__init__(
            data_params=DataParameters(
                in_shape=(in_channels, *image_shape), out_shape=(len(Label), *image_shape), labels=tuple(Label)
            ),
            **kwargs,
        )

        self._dataset_kwargs = {
            "path": dataset_path,
        }

    def setup(self, stage: Literal["fit", "test"]) -> None:  # noqa: D102
        if stage == "fit":
            self.datasets[Subset.TRAIN] = JSRT(image_set=Subset.TRAIN, transforms=self.transforms,
                                               **self._dataset_kwargs)
            self.datasets[Subset.VAL] = JSRT(image_set=Subset.VAL, **self._dataset_kwargs)
        if stage == "test":
            self.datasets[Subset.TEST] = JSRT(image_set=Subset.TEST, predict=True, **self._dataset_kwargs)
        if stage == TrainerFn.PREDICTING:
            self.datasets[Subset.PREDICT] = JSRT(image_set=Subset.TEST, predict=True, **self._dataset_kwargs)
            self.datasets[Subset.PREDICT] = JSRT(image_set=Subset.TEST, predict=True, **self._dataset_kwargs)

    def predict_dataloader(self) -> DataLoader:  # noqa: D102
        return DataLoader(self.datasets[Subset.PREDICT], batch_size=None, num_workers=self.num_workers, pin_memory=True)


class JSRTContourDataModule(JSRTDataModule):
    contour_to_mask_fn = LungContourToMask()
    umap_fn = LungUMap()
    skew_umap_fn = LungSkewUmap()

    def __init__(self, dataset_path: Union[str, Path], **kwargs):
        super().__init__(dataset_path, **kwargs)
        image_shape = (256, 256)
        in_channels = 1

        self.data_params = DataParameters(
            in_shape=(in_channels, *image_shape), out_shape=(120, 2), labels=tuple(Label))
