from pathlib import Path
from typing import Literal, Optional

from pytorch_lightning.trainer.states import TrainerFn
from vital.data.camus.config import CamusTags, Label
from vital.data.config import DataParameters, Subset

from contour_uncertainty.data.camus.dataset import CamusContour, ContourTags
from contour_uncertainty.augmentations.affine import RandomRotation, RandomTranslation
from contour_uncertainty.augmentations.augmentation import Compose
from contour_uncertainty.augmentations.brightnesscontrast import RandomBrightnessContrast
from contour_uncertainty.augmentations.gamma import RandomGamma
import vital.data.camus.data_module as dataset
from contour_uncertainty.data.camus.utils import USContourToMask, USUMap, USSkewUmap


class CamusDataModule(dataset.CamusDataModule):
    """Implementation of the ``VitalDataModule`` for the CAMUS dataset."""

    def __init__(
            self,
            points_per_side=11,
            da: bool = True,
            **kwargs,
    ):
        """Initializes class instance.

        Args:
            dataset_path: Path to the HDF5 dataset.
            labels: Labels of the segmentation classes to take into account (including background). If None, target all
                labels included in the data.
            fold: ID of the cross-validation fold to use.
            use_sequence: Enable use of full temporal sequences.
            num_neighbors: Number of neighboring frames on each side of an item's frame to include as part of an item's
                data.
            neighbor_padding: Mode used to determine how to pad neighboring instants at the beginning/end of a sequence.
                The options mirror those of the ``mode`` parameter of ``numpy.pad``.
            **kwargs: Keyword arguments to pass to the parent's constructor.
        """

        super().__init__(**kwargs)
        self._dataset_kwargs["points_per_side"] = points_per_side
        self._dataset_kwargs.pop("neighbors")
        self._dataset_kwargs.pop("neighbor_padding")

        self.tta_transforms = Compose([RandomRotation(3),
                                      RandomBrightnessContrast(0.2, 0.2),
                                      RandomGamma((0.8, 1.2)),
                                      RandomTranslation(5, 5)])

        # Set at attribute for use in Test Time data augmentation
        self.da_transforms = Compose([RandomRotation(3),
                                      RandomBrightnessContrast(0.2, 0.2),
                                      RandomGamma((0.8, 1.2)),
                                      RandomTranslation(5, 5)])

        self.transforms = self.da_transforms if da else None

    def setup(self, stage: Optional[str] = None) -> None:  # noqa: D102
        if stage == TrainerFn.FITTING:
            self.datasets[Subset.TRAIN] = CamusContour(image_set=Subset.TRAIN, transforms=self.transforms,
                                                       **self._dataset_kwargs)
        if stage in [TrainerFn.FITTING, TrainerFn.VALIDATING]:
            self.datasets[Subset.VAL] = CamusContour(image_set=Subset.VAL, **self._dataset_kwargs)
        if stage == TrainerFn.TESTING:
            self.datasets[Subset.TEST] = CamusContour(image_set=Subset.TEST, **self._dataset_kwargs)
        if stage == TrainerFn.PREDICTING:
            self.datasets[Subset.PREDICT] = CamusContour(image_set=Subset.TEST, predict=True, **self._dataset_kwargs)


class CamusContourDataModule(CamusDataModule):
    contour_to_mask_fn = USContourToMask()
    umap_fn = USUMap()
    skew_umap_fn = USSkewUmap()

    def __init__(self, points_per_side=10, **kwargs):
        super().__init__(points_per_side=points_per_side, **kwargs)

        remove_bg = int(Label.BG in self.data_params.labels)
        points_per_contour = points_per_side * 2 - 1
        nb_points = points_per_contour * (len(self.data_params.labels) - remove_bg)

        self.data_params = DataParameters(
            in_shape=self.data_params.in_shape, out_shape=(nb_points, 2), labels=self.data_params.labels
        )


if __name__ == "__main__":
    """
    This script can be run to test and visualize the data from the dataset.
    """
    from argparse import ArgumentParser

    args = ArgumentParser(add_help=False)
    args.add_argument("path", type=str)
    params = args.parse_args()

    dm = CamusContourDataModule(Path(params.path), fold=5, batch_size=32, num_workers=0)
    dm.setup("fit")

    train_loader = dm.val_dataloader()
    for batch in iter(train_loader):
        x = batch[CamusTags.img]
        y = batch[ContourTags.contour]

        print(x.shape)
        print(y.shape)
