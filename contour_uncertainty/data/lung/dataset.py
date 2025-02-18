from pathlib import Path
from typing import Callable, Dict, List, Tuple

import h5py
import numpy as np
import torch
import torchvision
from torch import Tensor
from torchvision.datasets import VisionDataset
from torchvision.transforms.functional import to_tensor

from contour_uncertainty.augmentations.affine import RandomRotation, RandomTranslation
from contour_uncertainty.augmentations.augmentation import Compose
from contour_uncertainty.augmentations.gamma import RandomGamma
from contour_uncertainty.data.config import ContourTags
from vital.data.config import Subset
from vital.data.config import Tags
from vital.utils.decorators import squeeze
from vital.utils.image.transform import segmentation_to_tensor

from contour_uncertainty.data.lung.utils import LungContourToMask


class JSRT(VisionDataset):
    """Implementation of torchvision's ``VisionDataset`` for the  JSRT dataset."""

    def __init__(
            self,
            path: Path,
            image_set: Subset,
            predict: bool = False,
            transforms: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]] = None,
    ):
        """Initializes class instance.

        Args:
            path: Path to the HDF5 dataset.
            image_set: Subset of images to use.
            predict: Whether to receive the data in a format fit for inference (``True``) or training (``False``).
            transforms: Function that takes in an input/target pair and transforms them in a corresponding way.
                (only applied when `predict` is `False`, i.e. in train/validation mode)
        """
        super().__init__(path, transforms=transforms)
        self.image_set = image_set
        self.predict = predict

        self._base_transform = torchvision.transforms.ToTensor()

        # Determine whether to return data in a format suitable for training or inference
        self.item_list = self.list_items()
        self.getter = self._get_test_item if self.predict else self._get_train_item

    def __getitem__(self, index) -> Dict[str, Tensor]:
        """Fetches an item, whose structure depends on the ``predict`` value, from the internal list of items.

        Notes:
            - When in ``predict`` mode (i.e. for test-time inference), an item corresponds to the views' ultrasound
              images and groundtruth segmentations for a patient.
            - When not in ``predict`` mode (i.e. during training), an item corresponds to an image/segmentation pair for
              a single frame.

        Args:
            index: Index of the item to fetch from the internal sequence of items.

        Returns:
            Item from the internal list at position ``index``.
        """
        return self.getter(index)

    def __len__(self):  # noqa: D105
        return len(self.item_list)

    def list_items(self) -> List[str]:
        """Lists the paths of the different items.

        Returns:
            IDs of the different levels of groups/clusters data samples in ``self.image_set`` can belong to.
        """
        with h5py.File(self.root, "r") as dataset:
            # List the patients
            items = [f"{self.image_set}/{x}" for x in dataset[self.image_set.value].keys()]

        return items

    def _get_item(self, index: int) -> Dict:
        """Fetches data and metadata related to an instant (single image/groundtruth pair + metadata).

        Args:
            index: Index of the instant sample in the train/val set's ``self.item_list``.

        Returns:
            Data and metadata related to an instant.
        """
        item_key = self.item_list[index]

        # Collect data
        with h5py.File(self.root, "r") as dataset:
            img, gt, lnd = self._get_data(dataset, item_key, Tags.img, Tags.gt, ContourTags.contour)

        img = img / 255

        img, gt, lnd = to_tensor(img), segmentation_to_tensor(gt), torch.Tensor(lnd)

        if self.transforms is not None:
            transformed = self.transforms(image=img, mask=gt, keypoints=lnd)
            img = transformed["image"]
            gt = transformed["mask"]
            lnd = transformed["keypoints"]

        return {
            Tags.id: f"{item_key}",
            Tags.img: img,
            Tags.gt: gt,
            ContourTags.contour: lnd
        }

    def _get_test_item(self, index: int) -> Dict:
        item = self._get_item(index)
        item[Tags.img] = item[Tags.img][None]
        item[Tags.gt] = item[Tags.gt][None]
        item[ContourTags.contour] = item[ContourTags.contour][None]
        return item

    def _get_train_item(self, index: int) -> Dict:
        return self._get_item(index)

    @staticmethod
    @squeeze
    def _get_data(file: h5py.File, patient_view_key: str, *data_tags: str) -> List[np.ndarray]:
        """Fetches the requested data for a specific patient/view dataset from the HDF5 file.

        Args:
            file: HDF5 dataset file.
            patient_view_key: `patient/view` access path of the desired view group.
            *data_tags: Names of the datasets to fetch from the view.

        Returns:
            Dataset content for each tag passed in the parameters.
        """
        patient_view = file[patient_view_key]
        return [patient_view[data_tag][()] for data_tag in data_tags]


if __name__ == "__main__":
    from argparse import ArgumentParser

    from matplotlib import pyplot as plt

    args = ArgumentParser(add_help=False)
    args.add_argument("path", type=str)
    args.add_argument("--predict", action="store_true")
    params = args.parse_args()

    ds = JSRT(Path(params.path), image_set=Subset.TRAIN, predict=params.predict,
              transforms=Compose([RandomRotation(3),
                                  RandomGamma((60, 140)),
                                  RandomTranslation(5, 5)]))

    sample = ds[0]  # random.randint(0, len(ds) - 1)]
    img = sample[Tags.img].squeeze()
    gt = sample[Tags.gt]

    contour = sample['contour']

    LungContourToMask()(contour.numpy(), reconstruction_type='linear')

    print("Image shape: {}".format(img.shape))
    print("GT shape: {}".format(gt.shape))

    print(img.min())
    print(img.max())
    print(img.mean())
    print(img.std())

    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img)
    ax2.imshow(gt)
    plt.show(block=False)

    plt.figure(2)
    plt.imshow(img, cmap="gray")
    plt.imshow(gt, alpha=0.2)
    plt.show()
