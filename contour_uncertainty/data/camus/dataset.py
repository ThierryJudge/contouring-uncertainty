from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import h5py
import numpy as np
import skimage
import torch
import torchvision
from tqdm import tqdm
from vital.data.camus.config import CamusTags
from vital.data.camus.config import Label
from vital.data.camus.dataset import Camus
from vital.data.camus.dataset import InstantItem
from vital.data.config import Subset
from vital.utils.image.transform import segmentation_to_tensor

from contour_uncertainty.data.camus.extract_points import get_contour_points


@dataclass(frozen=True)
class ContourTags(CamusTags):
    contour: str = "contour"  # Contour GT (K,2)
    contour_proc: str = "contour_proc"  # Contour GT (K,2)
    contour_pred: str = "contour_pred"  # Contour prediction (K,2)
    contour_sigma: str = "contour_sigma"  # Contour prediction covariance matrices (K,2,2)
    contour_unc: str = 'contour_unc'  # Contour uncertainty per point (K,)
    image_unc: str = 'image_unc'  # Global contour uncertainty per image (1,)


class CamusContour(Camus):
    """Implementation of torchvision's ``VisionDataset`` for the CAMUS dataset."""

    def __init__(self, path: Path, fold: int, image_set: Subset, points_per_side: int = 11, *args, **kwargs):
        super().__init__(path, fold, image_set, *args, **kwargs)
        self.points_per_side = points_per_side
        self.points_dict = {Label.LV: 21,
                            Label.MYO: 21,
                            Label.ATRIUM: 15}

        contour_filename = path.parent / f'{path.stem}_{image_set}_{self.points_per_side}.pt'
        if contour_filename.exists():
            self.contours = torch.load(str(contour_filename))
        else:
            self.contours = self.get_contours()
            torch.save(self.contours, contour_filename)

        self._base_transform = torchvision.transforms.ToTensor()
        self._base_target_transform = segmentation_to_tensor

    def get_contours(self):
        contours = {}
        view_instants = defaultdict(list)
        for view_key, instant in self._get_instant_paths():
            view_instants[view_key].append(instant)
        # c = 0
        with h5py.File(self.root, "r") as dataset:
            for key, instants in tqdm(view_instants.items(), desc='Computing contours'):
                gt = self._get_data(dataset, key, CamusTags.gt_proc)
                instant_indices = np.ones(len(gt), dtype=np.int8) * -1
                view_contours = defaultdict(list)
                for i, instant in enumerate(instants):
                    # print(c)
                    # lv_points, myo_points, la_points = get_contour_points(gt[instant], self.points_dict)
                    lv_points, myo_points = get_contour_points(gt[instant], self.points_dict)
                    view_contours[Label.LV].append(lv_points)
                    view_contours[Label.MYO].append(myo_points)
                    # view_contours[Label.ATRIUM].append(la_points)
                    instant_indices[instant] = i
                    # c+= 1

                for label in Label:
                    if label in view_contours.keys():
                        view_contours[label] = np.array(view_contours[label])

                contours[key] = {"contours": view_contours, "indices": instant_indices}

        return contours

    def _get_predict_item(self, index: int) -> Dict:
        view_dict = super()._get_predict_item(index)

        view_contours = self.contours[view_dict['id']]["contours"]
        view_contours = np.concatenate([view_contours[label] for label in self.labels if label in view_contours.keys()],
                                       axis=1)

        view_dict[ContourTags.contour] = view_contours

        patient_view_key = self.item_list[index]

        with h5py.File(self.root, "r") as dataset:
            if 'ImageQuality' in dataset[patient_view_key].keys():
                view_dict['ImageQuality'] = Camus._get_metadata(dataset, patient_view_key, 'ImageQuality')

        view_dict[CamusTags.img] = view_dict[CamusTags.img] / 255
        
        return view_dict

    def _get_train_item(self, index: int) -> InstantItem:
        """Fetches data and metadata related to an instant (single image/groundtruth pair + metadata).

        Args:
            index: Index of the instant sample in the dataset's ``self.item_list``.

        Returns:
            Data and metadata related to an instant.
        """
        patient_view_key, instant = self.item_list[index]

        # Collect data
        with h5py.File(self.root, "r") as dataset:
            view_imgs, view_gts = self._get_data(dataset, patient_view_key, CamusTags.img_proc, CamusTags.gt_proc)

        # Format data
        img = view_imgs[instant]
        gt = self._process_target_data(view_gts[instant])

        img = img / 255

        img, gt = self._base_transform(img), self._base_target_transform(gt)

        view_contours = self.contours[patient_view_key]["contours"]

        view_contours = np.concatenate(
            [view_contours[label] for label in self.labels if label in view_contours.keys()],
            axis=1
        )
        idx = self.contours[patient_view_key]["indices"][instant]
        contour = torch.Tensor(view_contours[idx])

        # Apply transforms on the data
        if self.transforms:
            transformed = self.transforms(image=img, mask=gt, keypoints=contour)
            img = transformed["image"]
            gt = transformed["mask"]
            contour = transformed['keypoints']

        # Compute attributes on the data
        frame_pos = torch.tensor([instant / len(view_imgs)])

        return {
            CamusTags.id: f"{patient_view_key}/{instant}",
            CamusTags.group: patient_view_key,
            CamusTags.img: img,
            CamusTags.gt: gt,
            CamusTags.frame_pos: frame_pos,
            ContourTags.contour: contour,
        }
