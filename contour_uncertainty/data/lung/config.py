from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Sequence

import numpy as np

from vital.data.config import LabelEnum, Tags


class Label(LabelEnum):
    """Enumeration of tags related to the different anatomical structures segmented in the dataset.

    Attributes:
        BG: Label of the background.
        FG: Label of the forground (lung) class
    """

    BG = 0
    LUNG = 1
    HEART = 2


@dataclass(frozen=True)
class View:
    """Collection of tags related to the different views available for each patient.

    Args:
        front: Tag referring to the posteroanterior (pa) view
    """

    PA: str = "pa"


RLUNG = 44
LLUNG = 50
HEART = 26
RCLAV = 23

RL_SPECIAL = [0, 21, 29]
LL_SPECIAL = [0, 21, 27, 40, 44]
H_SPECIAL = [0, 6, 12, 18]

in_channels: int = 1
"""Number of input channels of the images in the dataset."""

img_save_options: Dict[str, Any] = {"dtype": np.float32, "compression": "gzip", "compression_opts": 4}
"""Options to pass along when saving the ultrasound image in an HDF5 file."""

seg_save_options: Dict[str, Any] = {"dtype": np.uint8, "compression": "gzip", "compression_opts": 4}
"""Options to pass along when saving the segmentation mask in an HDF5 file."""
