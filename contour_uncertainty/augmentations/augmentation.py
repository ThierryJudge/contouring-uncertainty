from typing import List, Dict, Callable, Sequence, Optional

import torch


class Augmentation:
    """
    Re-implementation of standard data augmentation techniques with appropriate un-apply functions for test-time augmentation
    """

    def __init__(self):
        self.params = None  # Keep to un-apply

    @property
    def apply_targets(self) -> Dict[str, Callable]:
        return {
            "image": self.apply_img,
            "mask": self.apply_mask,
            "keypoints": self.apply_keypoints,
        }

    @property
    def un_apply_targets(self) -> Dict[str, Callable]:
        return {
            "image": self.un_apply_img,
            "mask": self.un_apply_mask,
            "keypoints": self.un_apply_keypoints,
        }

    def __call__(
            self,
            image: torch.Tensor = None,
            mask: torch.Tensor = None,
            keypoints: torch.Tensor = None,
            *args,
            **kwargs
    ):
        items = {}
        if image is not None:
            items['image'] = image
        if mask is not None:
            items['mask'] = mask
        if keypoints is not None:
            items['keypoints'] = keypoints

        return self.apply(items)

    # TODO change input from dict to params
    def apply(self, items: dict, params=None) -> Dict[str, torch.Tensor]:
        output = {}
        if params is None:
            params = self.get_params()
        self.params = params  # Keep to un-apply
        for key, item in items.items():
            output[key] = self.apply_targets[key](item, **params)

        return output

    def un_apply(self, items: dict, params=None) -> Dict[str, torch.Tensor]:
        output = {}
        if params is None:
            params = self.params
            assert params is not None

        for key, item in items.items():
            output[key] = self.un_apply_targets[key](item, **params)

        return output

    def apply_img(self, img: torch.Tensor, **params):
        raise NotImplementedError

    def apply_mask(self, mask: torch.Tensor, **params):
        return mask

    def apply_keypoints(self, keypoints: torch.Tensor, **params):
        return keypoints

    def un_apply_img(self, img: torch.Tensor, **params):
        raise NotImplementedError

    def un_apply_mask(self, mask: torch.Tensor, **params):
        return mask

    def un_apply_keypoints(self, keypoints: torch.Tensor, **params):
        return keypoints

    def get_params(self):
        return {}


class Compose(Augmentation):

    def __init__(self, transforms: List[Augmentation]):
        super().__init__()
        self.transforms = transforms

    def apply(self, item: Dict[str, torch.Tensor], params: Optional[List[Dict]] = None) -> Dict[str, torch.Tensor]:
        if params is not None:
            assert len(params) == len(self.transforms)
        else:
            params = [None] * len(self.transforms)

        for transform, params in zip(self.transforms, params):
            item = transform.apply(item)
        return item

    def un_apply(self, item: Dict[str, torch.Tensor], params: Optional[List[Dict]] = None) -> Dict[str, torch.Tensor]:
        if params is not None:
            assert len(params) == len(self.transforms)
        else:
            params = [None] * len(self.transforms)

        for transform, params in list(zip(self.transforms, params))[::-1]:
            item = transform.un_apply(item)

        return item

    def get_params(self) -> List[Dict]:
        params = []
        for transform in self.transforms:
            params.append(transform.get_params())
        return params


def to_tuple(param, low=None, bias=None):
    """Convert input argument to min-max tuple

    COPIED FROM albumentations/core/transforms_interface.py

    Args:
        param (scalar, tuple or list of 2+ elements): Input value.
            If value is scalar, return value would be (offset - value, offset + value).
            If value is tuple, return value would be value + offset (broadcasted).
        low:  Second element of tuple can be passed as optional argument
        bias: An offset factor added to each element
    """
    if low is not None and bias is not None:
        raise ValueError("Arguments low and bias are mutually exclusive")

    if param is None:
        return param

    if isinstance(param, (int, float)):
        if low is None:
            param = -param, +param
        else:
            param = (low, param) if low < param else (param, low)
    elif isinstance(param, Sequence):
        param = tuple(param)
    else:
        raise ValueError("Argument param must be either scalar (int, float) or tuple")

    if bias is not None:
        return tuple(bias + x for x in param)

    return tuple(param)
