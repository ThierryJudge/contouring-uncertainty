from collections import OrderedDict
from copy import deepcopy
from typing import Optional, Dict

from torch import nn, Tensor
from torch.hub import load_state_dict_from_url
from torch.nn import functional as F
from torchvision.utils import _log_api_usage_once


class _SimpleSegmentationModel(nn.Module):
    __constants__ = ["aux_classifier"]

    def __init__(self, backbone: nn.Module, classifier: nn.Module, aux_classifier: Optional[nn.Module] = None,
                 features: bool = False,
                 n_heads: int = 1) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier
        self.features = features

        if isinstance(self.classifier, nn.ModuleList):
            self.n_heads = len(classifier)
        else:
            self.n_heads = 1

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()
        if self.n_heads == 1:
            x = features["out"]
            x = self.classifier(x)
            x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
            result["out"] = x
        else:
            for i in range(self.n_heads):
                x = features["out"]
                x = self.classifier[i](x)
                x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
                result[f"out{i}"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
            result["aux"] = x

        # Added code to output features
        if self.features:
            result["features"] = features["out"]

        return result


def _load_weights(arch: str, model: nn.Module, model_url: Optional[str], progress: bool) -> None:
    if model_url is None:
        raise ValueError(f"No checkpoint is available for {arch}")
    state_dict = load_state_dict_from_url(model_url, progress=progress)
    model.load_state_dict(state_dict)
