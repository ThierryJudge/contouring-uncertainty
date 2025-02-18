from typing import Literal, Tuple, Union
from torchinfo import summary
from torch import Tensor, nn

from vital.data.transforms import GrayscaleToRGB
from contour_uncertainty.models import deeplabv3


class ConfidenceNet(nn.Module):
    """
        Bottleneck (N, C, Hb, Wb) to output_size (N, K)
    """

    def __init__(self, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )

    def forward(self, x):
        return self.model(x)


class DeepLabv3(nn.Module):
    """Wrapper around torchvision's implementation of the DeepLabv3 model that allows for single-channel inputs."""

    def __init__(
            self,
            input_shape: Tuple[int, ...],
            output_shape: Tuple[int, ...],
            backbone: Literal["resnet50", "resnet101", "deeplabv3_mobilenet_v3_large"] = "resnet50",
            convert_grayscale_to_rgb: bool = True,
            pretrained: bool = False,
            pretrained_backbone: bool = False,
            dropout: float = 0,
            bottleneck_out: bool = False,
            n_heads: int = 1,
    ):
        """Initializes class instance.

        Args:
            input_shape: (in_channels, H, W), Shape of the input images.
            output_shape: (num_classes, H, W), Shape of the output segmentation map.
            backbone: The network used by the DeepLabv3 architecture to compute the features for the model.
            convert_grayscale_to_rgb: If ``True``, the forward pass will automatically convert single channel grayscale
                inputs to 3-channel RGB, where r == g == b, to fit with DeepLabv3's hardcoded 3 channel input layer.
                If ``False``, the input is assumed to already be 3 channel and is not transformed in any way.
            pretrained: Whether to use torchvision's pretrained weights for the DeepLabV3-specific modules.
            pretrained_backbone: Whether to use torchvision's pretrained weights for the backbone used by DeepLabV3,
                e.g. ResNet50.
        """
        super().__init__()
        self._convert_grayscale_to_rgb = convert_grayscale_to_rgb
        if self._convert_grayscale_to_rgb:
            self._grayscale_trans = GrayscaleToRGB()
        model_cls = deeplabv3.__dict__[f"deeplabv3_{backbone}"]

        assert not (n_heads > 1 and bottleneck_out), 'Use of n_heads > 1 and bottleneck_out=True not supported'

        self.bottleneck_out = bottleneck_out
        self.n_heads = n_heads

        self._network = model_cls(
            pretrained=pretrained, pretrained_backbone=pretrained_backbone, aux_loss=False, num_classes=output_shape[0],
            features=bottleneck_out, n_heads=n_heads, dropout=dropout
        )
        self.confidence_net = ConfidenceNet  # Reference for confidence net

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, ...]]:
        """Defines the computation performed at every call.

        Args:
            x: (N, 1|3, H, W), Input image to segment.

        Returns:
            (N, ``num_classes``, H, W), Raw, unnormalized scores for each class in the input's segmentation.
        """
        if self._convert_grayscale_to_rgb and x.shape[1] != 3:
            x = self._grayscale_trans(x)

        if self.bottleneck_out:
            return self._network(x)["out"], self._network(x)["features"]
        network_output = self._network(x)
        if self.n_heads > 1:
            out = []
            for i in range(self.n_heads):
                out.append(network_output[f"out{i}"])
            return tuple(out)
        else:
            return network_output["out"]


if __name__ == "__main__":
    import torch

    # n_heads = 5
    # bottleneck_out = False

    n_heads = 1
    bottleneck_out = True

    model = DeepLabv3(input_shape=(1, 256, 256), output_shape=(19, 2), bottleneck_out=bottleneck_out,
                      n_heads=n_heads)

    x = torch.rand(2, *(1, 256, 256)).type(torch.float)

    model_summary = summary(
        model,
        input_data=x,
        col_names=["input_size", "output_size", "kernel_size", "num_params"],
        verbose=0,
    )

    print(model_summary)

    if bottleneck_out:
        y, z = model(x)
        print(y.shape)
        print(z.shape)
    if n_heads > 1:
        y = model(x)
        for yi in y:
            print(yi.shape)
            print(yi[0])
