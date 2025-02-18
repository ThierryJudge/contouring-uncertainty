import copy
from typing import Any, Callable, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch import Tensor
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1, conv3x3


class DropoutBottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            dropout: float = 0.0,
    ) -> None:
        super(DropoutBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.drop1 = nn.Dropout2d(dropout)

        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.drop2 = nn.Dropout2d(dropout)

        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.drop3 = nn.Dropout2d(dropout)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.drop2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        out = self.drop3(out)

        return out


class _ResNet(models.ResNet):
    def __init__(
            self,
            block: DropoutBottleneck,
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            dropout: float = 0.0,
            sigma_out: int = 0,
    ) -> None:
        self.dropout = dropout
        self.sigma_out = sigma_out
        super().__init__(
            block,
            layers,
            num_classes,
            zero_init_residual,
            groups,
            width_per_group,
            replace_stride_with_dilation,
            norm_layer,
        )
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        if sigma_out > 0:
            # self.sigma_layer4 = self._make_layer(block, 512, layers[3], stride=2,
            #                                      dilate=replace_stride_with_dilation[2])
            self.sigma_layer3 = copy.deepcopy(self.layer3)
            self.sigma_layer4 = copy.deepcopy(self.layer4)
            self.sigma_avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.sigma_fc = nn.Linear(512 * block.expansion, sigma_out)

    def _make_layer(
            self, block: DropoutBottleneck, planes: int, blocks: int, stride: int = 1, dilate: bool = False
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                dropout=self.dropout,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        sigma_split = x
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if self.sigma_out:
            sigma_split = self.sigma_layer3(sigma_split)
            sigma_split = self.sigma_layer4(sigma_split)
            sigma_split = self.sigma_avgpool(sigma_split)
            sigma_split = torch.flatten(sigma_split, 1)
            sigma = self.sigma_fc(sigma_split)
            # sigma = self.sigma_fc(x)
            return self.fc(x), sigma
        else:
            return self.fc(x)

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        return self._forward_impl(x)


class Resnet(nn.Module):
    def __init__(
            self,
            input_shape: Tuple[int, ...],
            output_shape: Tuple[int, ...],
            dropout: float = 0.0,
            sigma_out: int = 0,
    ):
        super().__init__()
        in_channels = input_shape[0]
        self.output_shape = output_shape
        self.sigma_out = sigma_out
        self.dropout = dropout

        self.module = _ResNet(
            DropoutBottleneck,
            [3, 4, 6, 3],
            num_classes=int(np.prod(output_shape)),
            dropout=dropout,
            sigma_out=sigma_out * output_shape[0],
        )
        self.module.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if self.sigma_out:
            x, sigma = self.module(x)
            return x.reshape((-1,) + tuple(self.output_shape)), \
                   sigma.reshape((-1, self.output_shape[0], self.sigma_out))
        else:
            return self.module(x).reshape((-1,) + tuple(self.output_shape))


def _resnet(
        arch: str,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        pretrained: bool,
        progress: bool,
        **kwargs: Any,
) -> _ResNet:
    model = _ResNet(block, layers, **kwargs)
    if pretrained:
        raise NotImplementedError
        # state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        # model.load_state_dict(state_dict)
    return model


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> _ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", DropoutBottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> _ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet101", DropoutBottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


if __name__ == "__main__":
    from torchsummary import summary

    model = Resnet(input_shape=(1, 256, 256), output_shape=(19, 2))

    summary(model, (1, 256, 256), device="cpu")
