from typing import Tuple, Union

import torch
import vital.models.segmentation.enet as vital_enet
from torch import Tensor, nn
from vital.models.segmentation.enet import _UpsamplingBottleneck, _RegularBottleneck


class ConfidenceNet(nn.Module):
    """
        Bottleneck (N, C, Hb, Wb) to output_size (N, K)
    """

    def __init__(self, output_size):
        super().__init__()
        self.model = nn.Sequential(
            # vital_enet._DownsamplingBottleneck(64, 128),
            # vital_enet._DownsamplingBottleneck(128, 128, internal_ratio=4)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(128 * 4 * 4, output_size),
            # nn.ReLU(),
            # nn.Linear(1024, 512),
            # nn.ReLU(),
            # nn.Linear(512, output_size)
        )

    def forward(self, x):
        return self.model(x)


class Enet(vital_enet.Enet):
    """Implementation of the ENet model.

    References:
        - Paper that introduced the model: http://arxiv.org/abs/1606.02147
    """

    def __init__(
            self,
            input_shape: Tuple[int],
            output_shape: Tuple[int],
            init_channels: int = 16,
            dropout: float = 0.1,
            encoder_relu: bool = True,
            decoder_relu: bool = True,
            bottleneck_out: bool = False,
            n_heads: int = 1,
            ssn_rank: int = 0,
    ):
        """Initializes class instance.

        Args:
            input_shape: Shape of the input images.
            output_shape: Shape of the output segmentation map.
            init_channels: Number of output feature maps from the first layer, used to compute the number of feature
                maps in following layers.
            dropout: Probability of an element to be zeroed (e.g. 0 means no dropout).
                NOTE: In the initial block, the dropout probability is divided by 10.
            encoder_relu: When ``True`` ReLU is used as the activation function in the encoder blocks/layers; otherwise,
                PReLU is used.
            decoder_relu: When ``True`` ReLU is used as the activation function in the decoder blocks/layers; otherwise,
                PReLU is used.

        """
        super().__init__(input_shape, output_shape, init_channels, dropout, encoder_relu, decoder_relu)
        out_channels = output_shape[0]

        n_heads_size = [out_channels] * (n_heads - 1)

        if ssn_rank > 0:
            n_heads = 3
            n_heads_size = [out_channels, out_channels * ssn_rank]

        assert not (n_heads > 1 and bottleneck_out), 'Use of n_heads > 1 and bottleneck_out=True not supported'

        self.bottleneck_out = bottleneck_out
        self.n_heads = n_heads

        self.heads = [nn.ModuleDict({'upsample5_0': self.upsample5_0,
                                     'regular5_1': self.regular5_1,
                                     'transposed_conv': self.transposed_conv})]
        if self.n_heads > 1:
            for head_size in n_heads_size:
                self.heads.append(
                    nn.ModuleDict({
                        'upsample5_0': _UpsamplingBottleneck(
                            init_channels * 2, init_channels, padding=1, dropout=dropout, relu=decoder_relu),
                        'regular5_1': _RegularBottleneck(init_channels, padding=1, dropout=dropout, relu=decoder_relu),
                        'transposed_conv': nn.ConvTranspose2d(
                            init_channels, head_size, kernel_size=3, stride=2, padding=1, output_padding=1,
                            bias=False
                        )})
                )

            self.heads = nn.ModuleList(self.heads)

        self.confidence_net = ConfidenceNet  # Reference for confidence net

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, ...]]:
        """Defines the computation performed at every call.

        Args:
            x: (N, ``in_channels``, H, W), Input image to segment.

        Returns:
            (N, ``out_channels``, H, W), Raw, unnormalized scores for each class in the input's segmentation.
        """
        # Initial block
        x = self.initial_block(x)

        # Stage 1 - Encoder
        x, max_indices1_0 = self.downsample1_0(x)
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        x = self.regular1_3(x)
        x = self.regular1_4(x)

        # Stage 2 - Encoder
        x, max_indices2_0 = self.downsample2_0(x)
        x = self.regular2_1(x)
        x = self.dilated2_2(x)
        x = self.asymmetric2_3(x)
        x = self.dilated2_4(x)
        x = self.regular2_5(x)
        x = self.dilated2_6(x)
        x = self.asymmetric2_7(x)
        x = self.dilated2_8(x)

        # Stage 3 - Encoder
        x = self.regular3_0(x)
        x = self.dilated3_1(x)
        x = self.asymmetric3_2(x)
        x = self.dilated3_3(x)
        x = self.regular3_4(x)
        x = self.dilated3_5(x)
        x = self.asymmetric3_6(x)
        x = self.dilated3_7(x)
        bottleneck = x

        # Decoder
        # Stage 4 - Decoder
        x = self.upsample4_0(x, max_indices2_0)
        x = self.regular4_1(x)
        x = self.regular4_2(x)

        if self.bottleneck_out:
            # Stage 5 - Decoder
            x = self.upsample5_0(x, max_indices1_0)
            x = self.regular5_1(x)
            return self.transposed_conv(x), bottleneck
        if self.n_heads > 1:
            out = []
            for i in range(self.n_heads):
                z = self.heads[i]['upsample5_0'](x, max_indices1_0)
                z = self.heads[i]['regular5_1'](z)
                z = self.heads[i]['transposed_conv'](z)
                out.append(z)
            return tuple(out)
        else:
            # Stage 5 - Decoder
            x = self.upsample5_0(x, max_indices1_0)
            x = self.regular5_1(x)
            return self.transposed_conv(x)


if __name__ == "__main__":
    from torchsummary import summary

    model = Enet(input_shape=(1, 256, 256), output_shape=(4, 256, 256), ssn_rank=5)

    summary(model, (1, 256, 256), device="cpu")

    x = torch.rand(2, 1, 256, 256)
    y = model(x)
    print(len(y))

    for i in range(len(y)):
        print(y[i].shape)

