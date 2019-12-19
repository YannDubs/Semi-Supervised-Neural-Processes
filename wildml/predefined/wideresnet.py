import math

import torch.nn as nn
import torch.nn.functional as F

from wildml.utils.torchextend import ReversedConv2d, ReversedLinear
from wildml.utils.helpers import (is_valid_image_shape, closest_power2)
from wildml.utils.initialization import weights_init

__all__ = ["WideResNet", "ReversedWideResNet"]

# to replicate https://github.com/brain-research/realistic-ssl-evaluation/
CONV_KWARGS = dict(kernel_size=3, padding=1, bias=False)
# The implementation above uses default tensorflow batch norm aruments so
# momentum = 1 - 0.999 = 1e-3. But this doesn't work with pytorch, probably due
# how the moving average is intialized (i.e would take too many steps to
# find good values in pytorch because bad init) => use pytorch default
BATCHNORM_KWARGS = dict(momentum=1e-1)


class WideResNet(nn.Module):
    """Wide Resnet as used in [1].

    Notes
    -----
    - Number of parameters will be around 1.5M (depends on inputs outputs).
    - Number of conv layers is `1 + 3 + 6 * n_res_unit`. Default 28.

    Parameters
    ----------
    x_shape : tuple of ints
        Shape of the input images. Only tested on square images with width being
        a power of 2 and greater or equal than 16. E.g. (1,32,32) or (3,64,64).

    n_out : int
        Number of outputs.

    n_res_unit : int, optional
        Number of residual layers for each of the 3 residual block.

    widen_factor : int, optional
        Factor used to control the number of channels of the hidden layers.

    leakiness : float, optional
        Leakiness for leaky relu.

    Return
    ------
    out : torch.Tensor, size = [batch, size]
        Flattent raw output (no activations).

    References
    ----------
    [1] Oliver, A., Odena, A., Raffel, C. A., Cubuk, E. D., & Goodfellow, I.
        (2018). Realistic evaluation of deep semi-supervised learning algorithms.
        In Advances in Neural Information Processing Systems (pp. 3235-3246).
    """

    def __init__(self, x_shape, n_out, n_res_unit=4, widen_factor=2, leakiness=0.1,
                 _Conv=nn.Conv2d, _Linear=nn.Linear, **kwargs):
        super().__init__()

        is_valid_image_shape(x_shape, min_width=16)
        self.x_shape = x_shape

        n_chan = [self.x_shape[0], 16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        self.n_chan_fin = n_chan[4]

        # padding 1 gives same output size in our case (pytorch doesn't suport "SAME")
        self.conv = _Conv(n_chan[0], n_chan[1], stride=1, **CONV_KWARGS)
        self.block1 = _get_res_block(n_res_unit, n_chan[1], n_chan[2], 1,
                                     leakiness, True, **kwargs)  # check TRUE
        self.block2 = _get_res_block(n_res_unit, n_chan[2], n_chan[3], 2,
                                     leakiness, False, **kwargs)  # check FALSE
        self.block3 = _get_res_block(n_res_unit, n_chan[3], self.n_chan_fin, 2,
                                     leakiness, False, **kwargs)  # check FALSE

        self.bn = nn.BatchNorm2d(self.n_chan_fin, **BATCHNORM_KWARGS)
        self.act = nn.LeakyReLU(negative_slope=leakiness)
        self.fc = _Linear(self.n_chan_fin, n_out)

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, x):
        out = self.conv(x)
        out = self.block1(out)  # check
        out = self.block2(out)  # check
        out = self.block3(out)  # check
        out = self.act(self.bn(out))  # check
        # global average over all pixels left
        out = F.adaptive_avg_pool2d(out, 1).view(-1, self.n_chan_fin)  # check
        return self.fc(out)  # check


class ReversedWideResNet(WideResNet):
    """
    Reversed version of `WideResNet`. The number of parameters is closer to 2.2M
    because has to add layers to undo global average pooling. Concolution layers
    are replaced with Transposed Convolutions.
    """

    def __init__(self, n_in, x_shape_out, **kwargs):

        super().__init__(x_shape_out, n_in,  # reverses
                         is_reverse=True,
                         _Conv=ReversedConv2d,
                         _Linear=ReversedLinear,
                         **kwargs)

        self.fc2 = nn.Linear(self.n_chan_fin, self.n_chan_fin * 2)
        self.fc3 = nn.Linear(self.n_chan_fin * 2, self.n_chan_fin * 4)

        # upsample by 2 until reaches the correct size => with default param
        # works with any images that are larger than (n_chan, 16, 16), and are
        # square with a width being a power of 2 (e.g. (3,64,64), (1,128,128), ...)
        tmp_upsampling = []
        for _ in range(int(math.log2(self.x_shape[1])) - 4 + 1):
            tmp_upsampling.append(nn.ConvTranspose2d(self.n_chan_fin, self.n_chan_fin,
                                                     4, stride=2, padding=1))
            tmp_upsampling.append(self.act)
        self.tmp_upsampling = nn.Sequential(*tmp_upsampling) if len(tmp_upsampling) != 0 else None

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, x):
        batch_size = x.size(0)
        out = self.act(self.fc(x))
        out = self.act(self.fc2(out))
        out = self.act(self.fc3(out))
        out = out.view(batch_size, self.n_chan_fin, -1)
        # make square image
        out = out.view(batch_size, self.n_chan_fin, int(out.size(-1)**0.5), int(out.size(-1)**0.5))
        out = self.bn(out)
        if self.tmp_upsampling is not None:
            out = self.tmp_upsampling(out)

        # reversed order
        out = self.block3(out)
        out = self.block2(out)
        out = self.block1(out)
        out = self.conv(out)

        return out


def _get_res_block(n_layers, in_filter, out_filter, stride, leakiness,
                   is_act_before_res, is_reverse=False):

    layer = _ResLayer if not is_reverse else _ReversedResLayer

    layer_transf = layer(in_filter, out_filter, stride,
                         is_act_before_res=is_act_before_res, leakiness=leakiness)
    layers_id = [layer(out_filter, out_filter, 1, is_act_before_res=False, leakiness=leakiness)
                 for i in range(1, n_layers)]

    if not is_reverse:
        # Encoder like
        layers = [layer_transf] + layers_id
    else:
        # Decoder like
        layers = layers_id + [layer_transf]

    return nn.Sequential(*layers)


class _ResLayer(nn.Module):
    def __init__(self, in_filter, out_filter, stride,
                 leakiness=1e-2, is_act_before_res=True, _Conv=nn.Conv2d):
        super().__init__()
        self.act = nn.LeakyReLU(negative_slope=leakiness)
        self.bn1 = nn.BatchNorm2d(in_filter, **BATCHNORM_KWARGS)
        self.conv1 = _Conv(in_filter, out_filter, stride=stride, **CONV_KWARGS)
        self.bn2 = nn.BatchNorm2d(out_filter, **BATCHNORM_KWARGS)
        self.conv2 = _Conv(out_filter, out_filter, stride=1, **CONV_KWARGS)
        self.is_in_neq_out = (in_filter != out_filter)
        if self.is_in_neq_out:
            self.conv_shortcut = _Conv(in_filter, out_filter, kernel_size=1,
                                       stride=stride, padding=0, bias=False)
        self.is_act_before_res = is_act_before_res

    def forward(self, x):
        if self.is_act_before_res:
            x = self.act(self.bn1(x))
            out = self.conv1(x)
        else:
            out = self.conv1(self.act(self.bn1(x)))

        out = self.conv2(self.act(self.bn2(out)))
        res = self.conv_shortcut(x) if self.is_in_neq_out else x
        return res + out


class _ReversedResLayer(_ResLayer):
    def __init__(self, in_filter, out_filter, stride, **kwargs):
        super().__init__(in_filter, out_filter, stride, _Conv=ReversedConv2d, **kwargs)
        self.stride = stride

    def forward(self, x):
        # Litteraly reverses everything
        if self.is_act_before_res:
            x = self.act(self.conv2(x))
            out = self.bn2(x)
        else:
            out = self.bn2(self.act(self.conv2(x)))

        output_size = out.size()
        # make sure that the output is a power of 2 because Conv2d is actually not
        # a bijective transformation => TransposeConv2d might return the wrong size
        output_size = output_size[:-2] + (closest_power2(output_size[-2] * self.stride),
                                          closest_power2(output_size[-2] * self.stride))
        out = self.bn1(self.act(self.conv1(out, output_size=output_size)))
        res = self.conv_shortcut(x, output_size=output_size) if self.is_in_neq_out else x
        return res + out
