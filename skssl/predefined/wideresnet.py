import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from skssl.helpers import is_valid_image_shape, closest_power2

# to replicate https://github.com/brain-research/realistic-ssl-evaluation/
CONV_KWARGS = dict(kernel_size=3, padding=1, bias=False)
BATCHNORM_KWARGS = dict(momentum=1e-3)


class WideResNet(nn.Module):
    def __init__(self, x_shape, num_classes, n_res_unit=4, widen_factor=2, leakiness=0.1):
        super().__init__()

        is_valid_image_shape(x_shape)
        self.x_shape = x_shape

        n_chan = [self.x_shape[0], 16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        self.n_chan_fin = n_chan[4]

        # padding 1 gives same output size in our case (pytorch doesn't suport "SAME")
        self.conv1 = nn.Conv2d(n_chan[0], n_chan[1], stride=1, **CONV_KWARGS)
        self.block1 = get_res_block(n_res_unit, n_chan[1], n_chan[2], 1, leakiness, True)
        self.block2 = get_res_block(n_res_unit, n_chan[2], n_chan[3], 2, leakiness, False)
        self.block3 = get_res_block(n_res_unit, n_chan[3], self.n_chan_fin, 2, leakiness, False)

        self.bn1 = nn.BatchNorm2d(self.n_chan_fin, **BATCHNORM_KWARGS)
        self.act = nn.LeakyReLU(negative_slope=leakiness)
        self.fc = nn.Linear(self.n_chan_fin, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.act(self.bn1(out))
        # global average over all pixels left
        out = F.adaptive_avg_pool2d(out, 1).view(-1, self.n_chan_fin)
        return self.fc(out)


class ResLayer(nn.Module):
    def __init__(self, in_filter, out_filter, stride,
                 leakiness=1e-2, is_act_before_res=True,
                 is_reverse=False, Conv=nn.Conv2d):
        super().__init__()
        self.act = nn.LeakyReLU(negative_slope=leakiness)
        self.bn1 = nn.BatchNorm2d(in_filter, **BATCHNORM_KWARGS)
        self.conv1 = Conv(in_filter, out_filter, stride=stride, **CONV_KWARGS)
        self.bn2 = nn.BatchNorm2d(out_filter, **BATCHNORM_KWARGS)
        self.conv2 = Conv(out_filter, out_filter, stride=1, **CONV_KWARGS)
        self.is_in_neq_out = (in_filter != out_filter)
        if self.is_in_neq_out:
            self.conv_shortcut = Conv(in_filter, out_filter, kernel_size=1,
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


def get_res_block(n_layers, in_filter, out_filter, stride, leakiness,
                  is_act_before_res, is_reverse=False):

    kwargs = dict(leakiness=leakiness, is_reverse=is_reverse)
    layer = ResLayer if not is_reverse else ReversedResLayer

    layer_transf = layer(in_filter, out_filter, stride,
                         is_act_before_res=is_act_before_res, **kwargs)
    layers_id = [layer(out_filter, out_filter, 1, is_act_before_res=False, **kwargs)
                 for i in range(n_layers)]

    if not is_reverse:
        # Encoder like
        layers = [layer_transf] + layers_id
    else:
        # Decoder like
        layers = layers_id + [layer_transf]

    return nn.Sequential(*layers)


def ReversedConv2d(in_filter, out_filter, **kwargs):
    """Called the exact same way as Conv2d => with same in and out filter!"""
    return nn.ConvTranspose2d(out_filter, in_filter, **kwargs)


class ReversedResLayer(ResLayer):
    def __init__(self, in_filter, out_filter, stride, **kwargs):
        super().__init__(in_filter, out_filter, stride, Conv=ReversedConv2d, **kwargs)
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


class ReversedWideResNet(nn.Module):
    def __init__(self, x_shape, num_classes, n_res_unit=4, widen_factor=2, leakiness=0.1):
        super().__init__()

        is_valid_image_shape(x_shape)

        self.x_shape = x_shape

        n_chan = [self.x_shape[0], 16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        self.n_chan_fin = n_chan[4]

        self.act = nn.LeakyReLU(negative_slope=leakiness)

        self.fc1 = nn.Linear(num_classes, self.n_chan_fin)
        self.fc2 = nn.Linear(self.n_chan_fin, self.n_chan_fin * 2)
        self.fc3 = nn.Linear(self.n_chan_fin * 2, self.n_chan_fin * 4)

        self.bn1 = nn.BatchNorm2d(self.n_chan_fin, **BATCHNORM_KWARGS)

        # upsample by 2 until reaches the correct size => with default param
        # works with any images that are larger than (n_chan, 16, 16), and are
        # square with a width being a power of 2 (e.g. (3,64,64), (1,128,128), ...)
        tmp_upsampling = []
        for _ in range(int(math.log2(self.x_shape[1])) - 4 + 1):
            tmp_upsampling.append(nn.ConvTranspose2d(self.n_chan_fin, self.n_chan_fin,
                                                     4, stride=2, padding=1))
            tmp_upsampling.append(self.act)
        self.tmp_upsampling = nn.Sequential(*tmp_upsampling) if len(tmp_upsampling) != 0 else None

        # reverse from the WideResnet above
        self.block1 = get_res_block(n_res_unit, n_chan[3], self.n_chan_fin, 2,
                                    leakiness, False, is_reverse=True)
        self.block2 = get_res_block(n_res_unit, n_chan[2], n_chan[3], 2,
                                    leakiness, False, is_reverse=True)
        self.block3 = get_res_block(n_res_unit, n_chan[1], n_chan[2], 1,
                                    leakiness, True, is_reverse=True)
        self.conv1 = ReversedConv2d(n_chan[0], n_chan[1], stride=1, **CONV_KWARGS)

    def forward(self, x):
        batch_size = x.size(0)
        out = self.act(self.fc1(x))
        out = self.act(self.fc2(out))
        out = self.act(self.fc3(out))
        out = out.view(batch_size, self.n_chan_fin, -1)
        # make square image
        out = out.view(batch_size, self.n_chan_fin, int(out.size(-1)**0.5), int(out.size(-1)**0.5))
        out = self.bn1(out)
        if self.tmp_upsampling is not None:
            out = self.tmp_upsampling(out)

        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.conv1(out)

        # all images in this framework are scaled in [0,1]
        return torch.sigmoid(out)
