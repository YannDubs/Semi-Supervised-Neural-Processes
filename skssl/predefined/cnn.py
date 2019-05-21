import numpy as np

import torch
from torch import nn

from skssl.utils.helpers import is_valid_image_shape, closest_power2
from skssl.utils.torchextend import ReversedConv2d, ReversedLinear


class SimpleCNN(nn.Module):
    def __init__(self, x_shape, n_out, _Conv=nn.Conv2d, _Linear=nn.Linear):
        r"""Simple convolutional encoder proposed in [1].

        Parameters
        ----------
        x_shape : tuple of ints
            Shape of the input images. Only tested on square images with width being
            a power of 2 and greater or equal than 16. E.g. (1,32,32) or (3,64,64).

        n_out : int
            Number of outputs.

        References
        ----------
            [1] Burgess, Christopher P., et al. "Understanding disentangling in
            $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
        """
        super().__init__()
        is_valid_image_shape(x_shape, min_width=32, max_width=64)

        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        self.n_out = n_out
        self.x_shape = x_shape

        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        self.conv1 = _Conv(n_chan, hid_channels, kernel_size, **cnn_kwargs)
        self.conv2 = _Conv(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.conv3 = _Conv(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        # If input image is 64x64 do fourth convolution
        if self.img_size[1] == self.img_size[2] == 64:
            self.conv_64 = _Conv(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        # Fully connected layers
        self.lin1 = _Linear(np.product(self.reshape), hidden_dim)
        self.lin2 = _Linear(hidden_dim, hidden_dim)
        self.lin3 = _Linear(hidden_dim, self.n_out)

    def forward(self, x):
        batch_size = x.size(0)

        # Convolutional layers with ReLu activations
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        if self.img_size[1] == self.img_size[2] == 64:
            x = torch.relu(self.conv_64(x))

        # Fully connected layers with ReLu activations
        x = x.view((batch_size, -1))
        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin2(x))
        x = self.lin3(x)

        return x


class ReversedSimpleCNN(SimpleCNN):
    """
    Reversed version of `WideResNet`. Convolution layers are replaced with
    Transposed Convolutions.
    """

    def __init__(self, x_shape, n_out):
        super().__init__(x_shape, n_out,
                         _Conv=ReversedConv2d, _Linear=ReversedLinear)

    def forward(self, x):
        # Litteraly reversing all
        batch_size = x.size(0)

        x = torch.relu(self.lin3(x))
        x = torch.relu(self.lin2(x))
        x = torch.relu(self.lin1(x))
        x = x.view(batch_size, *self.reshape)

        if self.img_size[1] == self.img_size[2] == 64:
            x = torch.relu(self.conv_64(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv2(x))
        x = self.conv1(x)

        return x
