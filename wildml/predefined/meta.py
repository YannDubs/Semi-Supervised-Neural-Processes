import torch
import torch.nn as nn

from essl.predefined import MLP
from essl.predefined.cnn import FCONVS
from essl.utils.initialization import weights_init
from essl.utils.helpers import mask_and_apply, prod, clip_interval


__all__ = ["merge_flat_input", "discard_ith_arg"]

# META ENCODERS
class DiscardIthArg(nn.Module):
    """
    Helper module which discard the i^th argument of the constructor and forward,
    before being given to `To`.
    """

    def __init__(self, *args, i=0, To=nn.Identity, **kwargs):
        super().__init__()
        self.i = i
        self.destination = To(*self.filter_args(*args), **kwargs)

    def filter_args(self, *args):
        return [arg for i, arg in enumerate(args) if i != self.i]

    def forward(self, *args, **kwargs):
        return self.destination(*self.filter_args(*args), **kwargs)


def discard_ith_arg(Module, i, **kwargs):
    def discarded_arg(*args, **kwargs2):
        return DiscardIthArg(*args, i=i, To=Module, **kwargs, **kwargs2)

    return discarded_arg


class MergeFlatInputs(nn.Module):
    """
    Extend a module to take 2 flat inputs. It simply returns
    the concatenated flat inputs to the module `module({x1; x2})`.

    Parameters
    ----------
    FlatModule: nn.Module
        Module which takes a flat inputs.

    x1_dim: int
        Dimensionality of the first flat inputs.

    x2_dim: int
        Dimensionality of the second flat inputs.

    n_out: int
        Size of ouput.

    is_sum_merge : bool, optional
        Whether to transform `flat_input` by an MLP first (if need to resize),
        then sum to `X` (instead of concatenating): useful if the difference in
        dimension between both inputs is very large => don't want one layer to
        depend only on a few dimension of a large input.

    kwargs:
        Additional arguments to FlatModule.
    """

    def __init__(self, FlatModule, x2_dim, x1_dim, n_out, is_sum_merge=False, **kwargs):
        super().__init__()
        self.is_sum_merge = is_sum_merge

        if self.is_sum_merge:
            dim = x1_dim
            self.resizer = MLP(x2_dim, dim)  # transform to be the correct size
        else:
            dim = x1_dim + x2_dim

        self.flat_module = FlatModule(dim, n_out, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, X, flat_x):
        if self.is_sum_merge:
            flat_x = self.resizer(flat_x)
            # use activation because if not 2 linear layers in a row => useless computation
            out = torch.relu(X + flat_x)
        else:
            out = torch.cat((X, flat_x), dim=-1)

        return self.flat_module(out)


class MergeMetaCNN(nn.Module):
    """
    Extend a CNN to take in some meta information in a flat vector `module({X; meta})` by generating 
    an initial filter using the metadata.

    Parameters
    ----------
    CNN: nn.Module
        CNN which takes a non flat inputs. It should be initialized with `CNN(in_channels, 
        out_channels)`.

    n_dim : int
        Number of dimensions of the input. The type of CNN to use will depend on it (Conv1d, 
        Conv2d, ...).

    flat_dim: int
        Dimensionality of the flat inputs.

    in_channels : int
        Number of channels in the input image.

    meta_kernel_size : int, optional
        Kernel size of the preprocessor derived from the metadata.

     meta_min_chan : int, optional
        Minimum number of channels outputed by the metadata preprocessing.

    kwargs:
        Additional arguments to CNN.
    """

    def __init__(
        self,
        Cnn,
        flat_dim,
        n_dim,
        in_channels,
        *args,
        meta_kernel_size=3,
        meta_min_chan=8,
        _meta_factor=0.1,
        **kwargs
    ):
        super().__init__()
        self.n_dim = n_dim
        self.tmp_channels = max(in_channels, meta_min_chan)
        self.in_channels = in_channels
        self._meta_factor = _meta_factor

        self.kernel_shape = tuple(meta_kernel_size for i in range(self.n_dim))
        self.weight_shape = (self.tmp_channels, self.in_channels) + self.kernel_shape
        self.bias_size = self.tmp_channels
        self.get_weights = MLP(
            flat_dim,
            prod(self.weight_shape) + self.bias_size,
            hidden_size=prod(self.weight_shape) + self.bias_size,
        )
        self.F_convNd = FCONVS[n_dim]

        self.cnn = Cnn(self.tmp_channels, *args, **kwargs)

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, X, flat_x):

        batch_size, in_channels, *grid_shape = X.shape

        weights = self.get_weights(flat_x)
        weights, bias = weights[..., : -self.bias_size], weights[..., -self.bias_size :]

        # to use a convolution which is has different parameters for all inputs, we stack
        # all batches on the channels, and use a group convolution
        weights = (
            weights.reshape(self.tmp_channels * batch_size, *self.weight_shape[1:])
            / prod(self.weight_shape[1:])
            * self._meta_factor
        )
        bias = bias.reshape(self.tmp_channels * batch_size) * self._meta_factor

        X = self.F_convNd(
            X.view(1, in_channels * batch_size, *grid_shape),
            weights,
            bias,
            padding=tuple(i // 2 for i in self.kernel_shape),
            groups=batch_size,
        )

        # Unstack for batch size
        X = X.view(batch_size, self.tmp_channels, *grid_shape)
        X = self.cnn(X)
        return X


class MergeNonFlatToFlat(nn.Module):
    """
    Extend a non flat input (e.g. image) to take in some meta information in a flat vector 
    `module({X; meta})` by preprocessing the metadata and concatenating it channelwise.
    Input should be as shape `(batch_size, in_channels, *grid_shape)`.

    Parameters
    ----------
    NonFlatModule : nn.Module
        Module which takes non flat inputs. It should be initialized with 
        `NonFlatModule(in_channels, *args)`.

    flat_dim : int
        Dimensionality of the flat inputs.
        
    in_channels : int
        Number of channels / y in the non grided input.

    args :
        Positional arguments to `NonFlatModule`.

    n_add_channels : int, optional
        Number of additional channels to add at each input location for the flat input. 
        
    is_clip_add_channels : bool, optional    
        Clip `n_add_channels` to be in the range given by `flat_dim` and `in_channels`.

    kwargs :
        Additional arguments to `NonFlatModule`.
    """

    def __init__(
        self,
        NonFlatModule,
        flat_dim,
        in_channels,
        *args,
        n_add_channels=3,
        is_clip_add_channels=True,
        **kwargs
    ):
        super().__init__()
        if is_clip_add_channels:
            n_add_channels = clip_interval(
                n_add_channels,
                flat_dim,
                in_channels,
                ["n_add_channels", "flat_dim", "in_channels"],
            )

        self.channel_adder = MLP(flat_dim, n_add_channels)
        self.non_flat_module = NonFlatModule(in_channels + n_add_channels, *args, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, X, flat_x):
        batch_size, in_channels, *grid_shape = X.shape
        add_channels = self.channel_adder(flat_x)
        exp_add_channels = add_channels.view(*add_channels.shape, *[1 for _ in grid_shape])
        exp_add_channels = exp_add_channels.expand(*add_channels.shape, *grid_shape)
        X = torch.cat((X, exp_add_channels), dim=1)
        return self.non_flat_module(X)


def merge_flat_input(Module, is_metacnn=True, **kwargs):
    """
    Extend a module to accept an additional flat input. I.e. the output should
    be constructed by `merge_flat_input(Module)(x_shape, flat_dim, *args, **kwargs)` and called
    by `Module(x, flat_input)`.

    Notes
    -----
    - if x_shape is an integer, it wraps the model around `MergeFlatInputs`.
    - if x_shape is a tuple, it wraps the model around `MergeMetaCNN` if `is_metacnn` else 
    `MergeNonFlatToFlat`. 
    """

    def merged_flat_input(x_shape, flat_dim, *args, **kwargs2):
        if isinstance(x_shape, int):
            return MergeFlatInputs(Module, flat_dim, x_shape, *args, **kwargs2, **kwargs)
        elif is_metacnn:
            return MergeMetaCNN(
                Module, flat_dim, len(x_shape) - 1, x_shape[0], *args, **kwargs2, **kwargs
            )
        else:
            return MergeNonFlatToFlat(Module, flat_dim, *args, **kwargs2, **kwargs)

    return merged_flat_input
