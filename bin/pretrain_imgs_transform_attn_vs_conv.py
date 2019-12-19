import os
from os.path import dirname, abspath
base_dir = dirname(dirname(abspath(__file__)))
os.chdir(base_dir)

import sys
sys.path.append("notebooks")
sys.path.append(".")

import torch

N_THREADS = 8
torch.set_num_threads(N_THREADS)


from functools import partial
from utils.data.ssldata import get_dataset, get_train_dev_test_ssl
from utils.data.helpers import train_dev_split


from wildml.transformers.neuralproc.datasplit import GridCntxtTrgtGetter, RandomMasker, no_masker, half_masker
from utils.data.tsdata import get_timeseries_dataset, SparseMultiTimeSeriesDataset

from wildml.transformers import AttentiveNeuralProcess, NeuralProcessLoss, GridConvNeuralProcess, GridNeuralProcessLoss
from wildml.predefined import UnetCNN, CNN, SelfAttention, MLP, SinusoidalEncodings, merge_flat_input
from wildml.transformers.neuralproc.datasplit import precomputed_cntxt_trgt_split


def cntxt_trgt_collate(get_cntxt_trgt, is_repeat_batch=False, is_grided=False):
    def mycollate(batch):

        if isinstance(batch[0][0], dict):
            min_length = min([v.size(0) for b in batch for k, v in b[0].items() if "X" in k])
            # chose first min_legth of each (assumes that randomized)

            batch = [({k: v[:min_length, ...] for k, v in b[0].items()}, b[1]) for b in batch]
            collated = torch.utils.data.dataloader.default_collate(batch)

            X = collated[0]["X"]
            y = collated[0]["y"]
        else:
            collated = torch.utils.data.dataloader.default_collate(batch)

            X = collated[0]
            y = None
            collated[0] = dict()

        if is_repeat_batch:
            X = torch.cat([X, X], dim=0)
            if y is not None:
                y = torch.cat([y, y], dim=0)
            collated[1] = torch.cat([collated[1], collated[1]], dim=0)  # targets

        if is_grided:
            collated = (dict(), collated[1])
            (collated[0]["X"], collated[0]["mask_context"], collated[0]["mask_target"]
             ) = get_cntxt_trgt(X, y,
                                is_grided=True)

        else:
            collated[0]["X"], collated[0]["y"], collated[0]["X_trgt"], collated[0]["y_trgt"] = get_cntxt_trgt(X, y)

        return collated
    return mycollate


if __name__ == "__main__":
    get_cntxt_trgt_test = GridCntxtTrgtGetter(context_masker=RandomMasker(min_nnz=0.01, max_nnz=0.50),
                                              target_masker=no_masker,
                                              is_add_cntxts_to_trgts=False)  # don't context points to tagrtes

    get_cntxt_trgt_feat = GridCntxtTrgtGetter(context_masker=no_masker,
                                              target_masker=no_masker,
                                              is_add_cntxts_to_trgts=False)  # don't context points to tagrtes

    get_cntxt_trgt = GridCntxtTrgtGetter(context_masker=RandomMasker(min_nnz=0.01, max_nnz=0.50),
                                         target_masker=RandomMasker(min_nnz=0.50, max_nnz=0.99),
                                         is_add_cntxts_to_trgts=False)  # don't context points to tagrtes

    cifar_train, _, cifar_test = get_train_dev_test_ssl("cifar10", dev_size=0)
    svhn_train, _, svhn_test = get_train_dev_test_ssl("svhn", dev_size=0)
    # mnist_train, _, mnist_test = get_train_dev_test_ssl("mnist", dev_size=0) # mnist trained already

    datasets = dict(svhn=(svhn_train, svhn_test),
                    cifar10=(cifar_train, cifar_test))

    data_specific_kwargs = dict(svhn=dict(y_dim=svhn_train.shape[0]),
                                cifar10=dict(y_dim=svhn_train.shape[0]))

    X_DIM = 2  # 2D spatial input
    #Y_DIM = data.shape[0]
    N_TARGETS = None

    models_grided = {}

    unet = partial(UnetCNN,
                   Conv=torch.nn.Conv2d,
                   Pool=torch.nn.MaxPool2d,
                   upsample_mode="bilinear",
                   n_layers=18,
                   is_double_conv=True,
                   is_depth_separable=True,
                   Normalization=torch.nn.BatchNorm2d,
                   is_chan_last=True,
                   bottleneck=None,
                   kernel_size=7,
                   max_nchannels=256,
                   is_force_same_bottleneck=True,
                   _is_summary=True,
                   )

    gnp_large_kwargs = dict(r_dim=64,
                            output_range=(0, 1),
                            is_normalize=True,
                            TmpSelfAttn=unet)

    # initialize one model for each dataset
    models_grided["transformer_gnp_large_unet"] = partial(GridConvNeuralProcess,
                                                          **gnp_large_kwargs)

    from utils.helpers import count_parameters

    for k, v in models_grided.items():
        print(k, "- N Param:", count_parameters(v(y_dim=3)))

    N_EPOCHS = 100
    BATCH_SIZE = 16
    IS_RETRAIN = True  # if false load precomputed
    chckpnt_dirname = "results/notebooks/neural_process_images/"

    from ntbks_helpers import train_models_

    _ = train_models_(datasets,
                      models_grided,
                      GridNeuralProcessLoss,
                      data_specific_kwargs=data_specific_kwargs,
                      patience=5,
                      chckpnt_dirname=chckpnt_dirname,  # chckpnt_dirname,
                      max_epochs=N_EPOCHS,
                      batch_size=BATCH_SIZE,
                      is_retrain=IS_RETRAIN,
                      callbacks=[],
                      iterator_train__collate_fn=cntxt_trgt_collate(get_cntxt_trgt,
                                                                    is_grided=True,
                                                                    is_repeat_batch=True),
                      iterator_valid__collate_fn=cntxt_trgt_collate(get_cntxt_trgt_test,
                                                                    is_grided=True),
                      mode="transformer")
