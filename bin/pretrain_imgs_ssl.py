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


import numpy as np
from functools import partial
from utils.data.ssldata import get_dataset, get_train_dev_test_ssl
from utils.data.helpers import train_dev_split
from skorch.callbacks import EarlyStopping

svhn_train, _, svhn_test = get_train_dev_test_ssl("svhn", dev_size=0)
cifar10_train, _, cifar10_test = get_train_dev_test_ssl("cifar10", dev_size=0)
mnist_train, _, mnist_test = get_train_dev_test_ssl("mnist", dev_size=0)

from econvcnp.transformers.neuralproc.datasplit import GridCntxtTrgtGetter, RandomMasker, no_masker, half_masker
from utils.data.tsdata import get_timeseries_dataset, SparseMultiTimeSeriesDataset

get_cntxt_trgt_test = GridCntxtTrgtGetter(context_masker=RandomMasker(min_nnz=0.01, max_nnz=0.50),
                                          target_masker=no_masker,
                                          is_add_cntxts_to_trgts=False)  # don't context points to tagrtes

get_cntxt_trgt_feat = GridCntxtTrgtGetter(context_masker=no_masker,
                                          target_masker=no_masker,
                                          is_add_cntxts_to_trgts=False)  # don't context points to tagrtes

get_cntxt_trgt = GridCntxtTrgtGetter(context_masker=RandomMasker(min_nnz=0.01, max_nnz=0.99),
                                     target_masker=RandomMasker(min_nnz=0.50, max_nnz=0.99),
                                     is_add_cntxts_to_trgts=False)  # don't context points to tagrtes


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


datasets = dict(svhn=(svhn_train, svhn_test),
                cifar10=(cifar10_train, cifar10_test),
                mnist=(mnist_train, mnist_test)
                )


data_specific_kwargs = dict(svhn=dict(y_dim=svhn_train.shape[0]),
                            cifar10=dict(y_dim=cifar10_train.shape[0]),
                            mnist=dict(y_dim=mnist_train.shape[0]))

X_DIM = 2  # 2D spatial input
#Y_DIM = data.shape[0]
N_TARGETS = 10

#label_percentages = [N_TARGETS, N_TARGETS*2, 0.01, 0.05, 0.1, 0.3, 0.5, 1]


from econvcnp.transformers import AttentiveNeuralProcess, GridNeuralProcessSSLLoss, GridConvNeuralProcess
from econvcnp.predefined import UnetCNN, CNN, MLP, SinusoidalEncodings, merge_flat_input
from econvcnp.transformers.neuralproc.datasplit import precomputed_cntxt_trgt_split
from copy import deepcopy


models = {}

m_clf = lambda y_dim: partial(GridConvNeuralProcess,
                              y_dim=y_dim,
                              r_dim=64,
                              output_range=(0, 1),
                              is_clf_features=False,
                              Classifier=partial(MLP, input_size=256, output_size=N_TARGETS,
                                                 dropout=0.,
                                                 hidden_size=128, n_hidden_layers=3, is_res=True),
                              TmpSelfAttn=partial(
                                  UnetCNN,
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

                              ))

# initialize one model for each dataset
models["ssl_classifier_gnp_large_unet"] = m_clf

m_trnsf = lambda y_dim: partial(GridConvNeuralProcess,
                                y_dim=y_dim,
                                r_dim=64,
                                output_range=(0, 1),
                                Classifier=None,
                                TmpSelfAttn=partial(
                                    UnetCNN,
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
                                    _is_summary=True)
                                )


models["transformer_gnp_large_unet"] = m_trnsf

from utils.helpers import count_parameters
for k, v in models.items():
    print(k, "- N Param:", count_parameters(v(y_dim=3)()))

from ntbks_helpers import train_models_
from skorch.dataset import CVSplit
from utils.data.ssldata import get_train_dev_test_ssl

N_EPOCHS = 100
BATCH_SIZE = 32
IS_RETRAIN = True  # if false load precomputed
chckpnt_dirname = "results/notebooks/neural_process_images/"

from econvcnp.utils.helpers import HyperparameterInterpolator


def load_pretrained_(models, data_name, datasets, data_specific_kwargs):

    # ALREADY INITALIZE TO BE ABLE TO LOAD
    models["ssl_classifier_gnp_large_unet"] = m_clf(**data_specific_kwargs[data_name])()
    models["transformer_gnp_large_unet"] = m_trnsf(**data_specific_kwargs[data_name])()

    # load all transformers
    for k, m in models.items():
        if "transformer" not in k:
            continue

        out = train_models_({data_name: datasets[data_name]}, {k: m},
                            chckpnt_dirname=chckpnt_dirname,
                            is_retrain=False)

        pretrained_model = out[list(out.keys())[0]].module_
        model_dict = models[k.replace("transformer", "ssl_classifier")].state_dict()
        model_dict.update(pretrained_model.state_dict())
        models[k.replace("transformer", "ssl_classifier")].load_state_dict(model_dict)


from skorch.callbacks import Freezer, LRScheduler


data_trainers = {}

for data_name, (data_train, data_test) in datasets.items():

    load_pretrained_(models, data_name, datasets, data_specific_kwargs)

    is_all_labels = True

    data_train, _, data_test = get_train_dev_test_ssl(data_name,
                                                      dev_size=0,
                                                      is_augment=True,
                                                      is_all_labels=is_all_labels)

    # add test as unlabeled data
    data_train.data = np.concatenate([data_train.data, data_test.data], axis=0)
    if data_name == "mnist":
        data_train.data = torch.from_numpy(data_train.data)  # mnist to have data as tensor

    data_train.targets = np.concatenate([data_train.targets, -1 * np.ones_like(data_test.targets)], axis=0)

    is_ssl_only = False
    if is_ssl_only:
        idcs = data_train.targets != -1
        data_train.data = data_train.data[torch.from_numpy(idcs)]
        data_train.targets = data_train.targets[idcs]
        sfx_ssl = "_ssl_only"
    else:
        sfx_ssl = ""

    n_max_elements = 1024

    label_perc = (data_train.targets != -1).sum() / len(data_train.targets)
    sfx_lab_perc = "" if label_perc is None else "_labperc"

    from econvcnp.utils.helpers import HyperparameterInterpolator
    n_steps_per_epoch = len(data_train) // BATCH_SIZE
    get_lambda_clf = HyperparameterInterpolator(1, 50, N_EPOCHS * n_steps_per_epoch, mode="linear")

    data_trainers.update(train_models_({data_name: (data_train, data_test)},
                                       {k + "_finetune_is_all_labels": m for k, m in models.items()
                                        if "ssl_classifier" in k},
                                       criterion=partial(GridNeuralProcessSSLLoss,
                                                         n_max_elements=n_max_elements,
                                                         label_perc=label_perc,
                                                         is_ssl_only=False,
                                                         get_lambda_unsup=lambda: 1,
                                                         get_lambda_ent=lambda: 0.5,
                                                         get_lambda_sup=lambda: get_lambda_clf(True),
                                                         get_lambda_neg_cons=lambda: 0.5,
                                                         min_sigma=0.1
                                                         ),
                                       patience=15,
                                       chckpnt_dirname=chckpnt_dirname,
                                       max_epochs=N_EPOCHS,
                                       batch_size=BATCH_SIZE,
                                       is_retrain=IS_RETRAIN,
                                       is_monitor_acc=True,
                                       callbacks=[],
                                       iterator_train__collate_fn=cntxt_trgt_collate(get_cntxt_trgt, is_grided=True, is_repeat_batch=True),
                                       iterator_valid__collate_fn=cntxt_trgt_collate(get_cntxt_trgt_feat, is_grided=True),
                                       mode="classifier",
                                       ))


for k, t in data_trainers.items():
    for e, h in enumerate(t.history[::-1]):
        if h["valid_loss_best"]:
            print(k, "epoch:", len(t.history) - e,
                  "val_loss:", h["valid_loss"],
                  "val_acc:", h["valid_acc"])
            break

print("finetune_sup")

data_trainers = {}
