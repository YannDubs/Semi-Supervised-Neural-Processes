from functools import partial
import itertools
from copy import deepcopy

import torch
import torch.nn as nn
from sklearn.model_selection import cross_val_score

from skorch.callbacks import ProgressBar, Checkpoint, EarlyStopping
from skssl.training import NeuralNetClassifier, NeuralNetTransformer


def zip_repeat(*args):
    """Reapeat all elements to be the same size of longest list and then zip."""
    # squeeze
    args = [a[0] if isinstance(a, list) and len(a) == 1 else a for a in args]
    lengths = [len(a) for a in args if isinstance(a, list)]
    if len(lengths) != 0:
        max_size = max(lengths)
        assert len([l for l in lengths if l not in [max_size]]) == 0, "All list should be same lengths"
    else:
        max_size = 1
    args = [a if isinstance(a, list) and len(a) == max_size else itertools.repeat(a, max_size)
            for a in args]
    return zip(*args)


def train_models_(datas, models, *args,
                  data_specific_kwargs={},
                  chckpnt_dirname=None,
                  is_retrain=False,
                  is_progress_bar=True,
                  batch_size=64,
                  max_epochs=50,
                  lr=1e-3,
                  patience=15,
                  mode="classifier",
                  callbacks=[],
                  is_monitor_acc=False,
                  is_cross_validate=False,
                  **kwargs):
    """Train or loads IN PLACE a dictionary containing a model and a datasets"""
    trainers = dict()
    callbacks_dflt = callbacks

    # if isinstance(models, dict):
    #models = [(k, v) for k, v in models.items()]

    # if isinstance(datas, dict):
    #datas = [(k, v) for k, v in datas.items()]

    for data_name, data_train in datas.items():

        for model_name, m in models.items():
            # for (data_name, data_train), (model_name, m) in zip_repeat(datas, models):

            callbacks = deepcopy(callbacks_dflt)

            try:
                data_train, data_test = data_train
            except:
                data_train, data_test = data_train, None

            suffix = data_name + "/" + model_name

            print()
            print("--- {} {} ---".format("Training" if is_retrain else "Loading", suffix))
            print()

            if len(data_specific_kwargs) != 0:
                m = partial(m, **data_specific_kwargs[data_name])

            if chckpnt_dirname is not None:
                if is_monitor_acc:
                    chckpt = Checkpoint(dirname=chckpnt_dirname + suffix,
                                        monitor='valid_acc_best')
                else:
                    chckpt = Checkpoint(dirname=chckpnt_dirname + suffix,
                                        monitor='valid_loss_best')
                callbacks.append(chckpt)

            if is_progress_bar:
                callbacks.append(ProgressBar())
            if patience is not None:
                if is_monitor_acc:
                    callbacks.append(EarlyStopping(patience=15,
                                                   monitor='valid_acc',
                                                   lower_is_better=False))
                else:
                    callbacks.append(EarlyStopping(patience=patience))

            if mode == "classifier":
                if "criterion" not in kwargs:
                    kwargs["criterion"] = nn.CrossEntropyLoss
                model = NeuralNetClassifier(m, *args,
                                            iterator_train__shuffle=True,  # shuffle iterator
                                            devset=data_test,
                                            max_epochs=max_epochs,
                                            batch_size=batch_size,
                                            lr=lr,
                                            callbacks=callbacks,
                                            **kwargs)
            elif mode == "transformer":
                model = NeuralNetTransformer(m, *args,
                                             devset=data_test,
                                             max_epochs=max_epochs,
                                             batch_size=batch_size,
                                             lr=lr,
                                             callbacks=callbacks,
                                             **kwargs)
            else:
                raise ValueError("Unkown mode={}.".format(mode))

            if is_retrain:
                if is_cross_validate:
                    import pdb
                    pdb.set_trace()
                    model.cv_scores = cross_val_score(model, data_train.X, data_train.y)

                else:
                    model.fit(data_train)

            if chckpnt_dirname is not None:
                # load in all case => even when training loads the best checkpoint
                model.initialize()
                model.load_params(checkpoint=chckpt)

            model.module_.cpu()  # make sure on cpu
            torch.cuda.empty_cache()  # empty cache for next run

            trainers[suffix] = model

            try:
                for epoch, history in enumerate(model.history[::-1]):
                    if history["valid_loss_best"]:
                        print(suffix, "best epoch:", len(model.history[::-1]) - epoch,
                              "val_loss:", history["valid_loss"])
                        break
            except:
                pass

    return trainers
