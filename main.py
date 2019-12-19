import logging


from utils.data import get_train_dev_test_datasets
from utils.helpers import (
    get_exponential_decay_gamma,
    format_container,
    hyperparam_to_path,
    _single_epoch_skipvalid,
    _scoring,
    get_hyperparam_names,
)

logger = logging.getLogger(__name__)

FILE_SCORE = "score.csv"
FILE_LOGS = "log.csv"
FILE_CLF_REP = "clf_report.csv"


@hydra.main(config_path="conf/config.yaml")
def main(args):
    """Main function for training and testing representations."""
    paths = update_config_(args)
    set_seed(args.seed)

    if not args.datasize.is_valid_all_epochs:
        # don't validate all epochs when validation >>> training
        rm_valid_epochs_()

    logger.info("Loading the dataset ...")
    datasets = get_train_dev_test_datasets(
        args.dataset.name,
        args.dataset.type,
        valid_size=args.dataset.valid_size,
        **args.dataset.kwargs,
    )

    # DROP FEATURES
    for k, dataset in datasets:
        datasets.drop_features_(drop_size=args.dataset.drop_size)

    # INTERPOLATE
    if args.interpolator is not None:

        interpolator = fit_interpolator(args, datasets)

        logger.info("Interpolating the dataset ...")
        datasets = transform(interpolator, datasets)


def update_config_(args):
    """Update the configuration values based on other values."""

    # increment the seed at each run
    args.seed = args.seed + args.run

    hyperparam_path = hyperparam_to_path(args.hyperparameters)
    args.paths = format_container(args.paths, dict(hyperparam_path=hyperparam_path))

    # multiply the number of examples by a factor size. Used to have number of examples depending
    # on number of labels. Usually factor is 1.
    args.datasize.n_examples = args.datasize.factor * args.datasize.n_examples

    if args.train.gamma is None:
        args.train.gamma = get_exponential_decay_gamma(
            args.train.scheduling_factor, args.train.kwargs.max_epochs
        )


def rm_valid_epochs_():
    """Don't validate every epoch."""
    NeuralNetTransformer._single_epoch = _single_epoch_skipvalid
    NeuralNetClassifier._single_epoch = _single_epoch_skipvalid
    skorch.callbacks.scoring.ScoringBase._scoring = _scoring


def fit_interpolator(args, datasets):
    """Fits an interpolator on the datasets."""
    logger.info("Instantiating the interpolator ...")

    Interpolator = get_Interpolator(
        args.interpolator.name,
        type=args.datset.type,
        seed=args.seed,
        **args.interpolator.architecture,
    )

    logger.info("Training the interpolator ...")

    interpolator = train_load(
        Interpolator,
        datasets,
        chckpnt_dirnames=append_sffx(args.paths["chckpnt_dirnames"], "interpolator/"),
        is_train=args.train.is_train_interpolator,
        tensorboard_dir=args.paths["tensorboard_curr_dir"] + "interpolator/",
        **args.train.kwargs,
    )

    return interpolator
