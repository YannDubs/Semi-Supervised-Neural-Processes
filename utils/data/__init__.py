def get_train_dev_test_datasets(dataset, data_type, valid_size=0.1, **kwargs):
    """Return the correct instantiated train, validation, test dataset
    
    Parameters
    ----------
    dataset : str
        Name of the dataset to load.

    data_type : {"imgs","texts","graphs","audio","timeseries"}
        Type of dataset.
    
    valid_size : float or int, optional
        Size of the validation set. If float, should be between 0.0 and 1.0 and represent the 
        proportion of the dataset. If int, represents the absolute number of valid samples.

    Returns
    -------
    datasets : dictionary of torch.utils.data.Dataset
        Dictionary of the `"train"`, `"valid"`, and `"valid"`.
    """
    datasets = dict()

    if data_type == "imgs":
        from .imgs import get_Dataset
    elif data_type == "texts":
        from .texts import get_Dataset
    elif data_type == "graphs":
        from .graphs import get_Dataset
    elif data_type == "audio":
        from .audio import get_Dataset
    elif data_type == "timeseries":
        from .timeseries import get_Dataset

    Dataset = get_Dataset(dataset)

    dataset = Dataset(split="train", **kwargs)
    datasets["train"], datasets["valid"] = dataset.train_test_split(test_size=valid_size)

    datasets["test"] = Dataset(dataset)(split="test", **kwargs)

    return datasets
