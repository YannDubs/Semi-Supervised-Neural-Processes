import os
import torch

import pandas as pd

from .base import BaseDataset

#! WIP. Wait until torchaudio does not require sox
class UrbanSound8K(BaseDataset):
    """UrbanSound8K dataset.

    Notes
    -----
    - has to be manually downloaded and untared from https://urbansounddataset.weebly.com/ 
    - credits : https://pytorch.org/tutorials/beginner/audio_classifier_tutorial.html?highlight=audio
    - usually UrbanSound8Kis evaluated with 10 fold cross validation. To be consistent with the
    rest of the library, we use a trick such that the `seed - base_seed` tells you what fold
    to use. So the seeed should only var between [base_seed, base_seed+10].

    Parameters
    ----------
    """

    base_seed = 123
    metadata = "/UrbanSound8K/metadata/UrbanSound8K.csv"
    file_path = "./UrbanSound8K/audio/"

    def __init__(self, **kwargs):
        import torchaudio

        super().__init__(self, **kwargs)
        self.load_data()
        test_fold = self.base_seed - self.seed

        if test_fold < 0 and test_fold > 10:
            raise ValueError(
                f"seed={self.seed} but needs to be between [{self.base_seed}, {self.base_seed}+10]"
            )

        if self.split == "train":
            self.folds = [i for i in range(10) if i != test_fold]
        elif self.split == "test":
            self.folds = [test_fold]

    def set_train_transforms(self):
        """Return the training transformation."""
        pass

    def set_test_transforms(self):
        """Return the testing transformation."""
        pass

    def load_data(self):
        csvData = pd.read_csv(os.path.join(self.root, self.metadata))
        # initialize lists to hold file names, labels, and folder numbers
        self.file_names = []
        self.labels = []
        self.folders = []
        # loop through the csv entries and only add entries from folders in the folder list
        for i in range(0, len(csvData)):
            if csvData.iloc[i, 5] in self.folds:
                self.file_names.append(csvData.iloc[i, 0])
                self.labels.append(csvData.iloc[i, 6])
                self.folders.append(csvData.iloc[i, 5])

    def __getitem__(self, index):
        breakpoint()

        # format the file path and load the file
        path = os.path.join(
            self.root, self.file_path, f"fold{self.folders[index]}", self.file_names[index]
        )
        sound = torchaudio.load(path, out=None, normalization=True)
        # load returns a tensor with the sound data and the sampling frequency (44.1kHz for UrbanSound8K)
        soundData = self.mixer(sound[0])
        # downsample the audio to ~8kHz
        tempData = torch.zeros([160000, 1])  # tempData accounts for audio clips that are too short
        if soundData.numel() < 160000:
            tempData[: soundData.numel()] = soundData[:]
        else:
            tempData[:] = soundData[:160000]

        soundData = tempData
        soundFormatted = torch.zeros([32000, 1])
        soundFormatted[:32000] = soundData[::5]  # take every fifth sample of soundData
        soundFormatted = soundFormatted.permute(1, 0)
        return soundFormatted, self.labels[index]

    def __len__(self):
        return len(self.file_names)
