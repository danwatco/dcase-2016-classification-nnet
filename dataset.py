#!/usr/bin/env python3
##################################################
## Custom dataset loader to load training and test dataset.
## Designed to be run on a HPC cluster.
##################################################
##################################################
## Author: Dan Watkinson
##################################################

from torch.utils.data.dataset import Dataset
import torch
from torchvision import transforms

import numpy as np
from pathlib import Path
import pandas as pd

class DCASE(Dataset):
    def __init__(self, root_dir: str, clip_duration: int, transform=None):
        self._root_dir = Path(root_dir)
        self._labels = pd.read_csv((self._root_dir / 'labels.csv'), names=['file', 'label'])
        # print(self._labels.head())
        self._labels['label'] = self._labels.label.astype('category').cat.codes.astype('int') #create categorical labels
        # print(self._labels['label'])
        self._clip_duration = clip_duration
        self._total_duration = 30 #DCASE audio length is 30s
        self.transform = transform
        self._data_len = len(self._labels)

    def __getitem__(self, index):
        #reading spectrograms
        filename, label = self._labels.iloc[index]
        # print(filename)
        filepath = self._root_dir / 'audio'/ filename
        spec = torch.from_numpy(np.load(filepath))

        #splitting spec

        spec = self.__trim__(spec)
        if self.transform:
            spec = self.transform(spec)
        return spec, label

    def __trim__(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Trims spectrogram into multiple clips of length specified in self._num_clips
        :param spec: tensor containing spectrogram of full audio signal of shape [1, 60, 1501]
        :return: tensor containing stacked spectrograms of shape [num_clips, 60, clip_length] ([10, 60, 150] with 3s clips)
        """
        time_steps = spec.size(-1)
        self._num_clips = self._total_duration // self._clip_duration
        time_interval = int(time_steps // self._num_clips)
        all_clips = []
        for clip_idx in range(self._num_clips):
            start = clip_idx * time_interval
            end = start + time_interval
            spec_clip = spec[:, start:end]
            # if self.transform:
            #     spec_clip = self.transform(spec_clip)
            #spec_clip = torch.squeeze(spec_clip)
            all_clips.append(spec_clip)

        specs = torch.stack(all_clips)
        return specs

    def get_num_clips(self) -> int:
        """
        Gets number of clips the raw audio has been split into
        :return: self._num_clips of type int
        """
        return self._num_clips

    def __len__(self):
        return self._data_len


class DCASE_Non_Full(Dataset):
    def __init__(self, root_dir: str, clip_duration: int, file_filter = None):
        self._root_dir = Path(root_dir)
        self._labels = pd.read_csv((self._root_dir / 'labels.csv'), names=['file', 'label'])
        if file_filter is not None:
            self._labels = self._labels[self._labels['file'].isin(file_filter)]
        self._labels['label'] = self._labels.label.astype('category').cat.codes.astype('int') #create categorical labels
        self._clip_duration = clip_duration
        self._total_duration = 30 #DCASE audio length is 30s

        self._data_len = len(self._labels)
        
        unique = []

        for i, row in self._labels.iterrows():
            identifier = row['file'].split('_')[0] # selects a000 from a000_0_30
            letter = identifier[0]
            number = identifier[1:]
            unique.append((letter, number))
        
        self._labels['unique'] = unique

    def __getitem__(self, index):
        #reading spectrograms
        filename, label, _ = self._labels.iloc[index]
        filepath = self._root_dir / 'audio'/ filename
        spec = torch.from_numpy(np.load(filepath))

        #splitting spec
        spec = self.__trim__(spec)
        return spec, label

    def __trim__(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Trims spectrogram into multiple clips of length specified in self._num_clips
        :param spec: tensor containing spectrogram of full audio signal of shape [1, 60, 1501]
        :return: tensor containing stacked spectrograms of shape [num_clips, 60, clip_length] ([10, 60, 150] with 3s clips)
        """
        time_steps = spec.size(-1)
        self._num_clips = self._total_duration // self._clip_duration
        time_interval = int(time_steps // self._num_clips)
        all_clips = []
        for clip_idx in range(self._num_clips):
            start = clip_idx * time_interval
            end = start + time_interval
            spec_clip = spec[:, start:end]
            #spec_clip = torch.squeeze(spec_clip)
            all_clips.append(spec_clip)

        specs = torch.stack(all_clips)
        return specs

    def get_num_clips(self) -> int:
        """
        Gets number of clips the raw audio has been split into
        :return: self._num_clips of type int
        """
        return self._num_clips

    def __len__(self):
        return self._data_len
