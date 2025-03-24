"""Protein dataset class."""
import gzip
import os
import pickle
from pathlib import Path
from glob import glob
from typing import Optional, Union
from functools import lru_cache
import numpy as np
import pandas as pd
import torch

from src.data.transform import FeatureTransform, BioFeatureTransform


class ProteinTrainingDataset(torch.utils.data.Dataset):
    """Random access to pickle protein objects of dataset.

    dict_keys(['atom_positions', 'aatype', 'atom_mask', 'residue_index', 'chain_index', 'b_factors'])

    Note that each value is a ndarray in shape (L, *), for example:
        'atom_positions': (L, 37, 3)
    """

    def __init__(self,
                 path_to_dataset: Union[Path, str],
                 transform: Optional[FeatureTransform] = None,
                 training: bool = True,
                 ):
        super().__init__()
        path_to_dataset = os.path.expanduser(path_to_dataset)

        if os.path.isfile(path_to_dataset):  # path to csv file
            assert path_to_dataset.endswith('.csv'), f"Invalid file extension: {path_to_dataset} (have to be .csv)"
            self._df = pd.read_csv(path_to_dataset)
            self._df.sort_values('modeled_seq_len', ascending=False)
            self._data = self._df['processed_path'].tolist()

        self.data = np.asarray(self._data)
        self.transform = transform
        self.training = training  # not implemented yet

    @property
    def num_samples(self):
        return len(self.data)

    def len(self):
        return self.__len__()

    def __len__(self):
        return self.num_samples

    def get(self, idx):
        return self.__getitem__(idx)

    @lru_cache(maxsize=100)
    def __getitem__(self, idx):
        """return single pyg.Data() instance
        """
        data_path = self.data[idx]
        accession_code = os.path.basename(data_path).split('.')[0]

        # Load pickled protein
        with open(data_path, 'rb') as f:
            data_object = pickle.load(f)

        # Apply data transform
        if self.transform is not None:
            if 'chain_ids' in data_object.keys():
                data_object.pop('chain_ids')
            data_object = self.transform(data_object)

        data_object['accession_code'] = accession_code
        return data_object  # dict of arrays


class BioTrainingDataset(torch.utils.data.Dataset):
    def __init__(self,
                 path_to_dataset: Union[Path, str],
                 transform: Optional[BioFeatureTransform] = None,
                 training: bool = True,
                 ):
        super().__init__()
        path_to_dataset = os.path.expanduser(path_to_dataset)

        assert path_to_dataset.endswith('.csv'), f"Invalid file extension: {path_to_dataset} (have to be .csv)"
        self._df = pd.read_csv(path_to_dataset)
        self._df.sort_values('token_num', ascending=False)
        self._data = self._df['processed_path'].tolist()

        self.data = np.asarray(self._data)
        self.transform = transform
        self.training = training  # not implemented yet

    @property
    def num_samples(self):
        return len(self.data)

    def len(self):
        return self.__len__()

    def __len__(self):
        return self.num_samples

    def get(self, idx):
        return self.__getitem__(idx)

    @lru_cache(maxsize=100)
    def __getitem__(self, idx):
        data_path = self.data[idx]
        accession_code = os.path.basename(data_path).split('.')[0]

        with gzip.open(data_path, 'rb') as f:
            data_object = pickle.load(f)

        if self.transform is not None:
            data_object = self.transform(data_object)
        data_object['accession_code'] = accession_code
        return data_object


class InferenceDataset(torch.utils.data.Dataset):
    """Random access to pickle protein objects of dataset.

    dict_keys(['atom_positions', 'aatype', 'atom_mask', 'residue_index', 'chain_index', 'b_factors'])

    Note that each value is a ndarray in shape (L, *), for example:
        'atom_positions': (L, 37, 3)
    """

    def __init__(self,
                 path_to_dataset: Union[Path, str],
                 suffix: str = 'pkl',
                 transform: Optional[FeatureTransform] = None,
                 ):
        super().__init__()
        path_to_dataset = os.path.expanduser(path_to_dataset)

        if os.path.isfile(path_to_dataset):  # path to csv file
            assert path_to_dataset.endswith('.csv'), f"Invalid file extension: {path_to_dataset} (have to be .csv)"
            self._df = pd.read_csv(path_to_dataset)
            self._df.sort_values('modeled_seq_len', ascending=False)
            self._data = self._df['processed_path'].tolist()
        elif os.path.isdir(path_to_dataset):
            self._data = glob(os.path.join(path_to_dataset, f'*.{suffix}'))

        self.data = np.asarray(self._data)
        self.transform = transform

    @property
    def num_samples(self):
        return len(self.data)

    def len(self):
        return self.__len__()

    def __len__(self):
        return self.num_samples

    def get(self, idx):
        return self.__getitem__(idx)

    @lru_cache(maxsize=100)
    def __getitem__(self, idx):
        """return single pyg.Data() instance
        """
        data_path = self.data[idx]
        accession_code = os.path.basename(data_path).split('.')[0]

        # Load pickled protein
        with open(data_path, 'rb') as f:
            data_object = pickle.load(f)

        # Apply data transform
        if self.transform is not None:
            if 'chain_ids' in data_object.keys():
                data_object.pop('chain_ids')
            data_object = self.transform(data_object)

        data_object['accession_code'] = accession_code
        return data_object  # dict of arrays
