"""Protein dataset class."""
import os
import pickle
from pathlib import Path
from glob import glob
from random import random
from typing import Optional, Sequence, List, Union
from functools import lru_cache
import tree

import numpy as np
import pandas as pd
import torch

from src.utils.model_utils import uniform_random_rotation, rot_vec_mul

CA_IDX = 1
DTYPE_MAPPING = {
    'aatype': torch.long,
    'atom_positions': torch.double,
    'atom_mask': torch.double,
}
atom_types = [
    'N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
    'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
    'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
    'CZ3', 'NZ', 'OXT'
]
# define the element atomic number for each atom type
element_atomic_number = [
    7, 6, 6, 6, 8, 6, 6, 6, 8, 8, 16, 6,
    6, 6, 7, 7, 8, 8, 16, 6, 6, 6, 6,
    7, 7, 7, 8, 8, 6, 7, 7, 8, 6, 6,
    6, 7, 8
]


def convert_atom_id_name(atom_names: list[str]):
    """
        Converts unique atom_id names to integer of atom_name. need to be padded to length 4.
        Each character is encoded as ord(c) - 32
    """
    onehot_dict = {}
    for index, key in enumerate(range(64)):
        onehot = [0] * 64
        onehot[index] = 1
        onehot_dict[key] = onehot

    mol_encode = []
    for atom_name in atom_names:
        # [4, 64]
        atom_encode = []
        for name_str in atom_name.ljust(4):
            atom_encode.append(onehot_dict[ord(name_str) - 32])
        mol_encode.append(atom_encode)
    onehot_tensor = torch.Tensor(mol_encode)
    return onehot_tensor


def convert_atom_name_id(onehot_tensor: torch.Tensor):
    """
        Converts integer of atom_name to unique atom_id names.
        Each character is encoded as chr(c + 32)
    """
    # Create reverse mapping from one-hot index to characters
    index_to_char = {index: chr(key + 32) for key, index in enumerate(range(64))}

    # Extract atom names from the tensor
    atom_names = []
    for atom_encode in onehot_tensor:
        atom_name = ''
        for char_onehot in atom_encode:
            index = char_onehot.argmax().item()  # Find the index of the maximum value
            atom_name += index_to_char[index]
        atom_names.append(atom_name.strip())  # Remove padding spaces

    return atom_names


def calc_centre_of_mass(coords, atom_mass):
    mass_coords = coords * atom_mass[:, None]
    return torch.sum(mass_coords, dim=0) / torch.sum(atom_mass)


def get_atom_features(data_object, ccd_atom14):
    atom_mask = data_object['atom_mask']
    atom_positions = torch.zeros(int(atom_mask.sum()), 3, dtype=torch.float32)
    atom_com = atom_positions.clone()

    ref_positions = torch.zeros(int(atom_mask.sum()), 3, dtype=torch.float32)
    ref_com = ref_positions.clone()

    token2atom_map = torch.zeros(int(atom_mask.sum()), dtype=torch.int64)
    atom_elements = torch.zeros(int(atom_mask.sum()), dtype=torch.int64)

    index_start, token = 0, 0
    atom_type = []
    for residues, residue_positions in zip(atom_mask, data_object['atom_positions']):
        length = int(residues.sum())
        index_end = index_start + length

        token2atom_map[index_start:index_end] += token

        atom_index = torch.where(residues)[0]
        atom_positions[index_start:index_end] = residue_positions[atom_index]
        atom_elements[index_start:index_end] = torch.tensor([element_atomic_number[i] for i in atom_index],
                                                            dtype=torch.int64)
        ref_positions[index_start:index_end] = ccd_atom14[int(data_object['aatype'][token])]['coord'][atom_index]

        crt_com = residue_positions[1]
        crt_ref_com = ccd_atom14[int(data_object['aatype'][token])]['com']

        atom_com[index_start:index_end] = crt_com * torch.ones_like(atom_positions[index_start:index_end])
        ref_com[index_start:index_end] = crt_ref_com * torch.ones_like(atom_positions[index_start:index_end])

        atom_type.extend([atom_types[i] for i in atom_index])

        index_start = index_end
        token += 1

    atom_space_uid = token2atom_map
    atom_name_char = convert_atom_id_name(atom_type)

    onehot_dict = {}
    for index, key in enumerate(range(32)):
        onehot = [0] * 32
        onehot[index] = 1
        onehot_dict[key] = onehot

    onehot_encoded_data = [onehot_dict[int(item)] for item in atom_elements]
    atom_elements = torch.Tensor(onehot_encoded_data)

    # Add random rotation
    rot_matrix = uniform_random_rotation(data_object['residue_index'].shape[0])
    rot_matrix_expand = [rot_matrix[i] for i in token2atom_map]
    rot_matrix = torch.stack(rot_matrix_expand, dim=0)

    ref_positions = rot_vec_mul(
        r=rot_matrix,
        t=ref_positions - ref_com,
    )

    atom_distances = (atom_positions[:, None, :] - atom_positions[None, :, :]).norm(dim=-1)

    output_batch = {
        'atom_positions': atom_positions,
        'atom_mask': torch.ones_like(atom_positions[:, 0]).squeeze(),
        'atom_com': atom_com,
        'lddt_mask': atom_distances < 15.0,
        'atom_to_token_index': token2atom_map,

        'residue_index': data_object['residue_index'],
        'chain_index': torch.zeros_like(data_object['residue_index'], dtype=torch.int32),
        'token_index': data_object['residue_index'],
        'seq_mask': torch.ones_like(data_object['residue_index'], dtype=torch.float32),
        'aatype': data_object['aatype'],

        'ref_positions': ref_positions,
        'ref_space_uid': atom_space_uid,
        'ref_atom_name_chars': atom_name_char,
        'ref_element': atom_elements,
        'ref_mask': torch.ones_like(atom_space_uid, dtype=torch.float)
    }
    return output_batch


class FeatureTransform:
    def __init__(self,
                 truncate_size: Optional[int] = None,
                 recenter_atoms: bool = True,
                 strip_missing_residues: bool = True,
                 eps: float = 1e-8,
                 ccd_info: str = './data/ccd_info.pt'
                 ):

        if truncate_size is not None:
            assert truncate_size > 0, f"Invalid truncate_length: {truncate_size}"
        self.truncate_length = truncate_size

        self.strip_missing_residues = strip_missing_residues
        self.recenter_and_scale = recenter_atoms
        self.eps = eps

        with open(ccd_info, 'rb') as f:
            self.ccd_info = pickle.load(f)

    def __call__(self, chain_feats):
        chain_feats = self.patch_feats(chain_feats)

        if self.strip_missing_residues:
            chain_feats = self.strip_ends(chain_feats)

        if self.truncate_length is not None:
            chain_feats = self.random_truncate(chain_feats, max_len=self.truncate_length)

        # Recenter and scale atom positions
        if self.recenter_and_scale:
            chain_feats = self.recenter_and_scale_coords(chain_feats, coordinate_scale=1.,
                                                         eps=self.eps)
        # Map to torch Tensor
        chain_feats = self.map_to_tensors(chain_feats)

        # transform to all-atom features
        chain_feats = get_atom_features(chain_feats, self.ccd_info)
        return chain_feats

    @staticmethod
    def patch_feats(chain_feats):
        seq_mask = chain_feats['atom_mask'][:, CA_IDX]  # a little hack here
        # residue_idx = np.arange(seq_mask.shape[0], dtype=np.int64)
        residue_idx = chain_feats['residue_index'] - np.min(
            chain_feats['residue_index'])  # start from 0, possibly has chain break
        patch_feats = {
            'seq_mask': seq_mask,
            'residue_mask': seq_mask,
            'residue_idx': residue_idx,
            'fixed_mask': np.zeros_like(seq_mask),
            'sc_ca_t': np.zeros(seq_mask.shape + (3,)),
        }
        chain_feats.update(patch_feats)
        return chain_feats

    @staticmethod
    def strip_ends(chain_feats):
        # Strip missing residues on both ends
        modeled_idx = np.where(chain_feats['aatype'] != 20)[0]
        min_idx, max_idx = np.min(modeled_idx), np.max(modeled_idx)

        chain_feats = tree.map_structure(
            lambda x: x[min_idx: (max_idx + 1)], chain_feats)
        return chain_feats

    @staticmethod
    def random_truncate(chain_feats, max_len):
        L = chain_feats['aatype'].shape[0]
        if L > max_len:
            # Randomly truncate
            start = np.random.randint(0, L - max_len + 1)
            end = start + max_len
            chain_feats = tree.map_structure(
                lambda x: x[start: end], chain_feats)
        return chain_feats

    @staticmethod
    def map_to_tensors(chain_feats):
        chain_feats = {k: torch.as_tensor(v) for k, v in chain_feats.items()}
        # Alter dtype
        for k, dtype in DTYPE_MAPPING.items():
            if k in chain_feats:
                chain_feats[k] = chain_feats[k].type(dtype)
        return chain_feats

    @staticmethod
    def recenter_and_scale_coords(chain_feats, coordinate_scale, eps=1e-8):
        # recenter and scale atom positions
        bb_pos = chain_feats['atom_positions'][:, CA_IDX]
        bb_center = np.sum(bb_pos, axis=0) / (np.sum(chain_feats['seq_mask']) + eps)
        centered_pos = chain_feats['atom_positions'] - bb_center[None, None, :]
        scaled_pos = centered_pos * coordinate_scale
        chain_feats['atom_positions'] = scaled_pos * chain_feats['atom_mask'][..., None]
        return chain_feats


class TrainingDataset(torch.utils.data.Dataset):
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
