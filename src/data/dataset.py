import os
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Union, Optional, Sequence, Dict, List

import numpy as np
import pandas as pd
import torch

DATA_MAPPING = {
    'atom_positions': torch.float32,
    'atom_mask': torch.long,
    'atom_to_token_index': torch.long,

    'aatype': torch.int32,
    'moltype': torch.int32,
    'chain_index': torch.int32,
    'residue_index': torch.int32,
    'token_index': torch.int32,

    'ref_positions': torch.float32,
    'ref_mask': torch.long,
    'ref_element': torch.int32,
    'ref_atom_name_chars': torch.int32,
}


class FeatureTransform:
    def __init__(self,
                 truncate_size: int = 384,
                 recenter_atoms: bool = True,
                 eps: float = 1e-6,
    ):
        self.truncate_size = truncate_size
        self.recenter_atoms = recenter_atoms
        self.eps = eps

    def __call__(self, data_object):
        atom_object, token_object = self.patch_feature(data_object)

        if self.truncate_size is not None:
            atom_object, token_object = self.truncate(atom_object, token_object, truncate_size=self.truncate_size)

        if self.recenter_atoms:
            atom_object = self.recenter(atom_object, eps=self.eps)

        data_object = {**atom_object, **token_object}
        data_object = {k: torch.Tensor(v) for k, v in data_object.items()}
        data_object = {k: v.to(dtype=DATA_MAPPING[k]) for k, v in data_object.items()}
        return data_object

    @staticmethod
    def patch_feature(data_object):
        residue_idx = data_object['residue_index'] - np.min(data_object['residue_index'])

        atom_object = {
            'atom_positions': data_object['atom_positions'],
            'atom_mask': data_object['atom_mask'],
            'atom_to_token_index': data_object['atom_to_token_index'],
            'ref_positions': data_object['ref_positions'],
            'ref_mask': data_object['ref_mask'],
            'ref_element': data_object['ref_element'],
            'ref_atom_name_chars': data_object['ref_atom_name_chars'],
        }
        token_object = {
            'aatype': data_object['aatype'],
            'moltype': data_object['moltype'],
            'chain_index': data_object['chain_index'],
            'residue_index': residue_idx,
            'token_index': data_object['token_index'],
        }
        return atom_object, token_object

    @staticmethod
    def truncate(atom_object, token_object, truncate_size=384):
        # Prepare output dictionaries as lists for later concatenation
        cropped_atom_object = {k: [] for k in atom_object}
        cropped_token_object = {k: [] for k in token_object}

        # Precompute chain indices for each unique chain ID
        chain_ids = np.unique(token_object['chain_index'])
        chain_indices = {cid: np.where(token_object['chain_index'] == cid)[0] for cid in chain_ids}

        # Shuffle the chain IDs using numpy's in-place shuffle
        shuffled_chain_ids = chain_ids.copy()
        np.random.shuffle(shuffled_chain_ids)

        n_added = 0
        n_remaining = token_object['token_index'].shape[0]

        for cid in shuffled_chain_ids:
            indices = chain_indices[cid]
            chain_size = indices.shape[0]
            n_remaining -= chain_size

            # Determine crop size limits for the current chain
            crop_size_max = min(chain_size, truncate_size - n_added)
            crop_size_min = min(chain_size, max(0, truncate_size - n_added - n_remaining))
            crop_size = np.random.randint(crop_size_min, crop_size_max + 1)
            if crop_size <= 2:
                continue
            n_added += crop_size

            # Get crop indices for tokens
            crop_start = np.random.randint(0, chain_size - crop_size + 1)
            crop_end = crop_start + crop_size
            token_crop_indices = indices[crop_start:crop_end]

            # Crop token_object for each key using the precomputed indices
            for k, v in token_object.items():
                cropped_token_object[k].append(v[token_crop_indices])

            # Determine atom cropping boundaries from token indices
            crop_atom_start = token_object['token_index'][token_crop_indices[0]]
            crop_atom_end = token_object['token_index'][token_crop_indices[-1] + 1] \
                if (token_crop_indices[-1] + 1) < token_object['token_index'].shape[0] \
                else token_object['token_index'][token_crop_indices[-1]]
            crop_atom_mask = (atom_object['atom_to_token_index'] >= crop_atom_start) & \
                             (atom_object['atom_to_token_index'] < crop_atom_end)
            for k, v in atom_object.items():
                cropped_atom_object[k].append(v[crop_atom_mask])

            # Stop if the desired total crop size is reached
            if n_added >= truncate_size:
                break

        # Concatenate the lists of arrays into single numpy arrays
        cropped_atom_object = {k: np.concatenate(v, axis=0) for k, v in cropped_atom_object.items()}
        cropped_token_object = {k: np.concatenate(v, axis=0) for k, v in cropped_token_object.items()}

        return cropped_atom_object, cropped_token_object

    @staticmethod
    def recenter(atom_object, eps=1e-8):
        atom_center = np.sum(atom_object['atom_positions'], axis=0) / np.sum(atom_object['atom_mask']) + eps  # to be revised to CA centers
        atom_object['atom_positions'] -= atom_center[None, :]
        return atom_object

    @staticmethod
    def get_ref_structure(data_object):
        return data_object

class TrainingDataset(torch.nn.utils.Dataset):
    def __init__(self,
                 path_to_dataset: Union[str, Path],
                 transform: Optional[callable] = None,
                 training: bool = True,
    ):
        super().__init__()

        self.path_to_dataset = os.path.expanduser(path_to_dataset)
        assert self.path_to_dataset.endswith('.csv'), 'Please provide a metadata in csv'

        self._df = pd.read_csv(path_to_dataset)
        self._df.sort_values('modeled_seq_len', ascending=False)
        self._data = self._df['processed_path'].tolist()
        self._data = np.asarray(self._data)

        self.transform = transform

    def __len__(self):
        return len(self._data)

    @lru_cache(maxsize=100)
    def __getitem__(self, idx):
        """
        data_object:
            atom_positions
            atom_mask
            atom_to_token_index
            aatype
            moltype
            chain_index
            residue_index
            token_index
            chain_index
        """

        data_path = self._data[idx]
        accession_code = os.path.splitext(os.path.basename(data_path))[0]

        with open(data_path, 'r') as f:
            data_object = pickle.load(f)

        if self.transform is not None:
            data_object = self.transform(data_object)

        data_object['accession_code'] = accession_code

        return data_object


class BatchTensorConverter:
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(self, target_keys: Optional[List] = None):
        self.target_keys = target_keys

    def __call__(self, raw_batch: Sequence[Dict[str, object]]):
        B = len(raw_batch)
        # Only do for Tensor
        target_keys = self.target_keys \
            if self.target_keys is not None else [k for k, v in raw_batch[0].items() if torch.is_tensor(v)]
        # Non-array, for example string, int
        non_array_keys = [k for k in raw_batch[0] if k not in target_keys]
        collated_batch = dict()
        for k in target_keys:
            collated_batch[k] = self.collate_dense_tensors([d[k] for d in raw_batch], pad_v=0.0)
        for k in non_array_keys:  # return non-array keys as is
            collated_batch[k] = [d[k] for d in raw_batch]
        return collated_batch

    @staticmethod
    def collate_dense_tensors(samples: Sequence, pad_v: float = 0.0):
        """
        Takes a list of tensors with the following dimensions:
            [(d_11,       ...,           d_1K),
             (d_21,       ...,           d_2K),
             ...,
             (d_N1,       ...,           d_NK)]
        and stack + pads them into a single tensor of:
        (N, max_i=1,N { d_i1 }, ..., max_i=1,N {diK})
        """
        if len(samples) == 0:
            return torch.Tensor()
        if len(set(x.dim() for x in samples)) != 1:
            raise RuntimeError(
                f"Samples has varying dimensions: {[x.dim() for x in samples]}"
            )
        (device,) = tuple(set(x.device for x in samples))  # assumes all on same device
        max_shape = [max(lst) for lst in zip(*[x.shape for x in samples])]
        result = torch.empty(
            len(samples), *max_shape, dtype=samples[0].dtype, device=device
        )
        result.fill_(pad_v)
        for i in range(len(samples)):
            result_i = result[i]
            t = samples[i]
            result_i[tuple(slice(0, k) for k in t.shape)] = t
        return result
