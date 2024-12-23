import os
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Union, Optional
import tree

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.utils import residue_constants, data_transforms, r3_utils

CA_IDX = residue_constants.atom_order['CA']
DTYPE_MAPPING = {
    'aatype': torch.long,
    'atom_positions': torch.double,
    'atom_mask': torch.double,
}


def get_encoded_features(data_object: dict):
    calpha_pos = data_object['atom_positions'][:, CA_IDX]

    # calpha_pos, (b, l, 3)
    # calpha_mask, (b, l)
    prev_calpha_pos = F.pad(calpha_pos[:, :-1], [0, 0, 1, 0])
    prev2_calpha_pos = F.pad(calpha_pos[:, :-2], [0, 0, 2, 0])

    next_calpha_pos = F.pad(calpha_pos[:, 1:], [0, 0, 0, 1])
    next2_calpha_pos = F.pad(calpha_pos[:, 2:], [0, 0, 0, 2])

    # (b, l, 3x3), (b, l, 3)
    left_gt_frames = r3_utils.rigids_from_3_points(
        point_on_neg_x_axis=prev_calpha_pos,
        origin=calpha_pos,
        point_on_xy_plane=prev2_calpha_pos)

    left_forth_atom_rel_pos = r3_utils.rigids_mul_vecs(
        r3_utils.invert_rigids(left_gt_frames),
        next_calpha_pos)

    right_gt_frames = r3_utils.rigids_from_3_points(
        point_on_neg_x_axis=next_calpha_pos,
        origin=calpha_pos,
        point_on_xy_plane=next2_calpha_pos)

    right_forth_atom_rel_pos = r3_utils.rigids_mul_vecs(
        r3_utils.invert_rigids(right_gt_frames),
        prev_calpha_pos)

    ret = {
        'left_gt_calpha3_frame_positions': left_forth_atom_rel_pos,
        'right_gt_calpha3_frame_positions': right_forth_atom_rel_pos,
    }
    data_object.update(ret)
    return data_object


class ProteinTransform:
    def __init__(self,
                 unit: Optional[str] = 'angstrom',
                 truncate_length: Optional[int] = None,
                 strip_missing_residues: bool = True,
                 recenter_and_scale: bool = True,
                 eps: float = 1e-8,
                 ):
        if unit == 'angstrom':
            self.coordinate_scale = 1.0
        elif unit in ('nm', 'nanometer'):
            self.coordiante_scale = 0.1
        else:
            raise ValueError(f"Invalid unit: {unit}")

        if truncate_length is not None:
            assert truncate_length > 0, f"Invalid truncate_length: {truncate_length}"
        self.truncate_length = truncate_length

        self.strip_missing_residues = strip_missing_residues
        self.recenter_and_scale = recenter_and_scale
        self.eps = eps

    def __call__(self, chain_feats, ccd_atom14):
        chain_feats = self.patch_feats(chain_feats)

        if self.strip_missing_residues:
            chain_feats = self.strip_ends(chain_feats)
        if self.truncate_length is not None:
            chain_feats = self.random_truncate(chain_feats, max_len=self.truncate_length)
        # Recenter and scale atom positions
        if self.recenter_and_scale:
            chain_feats = self.recenter_and_scale_coords(chain_feats, coordinate_scale=self.coordinate_scale, eps=self.eps)
        # Map to torch Tensor
        chain_feats = self.map_to_tensors(chain_feats)

        # Add extra features from AF2
        chain_feats = self.protein_data_transform(chain_feats)

        return chain_feats

    @staticmethod
    def patch_feats(chain_feats):
        seq_mask = chain_feats['atom_mask'][:, CA_IDX]  # a little hack here
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

    @staticmethod
    def protein_data_transform(chain_feats):
        chain_feats.update(
            {
                "all_atom_positions": chain_feats["atom_positions"],
                "all_atom_mask": chain_feats["atom_mask"],
            }
        )
        chain_feats = data_transforms.atom37_to_frames(chain_feats)
        chain_feats = data_transforms.atom37_to_torsion_angles("")(chain_feats)
        chain_feats = data_transforms.get_backbone_frames(chain_feats)
        chain_feats = data_transforms.get_chi_angles(chain_feats)
        chain_feats = data_transforms.make_pseudo_beta("")(chain_feats)
        chain_feats = data_transforms.make_atom14_masks(chain_feats)
        chain_feats = data_transforms.make_atom14_positions(chain_feats)

        # Add convenient key
        chain_feats.pop("all_atom_positions")
        chain_feats.pop("all_atom_mask")
        return chain_feats


class ProteinDataset(torch.nn.utils.Dataset):
    def __init__(self,
                 path_to_dataset: Union[str, Path],
                 transform: Optional[callable] = None,
                 training: bool = True,
    ):
        super.__init__()

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
            seq
            mask
            pair_mask
            chain_id
            dist_one_hot
            left_gt_calpha3_frame_positions
            right_gt_calpha3_frame_positions

            # recycle_features
            prev_seq
            prev_pair
        """

        data_path = self._data[idx]
        accession_code = os.path.splitext(os.path.basename(data_path))[0]

        with open(data_path, 'r') as f:
            data_object = pickle.load(f)

        if self.transform is not None:
            data_object = self.transform(data_object)

        data_object['accession_code'] = accession_code

        return data_object
