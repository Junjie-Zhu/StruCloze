import pickle
import random
from typing import Optional
import tree

import numpy as np
import torch

from src.utils.model_utils import uniform_random_rotation, rot_vec_mul
from src.data.cropping import single_chain_truncate, contiguous_truncate, spatial_truncate

CA_IDX = 1
DTYPE_MAPPING = {
    'atom_positions': torch.float32,
    'atom_mask': torch.long,
    'atom_to_token_index': torch.long,
    'atom_com': torch.float32,

    'aatype': torch.int64,
    'moltype': torch.int32,
    'chain_index': torch.int32,
    'residue_index': torch.int32,
    'token_index': torch.int32,

    'ref_positions': torch.float32,
    'ref_mask': torch.long,
    'ref_element': torch.int32,
    'ref_atom_name_chars': torch.int32,
    'ref_space_uid': torch.long,
    'ref_com': torch.float32,
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
    global_rot_matrix = uniform_random_rotation(1).expand(atom_positions.shape[0], 3, 3)
    rot_matrix = uniform_random_rotation(data_object['residue_index'].shape[0])
    rot_matrix_expand = [rot_matrix[i] for i in token2atom_map]
    rot_matrix = torch.stack(rot_matrix_expand, dim=0)

    atom_positions = rot_vec_mul(
        r=global_rot_matrix,
        t=atom_positions,
    )

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


class BioFeatureTransform:
    def __init__(self,
                 truncate_size: Optional[int] = None,
                 recenter_atoms: bool = True,
                 eps: float = 1e-8,
                 training: bool = True
                 ):

        if truncate_size is not None:
            assert truncate_size > 0, f"Invalid truncate_length: {truncate_size}"
        self.truncate_length = truncate_size

        self.recenter_and_scale = recenter_atoms
        self.eps = eps
        self.training = training

    def __call__(self, data_object):
        atom_object, token_object = self.patch_features(data_object)

        if self.truncate_length is not None:
            atom_object, token_object = self.truncate(atom_object, token_object, truncate_size=self.truncate_length)
        atom_object['atom_mask'] = np.ones(atom_object['atom_positions'].shape[0], dtype=np.int64)
        atom_object['ref_mask'] = atom_object['atom_mask']
        token_object['token_mask'] = np.ones(token_object['aatype'].shape[0], dtype=np.int64)

        # Recenter and scale atom positions
        if self.recenter_and_scale:
            atom_object = self.recenter_and_scale_coords(atom_object, eps=self.eps)
        data_object = {**atom_object, **token_object}

        data_object = self.map_to_tensors(data_object)
        data_object = self.update_ref_features(data_object, training=self.training)
        return data_object

    @staticmethod
    def patch_features(data_object):
        atom_object = {
            'atom_positions': data_object['atom_positions'],
            'atom_to_token_index': data_object['atom_to_token_index'],
            'atom_com': data_object['atom_com'],

            'ref_positions': data_object['ref_positions'],
            'ref_element': data_object['ref_element'],
            'ref_atom_name_chars': data_object['ref_atom_name_chars'],
            'ref_com': data_object['ref_com'],
        }
        token_object = {
            'aatype': data_object['aatype'],
            'moltype': data_object['moltype'],
            'residue_index': data_object['residue_index'],
            'chain_index': data_object['chain_index'],
            'token_index': data_object['token_index'],
        }
        return atom_object, token_object

    @staticmethod
    def map_to_tensors(chain_feats):
        chain_feats = {k: torch.as_tensor(v) for k, v in chain_feats.items()}
        # Alter dtype
        for k, dtype in DTYPE_MAPPING.items():
            if k in chain_feats:
                chain_feats[k] = chain_feats[k].type(dtype)
        return chain_feats

    @staticmethod
    def recenter_and_scale_coords(atom_object, eps=1e-8):
        atom_center = np.sum(atom_object['atom_positions'], axis=0) / (np.sum(atom_object['atom_mask']) + eps)
        atom_object['atom_positions'] -= atom_center[None, :]
        return atom_object

    @staticmethod
    def truncate(atom_object, token_object, truncate_size=384):
        random_state = random.random()
        if random_state < 0.3:
            return single_chain_truncate(atom_object, token_object, truncate_size)
        elif random_state < 0.6:
            return contiguous_truncate(atom_object, token_object, truncate_size)
        else:
            return spatial_truncate(atom_object, token_object, truncate_size)

    @staticmethod
    def update_ref_features(data_object, translation_scale=1.5, training=True):
        data_object['ref_space_uid'] = data_object['atom_to_token_index']
        ref_positions = data_object['ref_positions']

        if training:
            # Add random rotation and translation
            moltype = (data_object['moltype'] > 0).float() + 1.
            translation = (torch.randn((data_object['token_index'].shape[0], 3)) *
                           (moltype[:, None] * translation_scale))  # 2-fold deviation for nucleotides
            translation_expand = [translation[i] for i in data_object['atom_to_token_index']]
            translation = torch.stack(translation_expand, dim=0)

            if random.random() < 0.5:  # a masked learning strategy
                translation_mask = (torch.rand(translation.shape[0]) < 0.05).float()
                translation = translation * translation_mask[..., None]
            ref_positions += translation

        rot_matrix = uniform_random_rotation(data_object['token_index'].shape[0])
        rot_matrix_expand = [rot_matrix[i] for i in data_object['atom_to_token_index']]
        rot_matrix = torch.stack(rot_matrix_expand, dim=0)
        ref_positions = rot_vec_mul(
            r=rot_matrix,
            t=ref_positions - data_object['ref_com'],
        )
        data_object['ref_structure'] = ref_positions + data_object['atom_com']

        if training:
            # add masks for loss
            atom_distances = (data_object['atom_positions'][:, None, :] - data_object['atom_positions'][None, :, :]).norm(dim=-1)
            # extended_moltype = torch.cat(
            #     [moltype[int(i)] for i in data_object['atom_to_token_index']], dim=0
            # )
            # distance_threshold = (extended_moltype[:, None] + extended_moltype[None, :]) > 2.0  # molecule type specific threshold
            data_object['lddt_mask'] = atom_distances < 15.0
            # * (1 + distance_threshold.float()))
            data_object['bond_mask'] = atom_distances < 3.0
        return data_object



