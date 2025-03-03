import gzip
import os
import pickle
import string
import multiprocessing as mp
from multiprocessing.managers import Value

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import numpy as np
import pandas as pd
from tqdm import tqdm

import biom_constants as bc

# Global map from chain characters to integers. e.g, A -> 0, B -> 1, etc.
ALPHANUMERIC = string.ascii_letters + string.digits + ' '
CHAIN_TO_INT = {
    chain_char: i for i, chain_char in enumerate(ALPHANUMERIC)
}
COMPONENT_FILE = './protenix_dataset/components.v20240608.cif'


def chain_str_to_int(chain_str: str):
    chain_int = 0
    if len(chain_str) == 1:
        return CHAIN_TO_INT[chain_str]
    for i, chain_char in enumerate(chain_str):
        chain_int += CHAIN_TO_INT[chain_char] + (i * len(ALPHANUMERIC))
    return chain_int


def convert_atom_id_name(atom_names: str):
    """
        Converts unique atom_id names to integer of atom_name. need to be padded to length 4.
        Each character is encoded as ord(c) - 32
    """
    onehot_dict = {}
    for index, key in enumerate(range(64)):
        onehot = [0] * 64
        onehot[index] = 1
        onehot_dict[key] = onehot

    # [4, 64]
    atom_encode = []
    for name_str in atom_names.ljust(4):
        atom_encode.append(onehot_dict[ord(name_str) - 32])
    onehot_tensor = np.array(atom_encode)
    return onehot_tensor


def write_to_pkl(
        protein_dict: dict,
        output_dir: str,
):
    with open(output_dir, "wb") as f:
        pickle.dump(protein_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


def process_single_pkl(file_path):
    with gzip.open(file_path, 'rb') as f:
        data = pickle.load(f)

    # atom-level features
    atom_positions = []
    atom_to_token_index = []
    atom_mask = []

    # token-level features
    aatype = []
    moltype = []
    residue_index = []
    token_index = []
    chain_index = []

    # reference features
    ref_positions = []
    ref_mask = []
    ref_element = []
    ref_atom_name_chars = []

    atom_array = data['atom_array']  # a biotite atom array
    token_id = 0
    chain_id = 0
    chain_id_str = None
    for residues in struc.residue_iter(atom_array):
        if residues[0].hetero:
            continue  # do not process ligands

        restype_idx = bc.STD_RESIDUES.get(residues[0].res_name, 31)
        if restype_idx <= 20:
            mol_type = 0  # is protein
        elif restype_idx in [21, 22, 23, 24, 29]:
            mol_type = 1  # is RNA
        elif restype_idx in [25, 26, 27, 28, 30]:
            mol_type = 2  # is DNA
        else:
            raise ValueError(f"Unknown residue type: {residues[0].res_name}")

        pos = np.zeros((len(bc.RES_ATOMS_DICT[residues[0].res_name]), 3))
        mask = np.zeros((len(bc.RES_ATOMS_DICT[residues[0].res_name]),))
        a2t_id = np.full_like(mask, fill_value=token_id)
        for atom in residues:
            if atom.atom_name not in bc.RES_ATOMS_DICT[residues[0].res_name]:
                continue
            pos[bc.RES_ATOMS_DICT[residues[0].res_name][atom.atom_name]] = atom.coord
            mask[bc.RES_ATOMS_DICT[residues[0].res_name][atom.atom_name]] = 1.

        if np.mean(mask) < 0.3:
            raise ValueError(f"Too many missing atoms in {file_path}")

        atom_positions.append(pos)
        atom_mask.append(mask)
        atom_to_token_index.append(a2t_id)

        aatype.append(restype_idx)
        moltype.append(mol_type)
        residue_index.append(residues[0].res_id)
        token_index.append(token_id)
        chain_index.append(chain_id)

        # ref_pos, _mask, _element, _atom_name_chars = get_ccd_features(residues[0].res_name)
        # ref_positions.append(ref_pos)
        # ref_mask.append(_mask)
        # ref_element.append(_element)
        # ref_atom_name_chars.append(_atom_name_chars)

        # update token_id and chain_id
        token_id += 1
        if chain_id_str is not None and chain_id_str != residues[0].chain_id:
            chain_id += 1
        chain_id_str = residues[0].chain_id

    # concat data
    biom_dict = {
        "atom_positions": np.concatenate(atom_positions, axis=0),  # N_atom, 3
        "atom_mask": np.concatenate(atom_mask, axis=0),  # N_atom
        "atom_to_token_index": np.concatenate(atom_to_token_index, axis=0),  # N_atom

        "aatype": np.array(aatype),  # N_token
        "moltype": np.array(moltype),  # N_token
        "residue_index": np.array(residue_index),  # N_token
        "token_index": np.array(token_index),  # N_token
        "chain_index": np.array(chain_index),  # N_token

        "ref_positions": np.concatenate(ref_positions, axis=0),  # N_atom, 3
        "ref_mask": np.concatenate(ref_mask, axis=0),  # N_atom
        "ref_element": np.concatenate(ref_element, axis=0),  # N_atom, 32
        "ref_atom_name_chars": np.concatenate(ref_atom_name_chars, axis=0),  # N_atom, 4, 64
    }

    # get basic entry information
    biom_accession_code = os.path.basename(file_path).split('.')[0]
    biom_chain_num = np.max(biom_dict['chain_index']) + 1
    biom_token_num = np.max(biom_dict['token_index']) + 1
    biom_type = np.unique(biom_dict['moltype'])
    biom_metadata = {
        "accession_code": biom_accession_code,
        "chain_num": biom_chain_num,
        "token_num": biom_token_num,
        "type": biom_type,
    }
    return biom_dict, biom_metadata


ccd_cif = pdbx.CIFFile.read(COMPONENT_FILE)
def get_ccd_features(ccd_code):
    comp = pdbx.get_component(ccd_cif, data_block=ccd_code, use_ideal_coord=True)
    comp = comp[~np.isin(comp.element, ["H", "D"])]

    onehot_dict = {}
    for index, key in enumerate(range(32)):
        onehot = [0] * 32
        onehot[index] = 1
        onehot_dict[key] = onehot

    ref_pos = np.zeros((len(bc.RES_ATOMS_DICT[ccd_code]), 3))
    ref_mask = np.zeros((len(bc.RES_ATOMS_DICT[ccd_code]),))
    element = np.zeros((len(bc.RES_ATOMS_DICT[ccd_code]), 32))
    atom_name_chars = np.zeros((len(bc.RES_ATOMS_DICT[ccd_code]), 4, 64))
    for atom in comp:
        if atom.atom_name not in bc.RES_ATOMS_DICT[ccd_code]:
            continue
        ref_pos[bc.RES_ATOMS_DICT[ccd_code][atom.atom_name]] = atom.coord
        ref_mask[bc.RES_ATOMS_DICT[ccd_code][atom.atom_name]] = 1.
        element[bc.RES_ATOMS_DICT[ccd_code][atom.atom_name]] = onehot_dict[bc.ELEMENT_MAPPING[atom.element]]
        atom_name_chars[bc.RES_ATOMS_DICT[ccd_code][atom.atom_name]] = convert_atom_id_name(atom.atom_name)
    return ref_pos, ref_mask, element, atom_name_chars


def single_iteration(
        path_list: dict,
):
    input_path, output_path = path_list
    metadata = None
    fail_list = []
    try:
        protein_dict, metadata = process_single_pkl(input_path)
        write_to_pkl(protein_dict, output_path)
        metadata['processed_path'] = output_path
    except:
        fail_list.append(input_path)
    return metadata, fail_list


input_dir = './protenix_dataset/mmcif_bioassembly'
output_dir = './protenix_dataset/pkl'
path_list = [[os.path.join(input_dir, i), os.path.join(output_dir, i)] for i in os.listdir(input_dir)]
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# process all files
process_num = os.cpu_count()
fails = []
all_metadata = {
    "accession_code": [],
    "chain_num": [],
    "token_num": [],
    "type": [],
    "processed_path": [],
}
with mp.Pool(process_num) as pool:
    for (metadata, fail_list) in tqdm(pool.imap_unordered(single_iteration, path_list), total=len(path_list)):
        if len(fail_list) > 0:
            fails.extend(fail_list)
        if metadata is not None:
            for key, value in metadata.items():
                all_metadata[key].append(value)
# record metadata as csv
df_metadata = pd.DataFrame(all_metadata)
df_metadata.to_csv('./protenix_dataset/metadata.csv', index=False)

# record failed files
if len(fails) > 0:
    with open('./protenix_dataset/fail_list.txt', 'w') as f:
        for fail in fails:
            f.write(fail + '\n')
    print(f"Failed files: {len(fails)}")

# test read pkl
with open(df_metadata['processed_path'][0], 'rb') as f:
    test = pickle.load(f)
    for k, v in test.items():
        print(k, v.shape)

    # save the dict as txt
    with open('./protenix_dataset/test.txt', 'w') as f:
        f.write(str(test))

