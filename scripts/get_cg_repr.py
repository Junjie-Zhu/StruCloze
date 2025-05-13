# get CG representations from all-atom structures or from CG structures
import string
from functools import partial
from typing import Optional
import argparse
import os
import gzip
import pickle
import multiprocessing as mp

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import numpy as np
import pandas as pd
from tqdm import tqdm

import biom_constants as bc
import process_multi_particles as pm

# Global map from chain characters to integers. e.g, A -> 0, B -> 1, etc.
ALPHANUMERIC = string.ascii_letters + string.digits + ' '
CHAIN_TO_INT = {
    chain_char: i for i, chain_char in enumerate(ALPHANUMERIC)
}
COMPONENT_FILE = './protenix_dataset/components.cif'
ccd_cif = pdbx.CIFFile.read(COMPONENT_FILE)

onehot_dict = {}
for index, key in enumerate(range(32)):
    onehot = [0] * 32
    onehot[index] = 1
    onehot_dict[key] = onehot


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


def calc_center_of_mass(
    atom_positions: np.ndarray,
    atom_masses: np.ndarray,
    mask: Optional[np.ndarray] = None,
):
    """
    Calculate the center of mass of the atoms in a residue.
    """
    # Calculate the center of mass
    if mask is not None:
        atom_positions = atom_positions[mask]
        atom_masses = atom_masses[mask]
    return np.sum(atom_positions * atom_masses[:, None], axis=0) / np.sum(atom_masses)


def get_structure(file_path):
    if file_path.endswith(".pdb"):
        structure = struc.io.load_structure(file_path)
        try:
            struc_depth = structure.stack_depth()
            structure = structure[0]
        except AttributeError:
            struc_depth = 1
    elif file_path.endswith(".cif"):
        cif_file = pdbx.CIFFile.read(file_path)
        structure = pdbx.get_structure(cif_file)[0]
    else:
        raise ValueError(f"Invalid file format {file_path}")
    return structure


def get_cg_repr(
    structure,
):
    # CA, COM, scCOM, MARTINI for protein; P, IsRNA, MARTINI for RNA

    # atom-level features
    atom_positions = []
    atom_to_token_index = []

    # token-level features
    aatype = []
    moltype = []
    residue_index = []
    token_index = []
    chain_index = []

    # reference features
    ref_positions = []
    ref_element = []
    ref_atom_name_chars = []

    # cg representation
    atom_ca = []
    atom_com = []
    ref_com = []

    token_id = 0
    chain_id = 0
    chain_id_str = None
    for residues in struc.residue_iter(structure):
        if residues[0].hetero:
            continue  # do not process ligands

        comp = pdbx.get_component(ccd_cif, data_block=residues[0].res_name, use_ideal_coord=True)
        comp = comp[~np.isin(comp.element, ["H", "D"])]

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
        ca = np.zeros((3,))
        for atom in residues:
            if atom.atom_name not in bc.RES_ATOMS_DICT[residues[0].res_name]:
                continue
            pos[bc.RES_ATOMS_DICT[residues[0].res_name][atom.atom_name]] = atom.coord

            if atom.atom_name in ["CA", "C4'"]:  # for CA models we consider P in nucleotides as CG repr.
                ca = atom.coord

        ref_pos = np.zeros((len(bc.RES_ATOMS_DICT[residues[0].res_name]), 3))
        ref_mask = np.zeros((len(bc.RES_ATOMS_DICT[residues[0].res_name]),))
        element = np.zeros((len(bc.RES_ATOMS_DICT[residues[0].res_name]), 32))
        weight = np.zeros((len(bc.RES_ATOMS_DICT[residues[0].res_name]),))
        atom_name_chars = np.zeros((len(bc.RES_ATOMS_DICT[residues[0].res_name]), 4, 64))
        for atom in comp:
            if atom.atom_name not in bc.RES_ATOMS_DICT[residues[0].res_name]:
                continue
            ref_pos[bc.RES_ATOMS_DICT[residues[0].res_name][atom.atom_name]] = atom.coord
            ref_mask[bc.RES_ATOMS_DICT[residues[0].res_name][atom.atom_name]] = 1.
            element[bc.RES_ATOMS_DICT[residues[0].res_name][atom.atom_name]] = onehot_dict[bc.ELEMENT_MAPPING[atom.element]]
            weight[bc.RES_ATOMS_DICT[residues[0].res_name][atom.atom_name]] = bc.WEIGHT_MAPPING[atom.element]
            atom_name_chars[bc.RES_ATOMS_DICT[residues[0].res_name][atom.atom_name]] = convert_atom_id_name(atom.atom_name)

        ref_mask = ref_mask.astype(bool)
        atom_positions.append(pos[ref_mask])
        atom_to_token_index.append(a2t_id[ref_mask])

        aatype.append(restype_idx)
        moltype.append(mol_type)
        residue_index.append(residues[0].res_id)
        token_index.append(token_id)
        chain_index.append(chain_id)

        ref_positions.append(ref_pos[ref_mask])
        ref_element.append(element[ref_mask])
        ref_atom_name_chars.append(atom_name_chars[ref_mask])

        atom_ca.append(np.array([ca] * np.sum(ref_mask)))
        atom_com.append(
            np.tile(calc_center_of_mass(pos, weight, ref_mask)[np.newaxis, :], (np.sum(ref_mask), 1))
        )
        ref_com.append(
            np.tile(calc_center_of_mass(ref_pos, weight, ref_mask)[np.newaxis, :], (np.sum(ref_mask), 1))
        )

        # update token_id and chain_id
        token_id += 1
        if chain_id_str is not None and chain_id_str != residues[0].chain_id:
            chain_id += 1
        chain_id_str = residues[0].chain_id

    return {
        "atom_positions": np.concatenate(atom_positions),
        "atom_to_token_index": np.concatenate(atom_to_token_index),

        "aatype": np.array(aatype),
        "moltype": np.array(moltype),
        "residue_index": np.array(residue_index),
        "token_index": np.array(token_index),
        "chain_index": np.array(chain_index),

        "ref_positions": np.concatenate(ref_positions),
        "ref_element": np.concatenate(ref_element),
        "ref_atom_name_chars": np.concatenate(ref_atom_name_chars),

        "atom_ca": np.concatenate(atom_ca),
        "atom_com": np.concatenate(atom_com),
        "ref_com": np.concatenate(ref_com),
    }


def process_fn(
    input_path,
    output_dir,
):
    structure = get_structure(input_path)

    try:
        cg_repr = get_cg_repr(structure)
    except ValueError as e:
        return {}, f"{os.path.basename(input_path).replace('.cif', '')}\t{e}"

    output_path = os.path.join(output_dir, f"{os.path.basename(input_path).replace('.cif', '')}.pkl.gz")
    with gzip.open(output_path, "wb") as f:
        pickle.dump(cg_repr, f, protocol=pickle.HIGHEST_PROTOCOL)

    metadata = {
        "accession_code": os.path.basename(input_path).replace('.cif', ''),
        "token_num": len(cg_repr["token_index"]),
        "moltype": np.unique(cg_repr["moltype"]),
    }

    return metadata, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="Input file path")
    parser.add_argument("--output_dir", type=str, help="Output file path")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # get all cif files in input_dir
    target_files = [os.path.join(args.input_dir, i) for i in os.listdir(args.input_dir) if i.endswith(".cif")]
    process_fn_ = partial(
        process_fn,
        output_dir=args.output_dir,
    )

    cpu_num = os.cpu_count()
    if cpu_num > 1:
        with mp.Pool(cpu_num) as pool:
            results = list(tqdm(pool.imap_unordered(process_fn_, target_files), total=len(target_files)))
    else:
        results = []
        for target_file in tqdm(target_files):
            results.append(process_fn_(target_file))

    metadata = {
        "accession_code": [],
        "token_num": [],
        "moltype": [],
    }
    error = []
    for result in results:
        if result[1] is not None:
            error.append(result[1])
        else:
            metadata["accession_code"].append(result[0]["accession_code"])
            metadata["token_num"].append(result[0]["token_num"])
            metadata["moltype"].append(result[0]["moltype"])

    metadata = pd.DataFrame(metadata)
    metadata.to_csv(os.path.join(args.output_dir, "metadata.csv"), index=False)
    with open(os.path.join(args.output_dir, "error.log"), "w") as f:
        f.write("\n".join(error))
