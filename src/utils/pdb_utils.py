import os
import string
from typing import Any

import numpy as np
import torch
import biotite.structure as struc
import biotite.structure.io.pdb as pdb

import src.common.residue_constants as rc

ALPHANUMERIC = string.ascii_letters + string.digits + ' '
CHAIN_TO_INT = {
    chain_char: i for i, chain_char in enumerate(ALPHANUMERIC)
}
INT_TO_CHAIN = {
    i: chain_char for i, chain_char in enumerate(ALPHANUMERIC)
}


def to_pdb(
    input_feature_dict: dict[str, Any],
    atom_positions: torch.Tensor,
    output_dir: str,
):
    """save atom_positions as pdb with biotite"""
    restype = [rc.IDX_TO_RESIDUE[res.item()] for res in input_feature_dict["aatype"].squeeze()]
    atom_to_token_index = input_feature_dict["atom_to_token_index"].squeeze()

    atom_positions = atom_positions.squeeze()
    chain_index = [INT_TO_CHAIN[input_feature_dict["chain_index"].squeeze()[i.item()]] for i in atom_to_token_index]
    restype_ = [restype[i.item()] for i in atom_to_token_index]
    element_ = convert_atom_name_id(input_feature_dict["ref_atom_name_char"].squeeze())

    structure = struc.AtomArray(len(atom_positions))
    structure.coord = np.array(atom_positions.cpu())
    structure.chain_id = chain_index
    structure.res_name = restype_
    structure.res_id = np.array(atom_to_token_index.cpu())
    structure.atom_name = element_
    structure.element = element_

    pdb_file = pdb.PDBFile()
    pdb_file.set_structure(structure)

    if os.path.isdir(output_dir):
        output_dir = os.path.join(output_dir, f"{input_feature_dict['accession_code']}.pdb")
        pdb_file.write(output_dir)
    elif output_dir.endswith(".pdb"):
        pdb_file.write(output_dir)
    else:
        raise ValueError("output_dir must be a directory or a .pdb file path")


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