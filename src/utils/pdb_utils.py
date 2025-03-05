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
    restype = [rc.IDX_TO_RESIDUE[res] for res in input_feature_dict["aatype"].squeeze()]
    atom_to_token_index = input_feature_dict["atom_to_token_index"].squeeze()

    atom_positions = atom_positions.squeeze()
    chain_index = [INT_TO_CHAIN[input_feature_dict["chain_index"].squeeze()[i]] for i in atom_to_token_index]
    restype_ = [restype[i] for i in atom_to_token_index]
    element_ = []
    for res in restype:
        element_.extend(list(rc.RES_ATOMS_DICT[res].keys()))

    structure = struc.AtomArray(len(atom_positions))
    structure.coord = np.array(atom_positions.cpu())
    structure.chain_id = chain_index
    structure.res_name = restype_
    structure.res_id = np.array(atom_to_token_index.cpu())
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
