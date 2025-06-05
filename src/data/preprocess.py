# get CG representations from all-atom structures or from CG structures
import string
from typing import Optional

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import numpy as np

import src.common.residue_constants as rc
import src.utils.cg_utils as cg

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
    ref_ca = []
    ref_com = []

    # transformed ref_positions
    calv_positions = []
    isrna1_positions = []
    isrna2_positions = []

    token_id = 0
    chain_id = 0
    chain_id_str = None
    for residues in struc.residue_iter(structure):
        if residues[0].hetero:
            continue  # do not process ligands

        comp = pdbx.get_component(ccd_cif, data_block=residues[0].res_name, use_ideal_coord=True)
        comp = comp[~np.isin(comp.element, ["H", "D"])]

        restype_idx = rc.STD_RESIDUES.get(residues[0].res_name, 31)
        if restype_idx <= 20:
            mol_type = 0  # is protein
        elif restype_idx in [21, 22, 23, 24, 29]:
            mol_type = 1  # is RNA
        elif restype_idx in [25, 26, 27, 28, 30]:
            mol_type = 2  # is DNA
        else:
            raise ValueError(f"Unknown residue type: {residues[0].res_name}")

        pos = np.zeros((len(rc.RES_ATOMS_DICT[residues[0].res_name]), 3))
        mask = np.zeros((len(rc.RES_ATOMS_DICT[residues[0].res_name]),))
        a2t_id = np.full_like(mask, fill_value=token_id)
        ca = np.zeros((3,))
        for atom in residues:
            if atom.atom_name not in rc.RES_ATOMS_DICT[residues[0].res_name]:
                continue
            if atom.atom_name in ['OXT', 'OP3']:
                continue
            pos[rc.RES_ATOMS_DICT[residues[0].res_name][atom.atom_name]] = atom.coord

            if atom.atom_name in ["CA", "C4'"]:  # for CA models we consider P in nucleotides as CG repr.
                ca = atom.coord

        ref_pos = np.zeros((len(rc.RES_ATOMS_DICT[residues[0].res_name]), 3))
        ref_mask = np.zeros((len(rc.RES_ATOMS_DICT[residues[0].res_name]),))
        element = np.zeros((len(rc.RES_ATOMS_DICT[residues[0].res_name]), 32))
        weight = np.zeros((len(rc.RES_ATOMS_DICT[residues[0].res_name]),))
        atom_name_chars = np.zeros((len(rc.RES_ATOMS_DICT[residues[0].res_name]), 4, 64))
        for atom in comp:
            if atom.atom_name not in rc.RES_ATOMS_DICT[residues[0].res_name]:
                continue
            if atom.atom_name in ['OXT', 'OP3']:
                continue
            ref_pos[rc.RES_ATOMS_DICT[residues[0].res_name][atom.atom_name]] = atom.coord
            ref_mask[rc.RES_ATOMS_DICT[residues[0].res_name][atom.atom_name]] = 1.
            element[rc.RES_ATOMS_DICT[residues[0].res_name][atom.atom_name]] = onehot_dict[rc.ELEMENT_MAPPING[atom.element]]
            weight[rc.RES_ATOMS_DICT[residues[0].res_name][atom.atom_name]] = rc.WEIGHT_MAPPING[atom.element]
            atom_name_chars[rc.RES_ATOMS_DICT[residues[0].res_name][atom.atom_name]] = convert_atom_id_name(atom.atom_name)

            if atom.atom_name in ["CA", "C4'"]:  # for CA models we consider P in nucleotides as CG repr.
                ref_ca_ = atom.coord

        ref_mask = ref_mask.astype(bool)

        if mol_type == 1 or mol_type == 2:
            try:
                calv_rna_cg = cg.residue_to_calv_rna(residues)
            except:
                continue

            ref_calv_rna_cg = cg.residue_to_calv_rna(comp)
            ref_calv_rna_pos, _, _ = cg.align_single_residue(ref_pos[ref_mask], ref_calv_rna_cg, calv_rna_cg)
            # isrna1_cg = pm.residue_to_isrna1(residues)
            # ref_isrna1_cg = pm.residue_to_isrna1(comp)
            # ref_isrna1_pos, _, _  = pm.align_single_residue(ref_pos[ref_mask], ref_isrna1_cg, isrna1_cg)
            # isrna2_cg = pm.residue_to_isrna2(residues)
            # ref_isrna2_cg = pm.residue_to_isrna2(comp)
            # ref_isrna2_pos, _, _  = pm.align_single_residue(ref_pos[ref_mask], ref_isrna2_cg, isrna2_cg)

            calv_positions.append(ref_calv_rna_pos)
            # isrna1_positions.append(ref_isrna1_pos)
            # isrna2_positions.append(ref_isrna2_pos)
        elif mol_type == 0:
            martini_cg = cg.residue_to_martini(residues)
            ref_martini_cg = cg.residue_to_martini(comp)
            ref_martini_pos = cg.align_single_residue(ref_pos[ref_mask], ref_martini_cg, martini_cg)

            calv_positions.append(ref_martini_pos)
            isrna1_positions.append(ref_martini_pos)
            isrna2_positions.append(ref_martini_pos)
        else:
            raise ValueError(f"Unknown moltype: {mol_type}")

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
        ref_ca.append(np.array([ref_ca_] * np.sum(ref_mask)))
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
        "ref_ca": np.concatenate(ref_ca),
        "ref_com": np.concatenate(ref_com),

        "calv_positions": np.concatenate(calv_positions),
        # "isrna1_positions": np.concatenate(isrna1_positions),
        # "isrna2_positions": np.concatenate(isrna2_positions),
    }


