import gzip
import os
import multiprocessing as mp
import argparse
import pickle
from functools import partial
import warnings

import pandas as pd
from tqdm import tqdm
import torch
import biotite.structure as struc
import biotite.structure.io as strucio
import biotite.structure.io.pdbx as pdbx
import numpy as np
from tqdm import tqdm

import biom_constants as bc

warnings.filterwarnings("ignore", category=UserWarning)

NA_keys = ["A", "G", "C", "U", "N", "DA", "DG", "DC", "DT", "DN"]


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


def load_structure(file_path):
    if file_path.endswith(".pdb"):
        return strucio.load_structure(file_path)
    elif file_path.endswith(".cif"):
        cif_file = pdbx.CIFFile.read(file_path)
        return pdbx.get_structure(cif_file)[0]


def process_structure(structure):
    # Remove HETATM and H
    structure = structure[structure.element != "H"]
    structure = structure[structure.hetero == False]
    return structure


def align_atom_num(structure, ref_structure):
    if len(struc.get_residues(structure)[0]) != len(struc.get_residues(ref_structure)[0]):
        raise ValueError("Number of residues must be the same in both structures.")

    _structure = []
    _ref_structure = []

    for residue_idx, (residue, ref_residue) in enumerate(zip(struc.residue_iter(structure), struc.residue_iter(ref_structure))):
        common_atom_set = set(residue.atom_name) & set(ref_residue.atom_name)

        for atom_name in common_atom_set:
            _structure.append(residue[residue.atom_name == atom_name])
            _ref_structure.append(ref_residue[ref_residue.atom_name == atom_name])

    return struc.concatenate(_structure), struc.concatenate(_ref_structure)


def get_distance_matrix(
    structure,
    sparse = False
):
    coords = structure.coord
    atom_elements = structure.element

    # Calculate pairwise distances
    if not sparse:
        distance = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        distance = np.linalg.norm(distance, axis=2)
    else:
        distance = np.concatenate(
            [np.linalg.norm(coords[i] - coords[i + 1:], axis=1) for i in range(len(coords) - 1)],
        )

    return distance, atom_elements


def get_steric_clashes(
    distance_matrix, atom_elements, threshold=0.4, sparse=False, reference_matrix=None
):
    # Calculate the sum of the van der Waals radii for each pair of atoms
    atom_radii = np.array([bc.van_der_waals_radius[element] for element in atom_elements])
    i, j = np.triu_indices(len(atom_elements), k=1)  # Upper triangular indices
    sum_radii = atom_radii[i] + atom_radii[j]

    if not sparse:
        distance_matrix = np.concatenate(
            [distance_matrix[i, i + 1:] for i in range(len(distance_matrix) - 1)]
        )

    if reference_matrix is not None:
        distance_mask = (reference_matrix < 8).astype(float) * (reference_matrix > sum_radii).astype(float)
        bond_mask = (reference_matrix < sum_radii - threshold)
    else:
        distance_mask = np.ones_like(distance_matrix)
        bond_mask = None
    distance_mask = distance_mask.astype(bool)

    clashes = distance_matrix[distance_mask] < sum_radii[distance_mask] - threshold
    return np.sum(clashes) / np.sum(distance_mask), bond_mask


def get_all_atom_rmsd(
    structure,
    ref_structure,
    align = True
):
    if align:
        structure, _ = struc.superimpose(ref_structure, structure)
    return struc.rmsd(ref_structure, structure)


def get_backbone_dihedrals(structure):
    phi, psi, _ = struc.dihedral_backbone(structure)
    return phi[1:-1], psi[1:-1]


def get_chi_angles(structure):
    chi_angles = []
    for residues in struc.residue_iter(structure):
        if residues[0].res_name not in bc.chi_angles_atoms.keys():
            continue

        angles = []
        for atom_groups in bc.chi_angles_atoms[residues[0].res_name]:
            try:
                atom_coords = [residues[residues.atom_name == atom_groups[i]].coord for i in range(4)]
                angles.append(struc.dihedral(*atom_coords))
            except:
                angles.append(None)

        chi_angles.append(angles)
    return chi_angles


def get_pseudo_rotation(structure):
    pseudo_rotations = []
    puckers = []
    moltype = []
    for residues in struc.residue_iter(structure):
        if residues[0].res_name not in NA_keys:
            continue

        mol = 0 if residues[0].res_name in ['A', 'G', 'C', 'U'] else 1

        angles = []
        for atom_groups in bc.na_pseudo_rotation:
            atom_coords = [residues[residues.atom_name == atom_groups[i]].coord for i in range(4)]
            angles_ = np.degrees(struc.dihedral(*atom_coords))
            angles.append(angles_ if angles_.shape[0] == 1 else None)

        if None in angles:
            continue

        pseudo_angle = float(np.degrees(
            np.arctan2((angles[4] - angles[1]) - (angles[3] - angles[0]), 2 * angles[2])
        ) % 360)

        if 0 <= pseudo_angle <= 60 or 300 <= pseudo_angle <= 360:
            pucker_type = 3
        elif 120 <= pseudo_angle <= 240:
            pucker_type = 2
        else:
            pucker_type = 0

        pseudo_rotations.append(pseudo_angle)
        puckers.append(pucker_type)
        moltype.append(mol)
    return pseudo_rotations, puckers, moltype


def get_hbond(
    distance,
    ref_distance,
    ref_structure,
    threshold=3.5,
    sparse=False
):
    assert distance.shape == ref_distance.shape, "Distance matrices must have the same shape"

    if not sparse:
        distance = np.concatenate(
            [distance[i, i + 1:] for i in range(len(distance) - 1)]
        )
        ref_distance = np.concatenate(
            [ref_distance[i, i + 1:] for i in range(len(ref_distance) - 1)]
        )

    restype = ref_structure.res_name
    atomtype = ref_structure.atom_name

    single_mask = np.zeros(atomtype.shape[0])
    moltype = []
    res_index = 0
    unique_res = ""
    for idx, (res, atom) in enumerate(zip(restype, atomtype)):
        if res not in bc.base_pair_atoms.keys():
            continue

        if res != unique_res:
            res_index += 1
            unique_res = res

        if atom in bc.base_pair_atoms[res]:
            single_mask[idx] = res_index

    i, j = np.triu_indices(len(single_mask), k=1)
    pair_mask = (single_mask[i] != single_mask[j]) & (single_mask[i] != 0) & (single_mask[j] != 0)

    hbond = (distance < threshold) & pair_mask
    ref_hbond = (ref_distance < threshold) & pair_mask

    return np.sum(ref_hbond & hbond), np.sum(ref_hbond & ~hbond), np.sum(~ref_hbond & hbond), np.sum(~ref_hbond & ~hbond)  # TP, FN, FP, TN


def process_fn(
    input_path,
    reference_dir,
    output_dir,
    sparse=False,
):
    moltype = ''
    if len(input_path) == 2:
        input_path, moltype = input_path

    accession_code = os.path.basename(input_path).split(".")[0]

    structure = process_structure(load_structure(input_path))
    reference_path = os.path.join(reference_dir, accession_code + ".pkl.gz")
    with gzip.open(reference_path, "rb") as f:
        ref_structure_data = pickle.load(f)

    struc_mask = (np.array(ref_structure_data["atom_positions"]) == 0.).astype(float)
    struc_mask = (np.sum(struc_mask, axis=1) == 0.).astype(bool)
    structure = structure[struc_mask]

    # construct ref_structure with data in dictionary
    ref_structure = struc.AtomArray(np.sum(struc_mask))
    ref_structure.coord = ref_structure_data["atom_positions"][struc_mask]
    ref_structure.atom_name = structure.atom_name
    ref_structure.element = structure.element
    ref_structure.res_name = structure.res_name
    ref_structure.res_id = structure.res_id

    distance, atom_elements = get_distance_matrix(structure, sparse=sparse)
    ref_distance, _ = get_distance_matrix(ref_structure, sparse=sparse)
    clashes, bond_mask = get_steric_clashes(distance, atom_elements, sparse=sparse, reference_matrix=ref_distance)
    bond = np.stack([distance[bond_mask], ref_distance[bond_mask]], axis=1)

    if '0' in moltype:
        phi, psi = get_backbone_dihedrals(structure)
        ref_phi, ref_psi = get_backbone_dihedrals(ref_structure)
        chi_angles = get_chi_angles(structure)
        ref_chi_angles = get_chi_angles(ref_structure)

        dihedrals = {  # its * 2 (pred and ground truth)
            "bond": bond,
            "phi": np.stack([phi, ref_phi], axis=1),
            "psi": np.stack([psi, ref_psi], axis=1),
            "chi": np.stack([chi_angles, ref_chi_angles], axis=1),
        }
        with gzip.open(os.path.join(output_dir, accession_code + ".dih.pkl.gz"), "wb") as f:
            pickle.dump(dihedrals, f)
        del dihedrals, phi, psi, chi_angles
    if ('1' in moltype) or ('2' in moltype):
        pseudo_rotations, puckers, na_moltype = get_pseudo_rotation(structure)
        ref_pseudo_rotations, ref_puckers, _ = get_pseudo_rotation(ref_structure)
        hbonds = get_hbond(distance, ref_distance, ref_structure, sparse=sparse)

        dihedrals = {
            "bond": bond,
            "pseudo_rotation": np.stack([pseudo_rotations, ref_pseudo_rotations], axis=1),
            "pucker": np.stack([puckers, ref_puckers], axis=1),
            "na_moltype": np.array(na_moltype),
            "hbond": hbonds
        }
        with gzip.open(os.path.join(output_dir, accession_code + ".dih.pkl.gz"), "wb") as f:
            pickle.dump(dihedrals, f)
        del dihedrals, pseudo_rotations, puckers, hbonds

    rmsd = get_all_atom_rmsd(structure, ref_structure)
    return {
        "accession_code": accession_code,
        "clashes": clashes,
        "rmsd": rmsd,
    }


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-i', '--input_dir', type=str, required=True)
    args.add_argument('-r', '--reference_dir', type=str, required=True)
    args.add_argument('-o', '--output_dir', type=str, required=True)
    args.add_argument('-p', '--indices', type=str, default=None)
    args = args.parse_args()

    process_fn_ = partial(
        process_fn,
        reference_dir=args.reference_dir,
        output_dir=args.output_dir,
        sparse=True,
    )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    cpu_num = os.cpu_count()
    print(f'Using {cpu_num} cpus')
    if args.indices is not None:
        df = pd.read_csv(args.indices)
        df = df[df['token_num'] < 2048]
        input_files = [(os.path.join(args.input_dir, f'{f}.pdb'), moltype)
                       for f, moltype in zip(df['accession_code'].tolist(), df['moltype'].tolist())
                       if os.path.exists(os.path.join(args.input_dir, f'{f}.pdb'))]
        input_files = [i for i in input_files if i[1] != '[0]']
    else:
        input_files = [os.path.join(args.input_dir, f)
            for f in os.listdir(args.input_dir)
            if f.endswith(".pdb")]

    if cpu_num > 1:
        with mp.Pool(cpu_num) as p:
            results = list(tqdm(p.imap_unordered(process_fn_, input_files), total=len(input_files), desc='Processing'))
    else:
        results = []
        for f in tqdm(input_files, desc='Processing'):
            results.append(process_fn_(f))

    metadata = {
        "accession_code": [],
        "clashes": [],
        "rmsd": [],
    }
    for r in results:
        metadata["accession_code"].append(r["accession_code"])
        metadata["clashes"].append(r["clashes"])
        metadata["rmsd"].append(r["rmsd"])
    metadata = pd.DataFrame(metadata)
    metadata.to_csv(os.path.join(args.output_dir, "metadata.csv"), index=False)

    print(f'Clash: {metadata["clashes"].mean()}, RMSD: {metadata["rmsd"].mean()}')

