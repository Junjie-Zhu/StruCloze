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
    distance_matrix, atom_elements, threshold=0.4, sparse=False
):
    # Calculate the sum of the van der Waals radii for each pair of atoms
    atom_radii = np.array([bc.van_der_waals_radius[element] for element in atom_elements])
    i, j = np.triu_indices(len(atom_elements), k=1)  # Upper triangular indices
    sum_radii = atom_radii[i] + atom_radii[j]

    if not sparse:
        distance_matrix = np.concatenate(
            [distance_matrix[i, i + 1:] for i in range(len(distance_matrix) - 1)]
        )

    clashes = distance_matrix < sum_radii - threshold
    return np.sum(clashes) / len(distance_matrix)


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
    for residues in struc.residue_iter(structure):
        if residues[0].res_name not in NA_keys:
            continue

        angles = []
        for atom_groups in bc.na_pseudo_rotation:
            try:
                atom_coords = [residues[residues.atom_name == atom_groups[i]].coord for i in range(4)]
                angles.append(np.degrees(struc.dihedral(*atom_coords)))
            except:
                angles.append(None)

        if None in angles:
            continue

        pseudo_angle = np.degrees(
            np.arctan2((angles[4] - angles[1]) - (angles[3] - angles[0]), 2 * angles[2])
        ) % 360

        if 0 <= pseudo_angle <= 60 or 300 <= pseudo_angle <= 360:
            pucker_type = 3
        elif 120 <= pseudo_angle <= 240:
            pucker_type = 2
        else:
            pucker_type = 0

        pseudo_rotations.append(pseudo_angle)
        puckers.append(pucker_type)
    return pseudo_rotations, puckers


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

    return np.sum(ref_hbond & hbond), np.sum(ref_hbond & ~hbond), np.sum(~ref_hbond & hbond)  # TP, TN, FP


def process_fn(
    input_path,
    reference_dir,
    output_dir,
    sparse=False
):
    accession_code = os.path.basename(input_path).split(".")[0]
    accession_code = accession_code.replace("['", "").replace("']", "")

    structure = load_structure(input_path)
    reference_path = os.path.join(reference_dir, accession_code + ".pkl.gz")
    with gzip.open(reference_path, "rb") as f:
        ref_structure_data = pickle.load(f)

    distance, atom_elements = get_distance_matrix(structure, sparse=sparse)
    clashes = get_steric_clashes(distance, atom_elements, sparse=sparse)
    phi, psi = get_backbone_dihedrals(structure)
    chi_angles = get_chi_angles(structure)

    dihedrals = {
        "phi": phi,
        "psi": psi,
        "chi_angles": chi_angles,
    }

    struc_mask = (np.array(ref_structure_data["atom_positions"]) == 0.).astype(float)
    struc_mask = (np.sum(struc_mask, axis=1) == 0.).astype(bool)
    structure = structure[struc_mask]

    # construct ref_structure with data in dictionary
    ref_structure = struc.AtomArray(np.sum(struc_mask))
    ref_structure.coord = ref_structure_data["atom_positions"][struc_mask]
    ref_structure.atom_name = structure.atom_name
    ref_structure.element = structure.element
    ref_structure.res_name = structure.res_name

    try:
        rmsd = get_all_atom_rmsd(structure, ref_structure)
    except:
        rmsd = np.nan

    with gzip.open(os.path.join(output_dir, accession_code + ".dih.pkl.gz"), "wb") as f:
        pickle.dump(dihedrals, f)
    del dihedrals, phi, psi, chi_angles

    return {
        "accession_code": accession_code,
        "clashes": clashes,
        "rmsd": rmsd,
    }


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--input_dir', type=str, required=True)
    args.add_argument('--reference_dir', type=str, required=True)
    args.add_argument('--output_dir', type=str, required=True)
    args = args.parse_args()

    process_fn_ = partial(
        process_fn,
        reference_dir=args.reference_dir,
        output_dir=args.output_dir,
    )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    cpu_num = os.cpu_count()
    input_files = [os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.endswith(".pdb")]

    if cpu_num > 1:
        with mp.Pool(cpu_num) as p:
            results = list(tqdm(p.imap(process_fn_, input_files), total=len(input_files), desc='Processing'))
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
