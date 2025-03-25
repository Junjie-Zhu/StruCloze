import os

import biotite.structure as struc
import biotite.structure.io as strucio
import biotite.structure.io.pdbx as pdbx
import numpy as np

import biom_constants as bc

NA_keys = ["A", "G", "C", "U", "N", "DA", "DG", "DC", "DT", "DN"]


def load_structure(file_path):
    if file_path.endswith(".pdb"):
        return strucio.load_structure(file_path)
    elif file_path.endswith(".cif"):
        cif_file = pdbx.CIFFile.read(file_path)
        return pdbx.get_structure(cif_file)[0]


def get_distance_matrix(structure):
    coords = structure.coord
    atom_elements = structure.element

    # Calculate pairwise distances
    distance = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    distance = np.linalg.norm(distance, axis=2)

    return distance, atom_elements


def get_steric_clashes(
    distance_matrix, atom_elements, threshold=0.4
):
    # Calculate the sum of the van der Waals radii for each pair of atoms
    sum_radii = np.array(
        [
            bc.van_der_waals_radius[atom_elements[i]] + bc.van_der_waals_radius[atom_elements[j]]
            for i in range(len(atom_elements))
            for j in range(i + 1, len(atom_elements))
        ]
    )

    clashes = distance_matrix < sum_radii - threshold
    return clashes


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

        if 0 <= pseudo_angle <= 36 or 324 <= pseudo_angle <= 360:
            pucker_type = 3
        elif 144 <= pseudo_angle <= 216:
            pucker_type = 2
        else:
            pucker_type = 0

        pseudo_rotations.append(pseudo_angle)
        puckers.append(pucker_type)
    return pseudo_rotations, puckers

