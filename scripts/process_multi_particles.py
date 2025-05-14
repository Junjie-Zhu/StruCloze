import biotite.structure as struc
import numpy as np
import torch

import biom_constants as bc


calvados_rna_topology = {
    "A": ["P", "N9"],
    "G": ["P", "N9"],
    "C": ["P", "N1"],
    "U": ["P", "N1"],
    "DA": ["P", "N9"],
    "DG": ["P", "N9"],
    "DC": ["P", "N1"],
    "DT": ["P", "N1"],
}

isrna1_base_topology = {
    "A": {
        "Rc": ["N9", "C8", "N7", "C5", "C4", "N3"],
        "Ac": ["C6", "N6"],
        "An": ["C2", "N1"],
    },
    "G": {
        "Rc": ["N9", "C8", "N7", "C5", "C4", "N3"],
        "Go": ["C6", "O6"],
        "Gn": ["C2", "N2", "N1"],
    },
    "C": {
        "Yc": ["N1", "C5", "C6"],
        "Cn": ["C2", "O2", "N3", "C4", "N4"],
    },
    "U": {
        "Yc": ["N1", "C5", "C6"],
        "Un": ["C2", "O2", "N3", "C4", "O4"],
    },
    "DA": {
        "Rc": ["N9", "C8", "N7", "C5", "C4", "N3"],
        "Ac": ["C6", "N6"],
        "An": ["C2", "N1"],
    },
    "DG": {
        "Rc": ["N9", "C8", "N7", "C5", "C4", "N3"],
        "Go": ["C6", "O6"],
        "Gn": ["C2", "N2", "N1"],
    },
    "DC": {
        "Yc": ["N1", "C5", "C6"],
        "Cn": ["C2", "O2", "N3", "C4", "N4"],
    },
    "DT": {
        "Yc": ["N1", "C5", "C6"],
        "Un": ["C2", "O2", "N3", "C4", "O4"],
    },
}

isrna2_base_topology = {
    "A": {
        "R1": ["N9", "C8", "N7", "C5", "C4", "N3"],
        "A1": ["C6", "N6"],
        "A2": ["C2", "N1"],
    },
    "G": {
        "R1": ["N9", "C8", "N7", "C5", "C4", "N3"],
        "G1": ["C6", "O6", "N1"],
        "G2": ["C2", "N2"],
    },
    "C": {
        "Y1": ["N1", "C5", "C6"],
        "Y2": ["C2", "O2"],
        "C1": ["N3", "C4", "N4"],
    },
    "U": {
        "Y1": ["N1", "C5", "C6"],
        "Y2": ["C2", "O2"],
        "U1": ["N3", "C4", "O4"],
    },
    "DA": {
        "R1": ["N9", "C8", "N7", "C5", "C4", "N3"],
        "A1": ["C6", "N6"],
        "A2": ["C2", "N1"],
    },
    "DG": {
        "R1": ["N9", "C8", "N7", "C5", "C4", "N3"],
        "G1": ["C6", "O6", "N1"],
        "G2": ["C2", "N2"],
    },
    "DC": {
        "Y1": ["N1", "C5", "C6"],
        "Y2": ["C2", "O2"],
        "C1": ["N3", "C4", "N4"],
    },
    "DT": {
        "Y1": ["N1", "C5", "C6"],
        "Y2": ["C2", "O2"],
        "U1": ["N3", "C4", "O4"],
    },
}

def residue_to_calv_rna(residue):
    assert residue.res_name[0] in calvados_rna_topology.keys(), f"Invalid residue name: {residue.res_name[0]}"
    # get positions of two beads
    atom_coords = []
    for atm in calvados_rna_topology[residue.res_name[0]]:
        atom_coords.append(residue[residue.atom_name == atm].coord.flatten())
    atom_coords = np.stack(atom_coords[::-1], axis=0)  # make sure the first atom is on base
    return atom_coords


def residue_to_isrna1(residue):
    assert residue.res_name[0] in isrna1_base_topology.keys(), f"Invalid residue name: {residue.res_name[0]}"
    # get positions of two beads
    atom_coords = []
    for atm in ["C4'", "P"]:
        atom_coords.append(residue[residue.atom_name == atm].coord.flatten())
    for atm_group in isrna1_base_topology[residue.res_name[0]].values():
        atm_group_com = np.zeros(3)
        atm_group_weight = 0
        for atm in atm_group:
            atm_weight = bc.WEIGHT_MAPPING[atm[0]]
            atm_group_com += residue[residue.atom_name == atm].coord.flatten() * atm_weight
            atm_group_weight += atm_weight
        atm_group_com = atm_group_com / atm_group_weight
        atom_coords.append(atm_group_com)
    atom_coords = np.stack(atom_coords, axis=0)
    return atom_coords


def residue_to_isrna2(residue):
    assert residue.res_name[0] in isrna2_base_topology.keys(), f"Invalid residue name: {residue.res_name[0]}"
    # get positions of two beads
    atom_coords = []
    for atm in ["C4'", "P"]:
        atom_coords.append(residue[residue.atom_name == atm].coord.flatten())
    for atm_group in isrna2_base_topology[residue.res_name[0]].values():
        atm_group_com = np.zeros(3)
        atm_group_weight = 0
        for atm in atm_group:
            atm_weight = bc.WEIGHT_MAPPING[atm[0]]
            atm_group_com += residue[residue.atom_name == atm].coord.flatten() * atm_weight
            atm_group_weight += atm_weight
        atm_group_com = atm_group_com / atm_group_weight
        atom_coords.append(atm_group_com)
    atom_coords = np.stack(atom_coords, axis=0)
    return atom_coords


def read_martini_topology(name):
    top_s = {}
    weight_s = {}
    with open(name) as fp:
        for line in fp:
            if line.startswith("RESI"):
                resName = line.strip().split()[1]
                top_s[resName] = []
                weight_s[resName] = []
            elif line.startswith("BEAD"):
                atmName_s = line.strip().split()[2:]
                weights = []
                for i_atm, atmName in enumerate(atmName_s):
                    if "_" in atmName:
                        atmName, weight = atmName.split("_")
                        atmWeight = bc.WEIGHT_MAPPING[atmName[0]]
                        weight = weight.split("/")
                        weight = float(weight[0]) / float(weight[1]) * atmWeight
                    else:
                        weight = bc.WEIGHT_MAPPING[atmName[0]]
                    weights.append(weight)
                top_s[resName].append(atmName_s)
                weight_s[resName].append(weights)
    return top_s, weight_s


top_s, weight_s = read_martini_topology('martini3.top')
def residue_to_martini(residue):
    crt_top_s = top_s[residue.res_name[0]]
    crt_weight_s = weight_s[residue.res_name[0]]

    particle_coords = []
    for atm_group, wit in zip(crt_top_s, crt_weight_s):
        # get COM for each group
        atom_coords = np.zeros(3)
        for atm, w in zip(atm_group, wit):
            atom_coords += residue[residue.atom_name == atm].coord.flatten() * w
        atom_coords /= np.sum(wit)
        particle_coords.append(atom_coords)
    return np.stack(particle_coords, axis=0)


def align_single_residue(pred_pose, ref_pose, true_pose, mask=None, allowing_reflection=False):
    """
    Aligns one local fragment (single residue) using Kabsch algorithm.

    Parameters:
        pred_pose: (N_atoms, 3) - fragment to be transformed (e.g., CCD atoms)
        ref_pose: (N_ref, 3) - local reference points (e.g., CG beads in CCD)
        true_pose: (N_ref, 3) - target reference points (e.g., CG beads in global coords)
        mask: (N_ref,) - optional binary mask to indicate valid reference points
        allowing_reflection: whether to allow improper rotations (reflection)

    Returns:
        aligned_pose: (N_atoms, 3) - transformed pred_pose
        R: (3, 3) - rotation matrix
        T: (1, 3) - translation vector
    """
    if mask is None:
        mask = np.ones(ref_pose.shape[0], dtype=np.float32)

    weight = mask[:, None]  # (N_ref, 1)

    # Center both ref and true coordinates
    ref_centroid = np.sum(ref_pose * weight, axis=0, keepdims=True) / np.sum(weight)
    true_centroid = np.sum(true_pose * weight, axis=0, keepdims=True) / np.sum(weight)

    ref_centered = ref_pose - ref_centroid
    true_centered = true_pose - true_centroid

    # Compute weighted covariance matrix
    H = (ref_centered * weight).T @ true_centered  # (3, 3)

    # SVD and rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if not allowing_reflection and np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    # Rotate and translate pred_pose
    pred_centered = pred_pose - ref_centroid  # (N_atoms, 3)
    aligned_pose = pred_centered @ R.T + true_centroid  # (N_atoms, 3)

    T = true_centroid - ref_centroid @ R.T  # (1, 3)

    return aligned_pose, R, T


