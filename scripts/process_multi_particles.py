import biotite.structure as struc
import numpy as np
import torch

import biom_constants as bc


calvados_rna_topology = {
    "A": ["P", "N1"],
    "G": ["P", "N1"],
    "C": ["P", "N9"],
    "U": ["P", "N9"],
    "DA": ["P", "N1"],
    "DG": ["P", "N1"],
    "DC": ["P", "N9"],
    "DT": ["P", "N9"],
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
}

def residue_to_calv_rna(residue):
    assert residue.res_name[0] in calvados_rna_topology.keys(), f"Invalid residue name: {residue.res_name[0]}"
    # get positions of two beads
    atom_coords = []
    for atm in calvados_rna_topology[residue.res_name[0]]:
        atom_coords.append(residue[residue.atom_name == atm].coord)
    atom_coords = np.stack(atom_coords[::-1], axis=0)  # make sure the first atom is on base
    return atom_coords


def residue_to_isrna1(residue):
    assert residue.res_name[0] in isrna1_base_topology.keys(), f"Invalid residue name: {residue.res_name[0]}"
    # get positions of two beads
    atom_coords = []
    for atm in ["C4'", "P"]:
        atom_coords.append(residue[residue.atom_name == atm].coord)
    for atm_group in isrna1_base_topology[residue.res_name[0]].values():
        atm_group_com = np.zeros((3,))
        atm_group_weight = 0
        for atm in atm_group:
            atm_weight = bc.WEIGHT_MAPPING[atm[0]]
            atm_group_com += residue[residue.atom_name == atm].coord * atm_weight
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
        atom_coords.append(residue[residue.atom_name == atm].coord)
    for atm_group in isrna2_base_topology[residue.res_name[0]].values():
        atm_group_com = np.zeros((3,))
        atm_group_weight = 0
        for atm in atm_group:
            atm_weight = bc.WEIGHT_MAPPING[atm[0]]
            atm_group_com += residue[residue.atom_name == atm].coord * atm_weight
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
        atom_coords = np.zeros((3,))
        for atm, w in zip(atm_group, wit):
            atom_coords += residue[residue.atom_name == atm].coord * w
        atom_coords /= np.sum(wit)
        particle_coords.append(atom_coords)
    return np.stack(particle_coords, axis=0)


import numpy as np


def align_local_frames_numpy(pred_pose, ref_pose, true_pose, mask=None, allowing_reflection=False):
    """
    Align local fragment `pred_pose` (N_res, N_atoms, 3) using transformation that aligns
    `ref_pose` (N_res, N_ref, 3) to `true_pose` (N_res, N_ref, 3).

    Parameters:
        pred_pose: (N_res, N_atoms, 3) - atoms to transform (e.g., CCD atoms)
        ref_pose: (N_res, N_ref, 3) - local CG beads (in CCD frame)
        true_pose: (N_res, N_ref, 3) - global CG beads (target frame)
        mask: (N_res, N_ref) or None
        allowing_reflection: whether to allow improper rotations (reflection)

    Returns:
        pred_aligned: (N_res, N_atoms, 3) - transformed coordinates
        R: (N_res, 3, 3) - rotation matrices
        T: (N_res, 1, 3) - translation vectors
    """
    N_res = pred_pose.shape[0]

    if mask is None:
        mask = np.ones(ref_pose.shape[:2], dtype=np.float32)  # (N_res, N_ref)
    weight = mask[..., None]  # (N_res, N_ref, 1)

    # 1. Compute centroids
    ref_centroid = np.sum(ref_pose * weight, axis=1, keepdims=True) / np.sum(weight, axis=1, keepdims=True)
    true_centroid = np.sum(true_pose * weight, axis=1, keepdims=True) / np.sum(weight, axis=1, keepdims=True)

    # 2. Center coordinates
    ref_centered = ref_pose - ref_centroid
    true_centered = true_pose - true_centroid

    # 3. Compute covariance matrix H
    H = np.einsum('nik,nij->nkj', ref_centered * weight, true_centered)  # (N_res, 3, 3)

    # 4. SVD and compute rotation matrix
    R = np.empty((N_res, 3, 3))
    for i in range(N_res):
        U, S, Vt = np.linalg.svd(H[i])
        R_temp = Vt.T @ U.T
        if not allowing_reflection and np.linalg.det(R_temp) < 0:
            Vt[2, :] *= -1
            R_temp = Vt.T @ U.T
        R[i] = R_temp

    # 5. Apply rotation and translation
    pred_centered = pred_pose - ref_centroid  # (N_res, N_atoms, 3)
    pred_rotated = np.einsum('nij,nkj->nki', R, pred_centered)  # (N_res, N_atoms, 3)
    pred_aligned = pred_rotated + true_centroid

    # 6. Compute translation vector
    T = true_centroid - np.einsum('nij,nkj->nki', R, ref_centroid)

    return pred_aligned, R, T

