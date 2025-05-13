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


def residue_to_calv_rna(residue):
    assert residue.res_name[0] in calvados_rna_topology.keys(), f"Invalid residue name: {residue.res_name[0]}"
    # get positions of two beads
    atom_coords = []
    for atm in calvados_rna_topology[residue.res_name[0]]:
        atom_coords.append(residue[residue.atom_name == atm].coord)
    atom_coords = np.stack(atom_coords[::-1], axis=0)  # make sure the first atom is on base
    return atom_coords


def residue_to_isrna1(residue):
    assert residue.res_name[0] in calvados_rna_topology.keys(), f"Invalid residue name: {residue.res_name[0]}"
    # get positions of two beads
    atom_coords = []
    for atm in ["C4'", "P"]:
        atom_coords.append(residue[residue.atom_name == atm].coord)
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


def kabsch_align(ref_CG: torch.Tensor, CG: torch.Tensor):
    """
    Aligns ref_CG to CG using the Kabsch algorithm.
    Args:
        ref_CG: (N_res, N_pts, 3) - CG beads in local frame
        CG:     (N_res, N_pts, 3) - CG beads in global frame
    Returns:
        R: (N_res, 3, 3) - rotation matrices
        t: (N_res, 1, 3) - translation vectors to align local coords
    """
    # 1. Center the CG and ref_CG
    ref_center = ref_CG.mean(dim=1, keepdim=True)  # (N_res, 1, 3)
    CG_center = CG.mean(dim=1, keepdim=True)       # (N_res, 1, 3)

    ref_centered = ref_CG - ref_center             # (N_res, N_pts, 3)
    CG_centered = CG - CG_center                   # (N_res, N_pts, 3)

    # 2. Compute covariance matrix
    H = torch.matmul(ref_centered.transpose(1, 2), CG_centered)  # (N_res, 3, 3)

    # 3. SVD decomposition
    U, S, Vh = torch.linalg.svd(H)  # U, Vh: (N_res, 3, 3)

    # 4. Compute determinant sign correction
    det = torch.det(torch.matmul(Vh.transpose(1, 2), U))  # (N_res,)
    det = det.view(-1, 1, 1)
    I = torch.eye(3, device=ref_CG.device).unsqueeze(0).repeat(ref_CG.size(0), 1, 1)
    I[:, 2, 2] = det.squeeze()

    # 5. Compute rotation
    R = torch.matmul(Vh.transpose(1, 2), torch.matmul(I, U.transpose(1, 2)))  # (N_res, 3, 3)

    # 6. Translation vector: align ref_CG to CG
    t = CG_center - torch.matmul(ref_center, R)  # (N_res, 1, 3)

    return R, t

