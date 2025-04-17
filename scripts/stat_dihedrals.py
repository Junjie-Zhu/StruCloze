import os
import pickle
import gzip

import numpy as np
import pandas as pd
from tqdm import tqdm


def main():
    target_dir = './dihedrals_pt'
    output_dir = './stats_pt'

    output_dict = {
        'clash_hist': None,
        'clash_edges': None,
        'bond_hist': None,
        'bond_xedges': None,
        'bond_yedges': None,
        'phi_hist': None,
        'psi_hist': None,
        'chi1_hist': None,
        'chi2_hist': None,
        'chi3_hist': None,
        'chi4_hist': None,
        'dihedral_xedges': None,
        'dihedral_yedges': None,
    }

    df = pd.read_csv(os.path.join(target_dir, 'metadata.csv'))
    output_dict['clash_hist'], output_dict['clash_edges'] = get_clash_histogram(
        df['clashes'], bins=72
    )

    data_init = load_data(os.path.join(target_dir, f"{df['accession_code'][0]}.dih.pkl.gz"))
    output_dict['bond_hist'], output_dict['bond_xedges'], output_dict['bond_yedges'] = get_bond_histogram(
        data_init['bond'], min_x=0, max_x=3.5, min_y=0, max_y=3.5, bins=72
    )
    output_dict['phi_hist'], output_dict['dihedral_xedges'], output_dict['dihedral_yedges'] = get_dihedral_histogram(
        data_init['phi'], min_x=-np.pi, max_x=np.pi, min_y=-np.pi, max_y=np.pi, bins=72
    )
    output_dict['psi_hist'], _, _ = get_dihedral_histogram(
        data_init['psi'], min_x=-np.pi, max_x=np.pi, min_y=-np.pi, max_y=np.pi, bins=72
    )
    chi_angles = process_chi_angles(data_init['chi'])
    for k, v in chi_angles.items():
        output_dict[f'{k}_hist'], _, _ = get_dihedral_histogram(
            v, min_x=-np.pi, max_x=np.pi, min_y=-np.pi, max_y=np.pi, bins=72
        )

    for system in tqdm(df['accession_code'][1:]):
        data = load_data(os.path.join(target_dir, f'{system}.dih.pkl.gz'))
        output_dict['bond_hist'], _, _ = get_bond_histogram(
            data['bond'], min_x=0, max_x=3.5, min_y=0, max_y=3.5, bins=72, accumulate=output_dict['bond_hist']
        )
        output_dict['phi_hist'], _, _ = get_dihedral_histogram(
            data['phi'], min_x=-np.pi, max_x=np.pi, min_y=-np.pi, max_y=np.pi, bins=72, accumulate=output_dict['phi_hist']
        )
        output_dict['psi_hist'], _, _ = get_dihedral_histogram(
            data['psi'], min_x=-np.pi, max_x=np.pi, min_y=-np.pi, max_y=np.pi, bins=72, accumulate=output_dict['psi_hist']
        )
        chi_angles = process_chi_angles(data['chi'])
        for k, v in chi_angles.items():
            output_dict[f'{k}_hist'], _, _ = get_dihedral_histogram(
                v, min_x=-np.pi, max_x=np.pi, min_y=-np.pi, max_y=np.pi, bins=72, accumulate=output_dict[f'{k}_hist']
            )

    with open(os.path.join(output_dir, 'stats.pkl'), 'wb') as f:
        pickle.dump(output_dict, f)


def load_data(file_path):
    with gzip.open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def get_bond_histogram(bonds, min_x=0, max_x=3.5, min_y=0, max_y=3.5, bins=72, accumulate=None):
    hist, xedges, yedges = np.histogram2d(
        bonds[:, 0], bonds[:, 1], bins=bins, range=[[min_x, max_x], [min_y, max_y]]
    )
    if accumulate is not None:
        return hist.T + accumulate, xedges, yedges
    return hist.T, xedges, yedges


def get_dihedral_histogram(dihedrals, min_x=-np.pi, max_x=np.pi, min_y=-np.pi, max_y=np.pi, bins=72, accumulate=None):
    hist, xedges, yedges = np.histogram2d(
        dihedrals[:, 0], dihedrals[:, 1], bins=bins, range=[[min_x, max_x], [min_y, max_y]]
    )
    if accumulate is not None:
        return hist.T + accumulate, xedges, yedges
    return hist.T, xedges, yedges


def get_clash_histogram(clashes, bins=72):
    hist, edges = np.histogram(clashes, bins=bins)
    return hist, edges


def process_chi_angles(chi):
    chi_angles = {
        'chi1': [],
        'chi2': [],
        'chi3': [],
        'chi4': []
    }
    for residue in chi:
        if residue is None or len(residue) != 2:
            continue
        pred, gt = residue
        if not isinstance(gt, list) or not isinstance(pred, list):
            continue
        max_len = min(len(pred), len(gt))
        for i in range(max_len):
            gt_i = gt[i]
            pred_i = pred[i]
            if gt_i is not None and pred_i is not None:
                chi_angles[f'chi{i + 1}'].append([pred_i, gt_i])
    chi_angles = {k: np.array(v).squeeze() for k, v in chi_angles.items()}
    return chi_angles


if __name__ == "__main__":
    main()
