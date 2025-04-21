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
        'phi_hist': None,
        'psi_hist': None,
        'ram_hist': None,
        'ram_ref_hist': None,
        'chi1_hist': None,
        'chi2_hist': None,
        'chi_34_hist': None,
        'chi_34_ref_hist': None,
        'dihedral_xedges': None,
        'dihedral_yedges': None,
        'chi1_acc': None,
        'chi2_acc': None,
    }

    df = pd.read_csv(os.path.join(target_dir, 'metadata.csv'))

    chi1_num, chi1_acc = 0, 0
    chi2_num, chi2_acc = 0, 0
    data_init = load_data(os.path.join(target_dir, f"{df['accession_code'][0]}.dih.pkl.gz"))
    output_dict['phi_hist'], output_dict['dihedral_xedges'], output_dict['dihedral_yedges'] = get_dihedral_histogram(
        data_init['phi'], min_x=-np.pi, max_x=np.pi, min_y=-np.pi, max_y=np.pi, bins=72
    )
    output_dict['psi_hist'], _, _ = get_dihedral_histogram(
        data_init['psi'], min_x=-np.pi, max_x=np.pi, min_y=-np.pi, max_y=np.pi, bins=72
    )
    ram = np.stack([data_init['phi'][:, 0], data_init['psi'][:, 0]], axis=1)
    output_dict['ram_hist'], _, _ = get_dihedral_histogram(
        ram, min_x=-np.pi, max_x=np.pi, min_y=-np.pi, max_y=np.pi, bins=72
    )
    ram_ref = np.stack([data_init['phi'][:, 1], data_init['psi'][:, 1]], axis=1)
    output_dict['ram_ref_hist'], _, _ = get_dihedral_histogram(
        ram_ref, min_x=-np.pi, max_x=np.pi, min_y=-np.pi, max_y=np.pi, bins=72
    )
    chi_angles = process_chi_angles(data_init['chi'])
    for k, v in chi_angles.items():
        output_dict[f'{k}_hist'], _, _ = get_dihedral_histogram(
            v, min_x=-np.pi, max_x=np.pi, min_y=-np.pi, max_y=np.pi, bins=72
        )
        if k == 'chi1':
            chi1_num += len(v)
            chi1_err = np.abs(v[:, 0] - v[:, 1])
            chi1_acc += np.sum(chi1_err < np.pi / 6)
            chi1_acc += np.sum(chi1_err > 11 * np.pi / 6)
        elif k == 'chi2':
            chi2_num += len(v)
            chi2_err = np.abs(v[:, 0] - v[:, 1])
            chi2_acc += np.sum(chi2_err < np.pi / 6)
            chi2_acc += np.sum(chi2_err > 11 * np.pi / 6)

    for system in tqdm(df['accession_code'][1:]):
        data = load_data(os.path.join(target_dir, f'{system}.dih.pkl.gz'))
        output_dict['phi_hist'], _, _ = get_dihedral_histogram(
            data['phi'], min_x=-np.pi, max_x=np.pi, min_y=-np.pi, max_y=np.pi, bins=72, accumulate=output_dict['phi_hist']
        )
        output_dict['psi_hist'], _, _ = get_dihedral_histogram(
            data['psi'], min_x=-np.pi, max_x=np.pi, min_y=-np.pi, max_y=np.pi, bins=72, accumulate=output_dict['psi_hist']
        )
        ram = np.stack([data['phi'][:, 0], data['psi'][:, 0]], axis=1)
        output_dict['ram_hist'], _, _ = get_dihedral_histogram(
            ram, min_x=-np.pi, max_x=np.pi, min_y=-np.pi, max_y=np.pi, bins=72, accumulate=output_dict['ram_hist']
        )
        ram_ref = np.stack([data['phi'][:, 1], data['psi'][:, 1]], axis=1)
        output_dict['ram_ref_hist'], _, _ = get_dihedral_histogram(
            ram_ref, min_x=-np.pi, max_x=np.pi, min_y=-np.pi, max_y=np.pi, bins=72, accumulate=output_dict['ram_ref_hist']
        )
        chi_angles = process_chi_angles(data['chi'])
        for k, v in chi_angles.items():
            try:
                output_dict[f'{k}_hist'], _, _ = get_dihedral_histogram(
                    v, min_x=-np.pi, max_x=np.pi, min_y=-np.pi, max_y=np.pi, bins=72, accumulate=output_dict[f'{k}_hist']
                )
            except:
                continue
            if k == 'chi1':
                chi1_num += len(v)
                chi1_acc += np.sum(np.abs(v[:, 0] - v[:, 1]) < np.pi / 6)
            elif k == 'chi2':
                chi2_num += len(v)
                chi2_acc += np.sum(np.abs(v[:, 0] - v[:, 1]) < np.pi / 6)

    output_dict['chi1_acc'] = (chi1_acc, chi1_num)
    output_dict['chi2_acc'] = (chi2_acc, chi2_num)

    with open(os.path.join(output_dir, 'stats.pkl'), 'wb') as f:
        pickle.dump(output_dict, f)


def load_data(file_path):
    with gzip.open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


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
        'chi4': [],
        'chi_34': [],
        'chi_34_ref': [],
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
        if max_len == 4:
            chi_angles['chi_34'].append([pred[2], pred[3]])
            chi_angles['chi_34_ref'].append([gt[2], gt[3]])
    chi_angles = {k: np.array(v).squeeze() for k, v in chi_angles.items()}
    return chi_angles


if __name__ == "__main__":
    main()
