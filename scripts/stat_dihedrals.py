import os
import pickle
import gzip

import numpy as np
import pandas as pd
from tqdm import tqdm


def main():
    target_dir = './dihedrals_pt'
    output_dir = './stats_pt'
    reference = './metadata.test.csv'

    output_dict = {
        'bond_hist': None,
        'bond_xedges': None,
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
    system_info = pd.read_csv(reference)
    protein_system = system_info[system_info['moltype'] == '[0]']

    df_others = df[~df['accession_code'].isin(protein_system['accession_code'].tolist())]
    df = df[df['accession_code'].isin(protein_system['accession_code'].tolist())]

    # output_dict = process_protein_entries(
    #     df['accession_code'][0], target_dir, output_dict, accumulate=False
    # )
    # for system in tqdm(df['accession_code'][1:], desc='processing protein systems'):
    #     output_dict = process_protein_entries(system, target_dir, output_dict, accumulate=True)

    output_dict = process_na_entries(
        df_others['accession_code'].tolist()[0], target_dir, output_dict, accumulate=False
    )
    for system in tqdm(df_others['accession_code'].tolist()[1:], desc='processing NA systems'):
        output_dict = process_na_entries(system, target_dir, output_dict, accumulate=True)

    with open(os.path.join(output_dir, 'stats_na.pkl'), 'wb') as f:
        pickle.dump(output_dict, f)


def load_data(file_path):
    with gzip.open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def get_bond_histogram(bonds, min_x=0., max_x=3.5, min_y=0., max_y=3.5, bins=72, accumulate=None):
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


def process_protein_entries(accession_code, target_dir, output_dict, accumulate=False):
    data = load_data(os.path.join(target_dir, f'{accession_code}.dih.pkl.gz'))
    if not accumulate:
        chi1_num, chi1_acc = 0, 0
        chi2_num, chi2_acc = 0, 0
        output_dict['bond_hist'], output_dict['bond_xedges'], output_dict['bond_yedges'] = get_bond_histogram(
            data['bond'], min_x=1.2, max_x=1.6, min_y=1.2, max_y=1.6, bins=72
        )
        output_dict['phi_hist'], output_dict['dihedral_xedges'], output_dict['dihedral_yedges'] = get_dihedral_histogram(
            data['phi'], min_x=-np.pi, max_x=np.pi, min_y=-np.pi, max_y=np.pi, bins=72
        )
        output_dict['psi_hist'], _, _ = get_dihedral_histogram(
            data['psi'], min_x=-np.pi, max_x=np.pi, min_y=-np.pi, max_y=np.pi, bins=72
        )
        ram = np.stack([data['phi'][:, 0], data['psi'][:, 0]], axis=1)
        output_dict['ram_hist'], _, _ = get_dihedral_histogram(
            ram, min_x=-np.pi, max_x=np.pi, min_y=-np.pi, max_y=np.pi, bins=72
        )
        ram_ref = np.stack([data['phi'][:, 1], data['psi'][:, 1]], axis=1)
        output_dict['ram_ref_hist'], _, _ = get_dihedral_histogram(
            ram_ref, min_x=-np.pi, max_x=np.pi, min_y=-np.pi, max_y=np.pi, bins=72
        )
        chi_angles = process_chi_angles(data['chi'])
        for k, v in chi_angles.items():
            output_dict[f'{k}_hist'], _, _ = get_dihedral_histogram(
                v, min_x=-np.pi, max_x=np.pi, min_y=-np.pi, max_y=np.pi, bins=72
            )
            if k == 'chi1':
                chi1_num += len(v)
                chi1_err = np.abs(v[:, 0] - v[:, 1])
                chi1_acc += np.sum(chi1_err < np.pi / 6)
            elif k == 'chi2':
                chi2_num += len(v)
                chi2_err = np.abs(v[:, 0] - v[:, 1])
                chi2_acc += np.sum(chi2_err < np.pi / 6)
        output_dict['chi1_acc'] = (chi1_acc, chi1_num)
        output_dict['chi2_acc'] = (chi2_acc, chi2_num)
    else:
        output_dict['bond_hist'], _, _ = get_bond_histogram(
            data['bond'], min_x=1.2, max_x=1.6, min_y=1.2, max_y=1.6, bins=72, accumulate=output_dict['bond_hist']
        )
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
                    v, min_x=-np.pi, max_x=np.pi, min_y=-np.pi, max_y=np.pi, bins=72,
                    accumulate=output_dict[f'{k}_hist']
                )
            except:
                continue
            if k == 'chi1':
                chi1_acc = np.sum(np.abs(v[:, 0] - v[:, 1]) < np.pi / 6)
                output_dict['chi1_acc'] = (output_dict['chi1_acc'][0] + chi1_acc, output_dict['chi1_acc'][1] + len(v))
            elif k == 'chi2':
                chi2_acc = np.sum(np.abs(v[:, 0] - v[:, 1]) < np.pi / 6)
                output_dict['chi2_acc'] = (output_dict['chi2_acc'][0] + chi2_acc, output_dict['chi2_acc'][1] + len(v))
    return output_dict


def process_na_entries(accession_code, target_dir, output_dict, accumulate=False):
    data = load_data(os.path.join(target_dir, f'{accession_code}.dih.pkl.gz'))
    if not accumulate:
        rna_rotation = data['pseudo_rotation'][data['na_moltype'] == 0]
        dna_rotation = data['pseudo_rotation'][data['na_moltype'] == 1]
        output_dict['rna_rotation'], _, _ = get_dihedral_histogram(
            rna_rotation, min_x=0, max_x=360, min_y=0, max_y=360, bins=72
        )
        output_dict['dna_rotation'], _, _ = get_dihedral_histogram(
            dna_rotation, min_x=0, max_x=360, min_y=0, max_y=360, bins=72
        )
        output_dict['hbond'] = [i for i in data['hbond']]
    else:
        # output_dict['bond_hist'], _, _ = get_bond_histogram(
        #     data['bond'], min_x=1.2, max_x=1.6, min_y=1.2, max_y=1.6, bins=72, accumulate=output_dict['bond_hist']
        # )
        rna_rotation = data['pseudo_rotation'][data['na_moltype'] == 0]
        dna_rotation = data['pseudo_rotation'][data['na_moltype'] == 1]
        output_dict['rna_rotation'], _, _ = get_dihedral_histogram(
            rna_rotation, min_x=0, max_x=360, min_y=0, max_y=360, bins=72,
            accumulate=output_dict['rna_rotation']
        )
        output_dict['dna_rotation'], _, _ = get_dihedral_histogram(
            dna_rotation, min_x=0, max_x=360, min_y=0, max_y=360, bins=72,
            accumulate=output_dict['dna_rotation']
        )
        output_dict['hbond'] = [i + j for i, j in zip(output_dict['hbond'], data['hbond'])]
    return output_dict


if __name__ == "__main__":
    main()
