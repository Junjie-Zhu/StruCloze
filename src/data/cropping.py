import numpy as np


def single_chain_truncate(atom_object, token_object, truncate_size=384):
    if token_object['token_index'].shape[0] <= truncate_size:
        return atom_object, token_object

    # Randomly select a chain ID
    chain_ids = np.unique(token_object['chain_index'])
    chain_id = np.random.choice(chain_ids)
    chain_indices = np.where(token_object['chain_index'] == chain_id)[0]

    # Crop if still too long
    if len(chain_indices) > truncate_size:
        crop_start = np.random.randint(0, len(chain_indices) - truncate_size + 1)
        crop_end = crop_start + truncate_size
        chain_indices = chain_indices[crop_start:crop_end]

    # Crop token_object for each key using the precomputed indices
    cropped_token_object = {k: v[chain_indices] for k, v in token_object.items()}

    # Determine atom cropping boundaries from token indices
    crop_atom_start = token_object['token_index'][chain_indices[0]]
    crop_atom_end = token_object['token_index'][chain_indices[-1]]
    crop_atom_mask = (atom_object['atom_to_token_index'] >= crop_atom_start) & \
                     (atom_object['atom_to_token_index'] <= crop_atom_end)

    # Crop atom_object for each key using the precomputed indices
    cropped_atom_object = {k: v[crop_atom_mask] for k, v in atom_object.items()}

    cropped_token_object['token_index'] -= np.min(cropped_token_object['token_index'])
    cropped_atom_object['atom_to_token_index'] -= np.min(cropped_atom_object['atom_to_token_index'])
    return cropped_atom_object, cropped_token_object


def contiguous_truncate(atom_object, token_object, truncate_size=384):
    if token_object['token_index'].shape[0] <= truncate_size:
        return atom_object, token_object

    # Prepare output dictionaries as lists for later concatenation
    cropped_atom_object = {k: [] for k in atom_object}
    cropped_token_object = {k: [] for k in token_object}

    # Precompute chain indices for each unique chain ID
    chain_ids = np.unique(token_object['chain_index'])
    chain_indices = {cid: np.where(token_object['chain_index'] == cid)[0] for cid in chain_ids}

    # Shuffle the chain IDs using numpy's in-place shuffle
    shuffled_chain_ids = chain_ids.copy()
    np.random.shuffle(shuffled_chain_ids)

    n_added = 0
    n_remaining = token_object['token_index'].shape[0]

    for cid in shuffled_chain_ids:
        indices = chain_indices[cid]
        chain_size = indices.shape[0]
        n_remaining -= chain_size

        # Determine crop size limits for the current chain
        crop_size_max = min(chain_size, truncate_size - n_added)
        crop_size_min = min(chain_size, max(0, truncate_size - n_added - n_remaining))
        crop_size = np.random.randint(crop_size_min, crop_size_max + 1)
        if crop_size <= 2:
            continue
        n_added += crop_size

        # Get crop indices for tokens
        crop_start = np.random.randint(0, chain_size - crop_size + 1)
        crop_end = crop_start + crop_size
        token_crop_indices = indices[crop_start:crop_end]

        # Crop token_object for each key using the precomputed indices
        for k, v in token_object.items():
            cropped_token_object[k].append(v[token_crop_indices])

        # Determine atom cropping boundaries from token indices
        crop_atom_start = token_object['token_index'][token_crop_indices[0]]
        crop_atom_end = token_object['token_index'][token_crop_indices[-1] + 1] \
            if (token_crop_indices[-1] + 1) < token_object['token_index'].shape[0] \
            else token_object['token_index'][token_crop_indices[-1]] + 1
        crop_atom_mask = (atom_object['atom_to_token_index'] >= crop_atom_start) & \
                         (atom_object['atom_to_token_index'] < crop_atom_end)
        for k, v in atom_object.items():
            cropped_atom_object[k].append(v[crop_atom_mask])

        # Stop if the desired total crop size is reached
        if n_added >= truncate_size:
            break

    # Concatenate the lists of arrays into single numpy arrays
    cropped_atom_object = {k: np.concatenate(v, axis=0) for k, v in cropped_atom_object.items()}
    cropped_token_object = {k: np.concatenate(v, axis=0) for k, v in cropped_token_object.items()}

    # Reindex token id and atom_to_token id
    token_id_mapping = {tid: idx for idx, tid in enumerate(cropped_token_object['token_index'])}
    cropped_token_object['token_index'] = np.array(
        [token_id_mapping[tid] for tid in cropped_token_object['token_index']])
    cropped_atom_object['atom_to_token_index'] = np.array(
        [token_id_mapping[tid] for tid in cropped_atom_object['atom_to_token_index']])

    return cropped_atom_object, cropped_token_object


def spatial_truncate(atom_object, token_object, truncate_size=384, truncate_dist=10):
    if token_object['token_index'].shape[0] <= truncate_size:
        return atom_object, token_object

