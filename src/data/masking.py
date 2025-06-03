import torch


def continuous_mask(data_object, mask_ratio=0.5):
    n_tokens = data_object['token_index'].shape[0]
    n_mask = int(n_tokens * mask_ratio)
    n_span = torch.randint(1, n_mask // 3, ()).item()

    proportions = torch.distributions.Dirichlet(torch.ones(n_span)).sample()
    span_lengths = (proportions * n_mask).long()
    span_lengths = span_lengths[span_lengths > 3]

    mask = torch.ones(n_tokens, dtype=torch.float)
    used = torch.zeros(n_tokens, dtype=torch.bool)

    start = 0
    for span_len in span_lengths:
        start += int(torch.poisson(torch.tensor(0.5 * n_tokens / len(span_lengths))).item())
        if not used[start:start + span_len].any():
            mask[start:start + span_len] = 0.0
            used[start:start + span_len] = True
        start += span_len.item()

    return mask


def random_mask(data_object, mask_ratio=0.5):
    n_tokens = data_object['token_index'].shape[0]
    n_mask = int(n_tokens * mask_ratio)

    mask_indices = torch.randperm(n_tokens)[:n_mask]
    mask = torch.ones(n_tokens, dtype=torch.float)
    mask[mask_indices] = 0.0

    print(f"Masked indices: {mask_indices.tolist()}")
    return mask


for data_length in [20, 100, 200, 500]:
    data = {'token_index': torch.arange(data_length)}
    print(f"Data length: {data_length}")

    print("Continuous Mask:")
    print(sum(continuous_mask(data, mask_ratio=0.5)))

