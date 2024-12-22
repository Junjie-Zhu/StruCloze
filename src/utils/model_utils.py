from typing_extensions import Optional

import torch


def aggregate_features(
        features: torch.Tensor,
        indices: torch.Tensor,
        edge_features: torch.Tensor,
):
    """
    aggregate features based on indices

    :param features: [B, N, F]
    :param indices: [B, 2, M]
    :param edge_features: [B, N, N, E]
    :return: [B, M, F]
    """

    B, N = features.shape[:2]
    M = indices.shape[-1]

    # [B, 2, M, F]
    gathered_features = torch.gather(
        features.unsqueeze(2).expand(B, N, M, -1),
        1,
        indices.unsqueeze(-1).expand(B, 2, M, -1),
    )

    edge_features = edge_features.reshape(B, N * N, -1)
    if M != N * N:
        edge_indices = indices[:, 0, :].squeeze() * N + indices[:, 1, :].squeeze()
        edge_features = torch.gather(
            edge_features,
            1,
            edge_indices.unsqueeze(-1).expand(B, -1, edge_features.shape[-1])
        )

    gathered_features = torch.cat([
        gathered_features[:, 0, :, :].squeeze(),
        edge_features,
        gathered_features[:, 1, :, :].squeeze()
    ], dim=-1)

    return gathered_features

