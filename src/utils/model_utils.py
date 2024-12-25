import warnings

import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler


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


def get_optimizer(
        configs: dict,
        model: nn.Module,
):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=configs['lr'],
        weight_decay=configs['weight_decay'],
        betas=(configs['beta1'], configs['beta2']),
    )
    return optimizer


def get_lr_scheduler(
        configs: dict,
        optimizer: torch.optim.Optimizer,
):
    scheduler = AlphaFold3LRScheduler(
        optimizer,
        lr=configs['lr'],
        warmup_steps=configs['warmup_steps'],
        decay_every_n_steps=configs['decay_every_n_steps'],
        decay_factor=configs['decay_factor'],
    )
    return scheduler


class AlphaFold3LRScheduler(LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        last_epoch: int = -1,
        verbose: bool = False,
        warmup_steps: int = 1000,
        lr: float = 1.8e-3,
        decay_every_n_steps: int = 50000,
        decay_factor: float = 0.95,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_every_n_steps
        self.lr = lr
        self.decay_factor = decay_factor
        super(AlphaFold3LRScheduler, self).__init__(
            optimizer=optimizer, last_epoch=last_epoch, verbose=verbose
        )

    def _get_step_lr(self, step):
        if step <= self.warmup_steps:
            lr = step / self.warmup_steps * self.lr
        else:
            decay_count = step // self.decay_steps
            lr = self.lr * (self.decay_factor**decay_count)
        return lr

    def get_lr(self) -> list[float]:
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )
        return [
            self._get_step_lr(self.last_epoch) for group in self.optimizer.param_groups
        ]

