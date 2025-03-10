from typing import Optional

import torch
import torch.nn as nn

from src.utils.model_utils import expand_at_dim, get_checkpoint_fn
from src.metrics.rmsd import weighted_rigid_align


def loss_reduction(loss: torch.Tensor, method: str = "mean") -> torch.Tensor:
    """reduction wrapper

    Args:
        loss (torch.Tensor): loss
            [...]
        method (str, optional): reduction method. Defaults to "mean".

    Returns:
        torch.Tensor: reduced loss
            [] or [...]
    """

    if method is None:
        return loss
    assert method in ["mean", "sum", "add", "max", "min"]
    if method == "add":
        method = "sum"
    return getattr(torch, method)(loss)


class MSELoss(nn.Module):
    """
    Implements Formula 2-4 [MSELoss] in AF3
    """

    def __init__(
        self,
        weight_mse: float = 1 / 3,
        eps=1e-6,
        reduction: str = "mean",
    ) -> None:
        super(MSELoss, self).__init__()
        self.weight_mse = weight_mse
        self.eps = eps
        self.reduction = reduction

    def weighted_rigid_align(
        self,
        pred_coordinate: torch.Tensor,
        true_coordinate: torch.Tensor,
        coordinate_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """compute weighted rigid alignment results

        Args:
            pred_coordinate (torch.Tensor): the denoised coordinates from diffusion module
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): the ground truth coordinates
                [..., N_atom, 3]
            coordinate_mask (torch.Tensor): whether true coordinates exist
                [N_atom] or [..., N_atom]
            is_dna / is_rna / is_ligand (torch.Tensor): mol type mask
                [N_atom] or [..., N_atom]

        Returns:
            true_coordinate_aligned (torch.Tensor): aligned coordinates for each sample
                [..., N_sample, N_atom, 3]
            weight (torch.Tensor): weights for each atom
                [N_atom] or [..., N_sample, N_atom]
        """
        N_sample = pred_coordinate.size(-3)
        # weight = (
        #     1
        #     + self.weight_dna * is_dna
        #     + self.weight_rna * is_rna
        #     + self.weight_ligand * is_ligand
        # )  # [N_atom] or [..., N_atom]
        weight = torch.ones_like(coordinate_mask)

        # Apply coordinate_mask
        weight = weight * coordinate_mask  # [N_atom] or [..., N_atom]
        true_coordinate = true_coordinate * coordinate_mask.unsqueeze(dim=-1)
        pred_coordinate = pred_coordinate * coordinate_mask[..., None, :, None]

        # Reshape to add "N_sample" dimension
        true_coordinate = expand_at_dim(
            true_coordinate, dim=-3, n=N_sample
        )  # [..., N_sample, N_atom, 3]
        if len(weight.shape) > 1:
            weight = expand_at_dim(
                weight, dim=-2, n=N_sample
            )  # [..., N_sample, N_atom]

        # Align GT coords to predicted coords
        d = pred_coordinate.dtype
        # Some ops in weighted_rigid_align do not support BFloat16 training
        with torch.cuda.amp.autocast(enabled=False):
            true_coordinate_aligned = weighted_rigid_align(
                x=true_coordinate.to(torch.float32),  # [..., N_sample, N_atom, 3]
                x_target=pred_coordinate.to(
                    torch.float32
                ),  # [..., N_sample, N_atom, 3]
                atom_weight=weight.to(
                    torch.float32
                ),  # [N_atom] or [..., N_sample, N_atom]
                stop_gradient=True,
            )  # [..., N_sample, N_atom, 3]
            true_coordinate_aligned = true_coordinate_aligned.to(d)

        return (true_coordinate_aligned.detach(), weight.detach())

    def forward(
        self,
        pred_coordinate: torch.Tensor,
        true_coordinate: torch.Tensor,
        coordinate_mask: torch.Tensor,
        per_sample_scale: torch.Tensor = None,
    ) -> torch.Tensor:
        """MSELoss

        Args:
            pred_coordinate (torch.Tensor): the denoised coordinates from diffusion module.
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): the ground truth coordinates.
                [..., N_atom, 3]
            coordinate_mask (torch.Tensor): whether true coordinates exist.
                [N_atom] or [..., N_atom]
            is_dna / is_rna / is_ligand (torch.Tensor): mol type mask.
                [N_atom] or [..., N_atom]
            per_sample_scale (torch.Tensor, optional): whether to scale the loss by the per-sample noise-level.
                [..., N_sample]

        Returns:
            torch.Tensor: the weighted mse loss.
                [...] is self.reduction is None else []
        """
        # True_coordinate_aligned: [..., N_sample, N_atom, 3]
        # Weight: [N_atom] or [..., N_sample, N_atom]
        # with torch.no_grad():
        #     true_coordinate_aligned, weight = self.weighted_rigid_align(
        #         pred_coordinate=pred_coordinate,
        #         true_coordinate=true_coordinate,
        #         coordinate_mask=coordinate_mask,
        #     )
        N_sample = pred_coordinate.size(-3)
        true_coordinate_aligned = expand_at_dim(
            true_coordinate, dim=-3, n=N_sample
        )  # [..., N_sample, N_atom, 3]

        weight = torch.ones_like(coordinate_mask)
        if len(weight.shape) > 1:
            weight = expand_at_dim(
                weight, dim=-2, n=N_sample
            )  # [..., N_sample, N_atom]

        # Calculate MSE loss
        per_atom_se = ((pred_coordinate - true_coordinate_aligned) ** 2).sum(
            dim=-1
        )  # [..., N_sample, N_atom]
        per_sample_weighted_mse = (weight * per_atom_se).sum(dim=-1) / (
            coordinate_mask.sum(dim=-1, keepdim=True) + self.eps
        )  # [..., N_sample]

        if per_sample_scale is not None:
            per_sample_weighted_mse = per_sample_weighted_mse * per_sample_scale

        weighted_align_mse_loss = self.weight_mse * (per_sample_weighted_mse).mean(
            dim=-1
        )  # [...]

        loss = loss_reduction(weighted_align_mse_loss, method=self.reduction)

        return loss


class SmoothLDDTLoss(nn.Module):
    """
    Implements Algorithm 27 [SmoothLDDTLoss] in AF3
    """

    def __init__(
        self,
        eps: float = 1e-10,
        reduction: str = "mean",
    ) -> None:
        """SmoothLDDTLoss

        Args:
            eps (float, optional): avoid nan. Defaults to 1e-10.
            reduction (str, optional): reduction method for the batch dims. Defaults to mean.
        """
        super(SmoothLDDTLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def _chunk_forward(self, pred_distance, true_distance, c_lm=None):
        dist_diff = torch.abs(pred_distance - true_distance.unsqueeze(1))
        # For save cuda memory we use inplace op
        dist_diff_epsilon = 0
        for threshold in [0.5, 1, 2, 4]:
            dist_diff_epsilon += 0.25 * torch.sigmoid(threshold - dist_diff)

        # Compute mean
        if c_lm is not None:
            lddt = torch.sum(c_lm * dist_diff_epsilon, dim=(-1, -2)) / (
                torch.sum(c_lm, dim=(-1, -2)) + self.eps
            )  # [..., N_sample]
        else:
            # It's for sparse forward mode
            lddt = torch.mean(dist_diff_epsilon, dim=-1)
        return lddt

    def forward(
        self,
        pred_distance: torch.Tensor,
        true_distance: torch.Tensor,
        distance_mask: torch.Tensor,
        lddt_mask: torch.Tensor,
        diffusion_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """SmoothLDDTLoss

        Args:
            pred_distance (torch.Tensor): the diffusion denoised atom-atom distance
                [..., N_sample, N_atom, N_atom]
            true_distance (torch.Tensor): the ground truth coordinates
                [..., N_atom, N_atom]
            distance_mask (torch.Tensor): whether true coordinates exist.
                [N_atom, N_atom]
            lddt_mask (torch.Tensor, optional): whether true distance is within radius (30A for nuc and 15A for others)
                [N_atom, N_atom]
            diffusion_chunk_size (Optional[int]): Chunk size over the N_sample dimension. Defaults to None.

        Returns:
            torch.Tensor: the smooth lddt loss
                [...] if reduction is None else []
        """
        c_lm = lddt_mask.bool().unsqueeze(dim=-3).detach()  # [..., 1, N_atom, N_atom]
        # Compute distance error
        # [...,  N_sample , N_atom, N_atom]
        if diffusion_chunk_size is None:
            lddt = self._chunk_forward(
                pred_distance=pred_distance, true_distance=true_distance, c_lm=c_lm
            )
        else:
            # Default use checkpoint for saving memory
            checkpoint_fn = get_checkpoint_fn()
            lddt = []
            N_sample = pred_distance.shape[-3]
            no_chunks = N_sample // diffusion_chunk_size + (
                N_sample % diffusion_chunk_size != 0
            )
            for i in range(no_chunks):
                lddt_i = checkpoint_fn(
                    self._chunk_forward,
                    pred_distance[
                        ...,
                        i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size,
                        :,
                        :,
                    ],
                    true_distance,
                    c_lm,
                )
                lddt.append(lddt_i)
            lddt = torch.cat(lddt, dim=-1)

        lddt = lddt.mean(dim=-1)  # [...]
        return 1 - loss_reduction(lddt, method=self.reduction)

    def sparse_forward(
        self,
        pred_coordinate: torch.Tensor,
        true_coordinate: torch.Tensor,
        lddt_mask: torch.Tensor,
        diffusion_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """SmoothLDDTLoss sparse implementation

        Args:
            pred_coordinate (torch.Tensor): the diffusion denoised atom coordinates
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): the ground truth atom coordinates
                [..., N_atom, 3]
            lddt_mask (torch.Tensor, optional): whether true distance is within radius (30A for nuc and 15A for others)
                [N_atom, N_atom]
            diffusion_chunk_size (Optional[int]): Chunk size over the N_sample dimension. Defaults to None.

        Returns:
            torch.Tensor: the smooth lddt loss
                [...] if reduction is None else []
        """
        lddt_indices = torch.nonzero(lddt_mask, as_tuple=True)
        true_coords_l = true_coordinate.index_select(-2, lddt_indices[0])
        true_coords_m = true_coordinate.index_select(-2, lddt_indices[1])
        true_distance_sparse_lm = torch.norm(true_coords_l - true_coords_m, p=2, dim=-1)
        if diffusion_chunk_size is None:
            pred_coords_l = pred_coordinate.index_select(-2, lddt_indices[0])
            pred_coords_m = pred_coordinate.index_select(-2, lddt_indices[1])
            # \delta?x_{lm} and \delta?x_{lm}^{GT} in the Algorithm 27
            pred_distance_sparse_lm = torch.norm(
                pred_coords_l - pred_coords_m, p=2, dim=-1
            )
            lddt = self._chunk_forward(
                pred_distance_sparse_lm, true_distance_sparse_lm, c_lm=None
            )
        else:
            checkpoint_fn = get_checkpoint_fn()
            lddt = []
            N_sample = pred_coordinate.shape[-3]
            no_chunks = N_sample // diffusion_chunk_size + (
                N_sample % diffusion_chunk_size != 0
            )
            for i in range(no_chunks):
                pred_coords_i_l = pred_coordinate[
                    i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size, :, :
                ].index_select(-2, lddt_indices[0])
                pred_coords_i_m = pred_coordinate[
                    i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size, :, :
                ].index_select(-2, lddt_indices[1])

                # \delta?x_{lm} and \delta?x_{lm}^{GT} in the Algorithm 27
                pred_distance_sparse_i_lm = torch.norm(
                    pred_coords_i_l - pred_coords_i_m, p=2, dim=-1
                )
                lddt_i = checkpoint_fn(
                    self._chunk_forward,
                    pred_distance_sparse_i_lm,
                    true_distance_sparse_lm,
                )
                lddt.append(lddt_i)
            lddt = torch.cat(lddt, dim=-1)

        lddt = lddt.mean(dim=-1)  # [...]
        return 1 - loss_reduction(lddt, method=self.reduction)

    def dense_forward(
        self,
        pred_coordinate: torch.Tensor,
        true_coordinate: torch.Tensor,
        lddt_mask: torch.Tensor,
        diffusion_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """SmoothLDDTLoss sparse implementation

        Args:
            pred_coordinate (torch.Tensor): the diffusion denoised atom coordinates
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): the ground truth atom coordinates
                [..., N_atom, 3]
            lddt_mask (torch.Tensor, optional): whether true distance is within radius (30A for nuc and 15A for others)
                [N_atom, N_atom]
            diffusion_chunk_size (Optional[int]): Chunk size over the N_sample dimension. Defaults to None.

        Returns:
            torch.Tensor: the smooth lddt loss
                [...] if reduction is None else []
        """
        c_lm = lddt_mask.bool().unsqueeze(dim=-3).detach()  # [..., 1, N_atom, N_atom]
        # Compute distance error
        # [...,  N_sample , N_atom, N_atom]
        true_distance = torch.cdist(true_coordinate, true_coordinate)
        if diffusion_chunk_size is None:
            pred_distance = torch.cdist(pred_coordinate, pred_coordinate)
            lddt = self._chunk_forward(
                pred_distance=pred_distance, true_distance=true_distance, c_lm=c_lm
            )
        else:
            checkpoint_fn = get_checkpoint_fn()
            lddt = []
            N_sample = pred_coordinate.shape[-3]
            no_chunks = N_sample // diffusion_chunk_size + (
                N_sample % diffusion_chunk_size != 0
            )
            for i in range(no_chunks):
                pred_distance_i = torch.cdist(
                    pred_coordinate[
                        i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size,
                        :,
                        :,
                    ],
                    pred_coordinate[
                        i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size,
                        :,
                        :,
                    ],
                )
                lddt_i = checkpoint_fn(
                    self._chunk_forward,
                    pred_distance_i,
                    true_distance,
                    c_lm,
                )
                lddt.append(lddt_i)
            lddt = torch.cat(lddt, dim=-1)

        lddt = lddt.mean(dim=-1)  # [...]
        return 1 - loss_reduction(lddt, method=self.reduction)


class BondLoss(nn.Module):
    """
    Implements Formula 5 [BondLoss] in AF3
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean") -> None:
        """BondLoss

        Args:
            eps (float, optional): avoid nan. Defaults to 1e-6.
            reduction (str, optional): reduction method for the batch dims. Defaults to mean.
        """
        super(BondLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def _chunk_forward(self, pred_distance, true_distance, bond_mask):
        # Distance squared error
        # [...,  N_sample , N_atom, N_atom]
        dist_squared_err = (pred_distance - true_distance.unsqueeze(dim=-3)) ** 2
        bond_loss = torch.sum(dist_squared_err * bond_mask, dim=(-1, -2)) / torch.sum(
            bond_mask + self.eps, dim=(-1, -2)
        )  # [..., N_sample]
        return bond_loss

    def forward(
        self,
        pred_distance: torch.Tensor,
        true_distance: torch.Tensor,
        distance_mask: torch.Tensor,
        bond_mask: torch.Tensor,
        per_sample_scale: torch.Tensor = None,
        diffusion_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """BondLoss

        Args:
            pred_distance (torch.Tensor): the diffusion denoised atom-atom distance
                [..., N_sample, N_atom, N_atom]
            true_distance (torch.Tensor): the ground truth coordinates
                [..., N_atom, N_atom]
            distance_mask (torch.Tensor): whether true coordinates exist.
                [N_atom, N_atom] or [..., N_atom, N_atom]
            bond_mask (torch.Tensor): bonds considered in this loss
                [N_atom, N_atom] or [..., N_atom, N_atom]
            per_sample_scale (torch.Tensor, optional): whether to scale the loss by the per-sample noise-level.
                [..., N_sample]
            diffusion_chunk_size (Optional[int]): Chunk size over the N_sample dimension. Defaults to None.

        Returns:
            torch.Tensor: the bond loss
                [...] if reduction is None else []
        """

        bond_mask = (bond_mask * distance_mask).unsqueeze(
            dim=-3
        )  # [1, N_atom, N_atom] or [..., 1, N_atom, N_atom]
        # Bond Loss
        if diffusion_chunk_size is None:
            bond_loss = self._chunk_forward(
                pred_distance=pred_distance,
                true_distance=true_distance,
                bond_mask=bond_mask,
            )
        else:
            checkpoint_fn = get_checkpoint_fn()
            bond_loss = []
            N_sample = pred_distance.shape[-3]
            no_chunks = N_sample // diffusion_chunk_size + (
                N_sample % diffusion_chunk_size != 0
            )
            for i in range(no_chunks):
                bond_loss_i = checkpoint_fn(
                    self._chunk_forward,
                    pred_distance[
                        ...,
                        i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size,
                        :,
                        :,
                    ],
                    true_distance,
                    bond_mask,
                )
                bond_loss.append(bond_loss_i)
            bond_loss = torch.cat(bond_loss, dim=-1)
        if per_sample_scale is not None:
            bond_loss = bond_loss * per_sample_scale

        bond_loss = bond_loss.mean(dim=-1)  # [...]
        return loss_reduction(bond_loss, method=self.reduction)

    def sparse_forward(
        self,
        pred_coordinate: torch.Tensor,
        true_coordinate: torch.Tensor,
        distance_mask: torch.Tensor,
        bond_mask: torch.Tensor,
        per_sample_scale: torch.Tensor = None,
    ) -> torch.Tensor:
        """BondLoss sparse implementation

        Args:
            pred_coordinate (torch.Tensor): the diffusion denoised atom coordinates
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): the ground truth atom coordinates
                [..., N_atom, 3]
            distance_mask (torch.Tensor): whether true coordinates exist.
                [N_atom, N_atom] or [..., N_atom, N_atom]
            bond_mask (torch.Tensor): bonds considered in this loss
                [N_atom, N_atom] or [..., N_atom, N_atom]
            per_sample_scale (torch.Tensor, optional): whether to scale the loss by the per-sample noise-level.
                [..., N_sample]
        Returns:
            torch.Tensor: the bond loss
                [...] if reduction is None else []
        """

        bond_mask = bond_mask * distance_mask
        bond_indices = torch.nonzero(bond_mask, as_tuple=True)
        pred_coords_i = pred_coordinate.index_select(-2, bond_indices[0])
        pred_coords_j = pred_coordinate.index_select(-2, bond_indices[1])
        true_coords_i = true_coordinate.index_select(-2, bond_indices[0])
        true_coords_j = true_coordinate.index_select(-2, bond_indices[1])

        pred_distance_sparse = torch.norm(pred_coords_i - pred_coords_j, p=2, dim=-1)
        true_distance_sparse = torch.norm(true_coords_i - true_coords_j, p=2, dim=-1)
        dist_squared_err_sparse = (pred_distance_sparse - true_distance_sparse) ** 2
        # Protecting special data that has size: tensor([], size=(x, 0), grad_fn=<PowBackward0>)
        if dist_squared_err_sparse.numel() == 0:
            return torch.tensor(
                0.0, device=dist_squared_err_sparse.device, requires_grad=True
            )
        bond_loss = torch.mean(dist_squared_err_sparse, dim=-1)  # [..., N_sample]
        if per_sample_scale is not None:
            bond_loss = bond_loss * per_sample_scale

        bond_loss = bond_loss.mean(dim=-1)  # [...]
        return bond_loss


class AllLosses(nn.Module):
    def __init__(self,
                 weight_mse: float = 1.,
                 eps: float = 1e-6,
                 reduction: str = "mean") -> None:
        super(AllLosses, self).__init__()
        self.mse_loss = MSELoss(weight_mse=weight_mse, eps=eps, reduction=reduction)
        self.smooth_lddt_loss = SmoothLDDTLoss(eps=eps, reduction=reduction)
        self.bond_loss = BondLoss(eps=eps, reduction=reduction)

    def calculate_losses(self,
                         pred_positions,
                         true_positions,
                         single_mask=None,
                         pair_mask=None,
                         lddt_enabled=False,
                         bond_enabled=False
    ):
        if single_mask is None:
            single_mask = torch.ones_like(true_positions[..., 0])
        if pair_mask is None:
            pair_mask = single_mask[..., None, :] * single_mask[..., :, None]  # (batch, N, N)

        losses = {}
        # Calculate MSE loss
        losses["mse_loss"] = self.mse_loss(
            pred_coordinate=pred_positions,
            true_coordinate=true_positions,
            coordinate_mask=single_mask
        )
        if lddt_enabled:
            # Calculate SmoothLDDT loss
            losses["smooth_lddt_loss"] = self.smooth_lddt_loss.dense_forward(
                pred_coordinate=pred_positions,
                true_coordinate=true_positions,
                lddt_mask=pair_mask,
            )
        if bond_enabled:
            # Calculate Bond loss, to revise
            losses["bond_loss"] = self.bond_loss.sparse_forward(
                pred_coordinate=pred_positions,
                true_coordinate=true_positions,
                distance_mask=pair_mask,
                bond_mask=pair_mask,
            )

        cum_loss = 0
        seperate_loss = []
        for loss_item, loss_outputs in losses.items():
            if isinstance(loss_outputs, tuple):
                loss, metrics = loss_outputs
            else:
                assert isinstance(loss_outputs, torch.Tensor)
                loss, metrics = loss_outputs, {}
            cum_loss += loss
            seperate_loss.append(loss)
        return cum_loss, seperate_loss

    def forward(self,
                pred_positions,
                true_positions,
                single_mask=None,
                pair_mask=None,
                lddt_enabled=False,
                bond_enabled=False
                ):
        return self.calculate_losses(
            pred_positions, 
            true_positions,
            single_mask=single_mask,
            pair_mask=pair_mask,
            lddt_enabled=lddt_enabled,
            bond_enabled=bond_enabled
        )
