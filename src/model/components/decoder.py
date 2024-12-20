import torch
from torch import nn

from src.common.all_atom import compute_backbone


class DenoisingNet(nn.Module):
    def __init__(self,
                 embedder: nn.Module,
                 translator: nn.Module,
                 ):
        super(DenoisingNet, self).__init__()
        self.embedder = embedder  # embedding module
        self.translator = translator  # translationIPA

    def forward(self, batch, as_tensor_7=False):
        """Forward computes the denoised frames p(X^t|X^{t+1})
        """
        # Frames as [batch, res, 7] tensors.
        node_mask = batch['residue_mask'].type(torch.float)  # [B, N]
        fixed_mask = batch['fixed_mask'].type(torch.float)
        edge_mask = node_mask[..., None] * node_mask[..., None, :]

        # Get embeddings.
        node_embed, edge_embed = self.embedder(
            residue_idx=batch['residue_idx'],
            t=batch['t'],
            fixed_mask=fixed_mask,
            self_conditioning_ca=batch['sc_ca_t'],
        )
        node_embed = node_embed * node_mask[..., None]  # (L, D)
        edge_embed = edge_embed * edge_mask[..., None]  # (L, L, D)

        # Translation for frames.
        model_out = self.translator(node_embed, edge_embed, batch)

        # Psi angle prediction
        gt_psi = batch['torsion_angles_sin_cos'][..., 2, :]
        psi_pred = gt_psi * fixed_mask[..., None] + model_out['psi'] * (1 - fixed_mask[..., None])
        rigids_pred = model_out['out_rigids']

        bb_representations = compute_backbone(
            rigids_pred, psi_pred, aatype=batch['aatype'] if 'aatype' in batch else None
        )
        atom37_pos = bb_representations[0].to(rigids_pred.device)
        atom14_pos = bb_representations[-1].to(rigids_pred.device)

        if as_tensor_7:
            rigids_pred = rigids_pred.to_tensor_7()

        return {
            'rigids': rigids_pred,
            'psi': psi_pred,
            'atom37': atom37_pos,
            'atom14': atom14_pos,
        }