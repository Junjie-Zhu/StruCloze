import math
from functools import partial

import torch
import torch.nn as nn

from src.model.components.encoder import EmbeddingAndSeqformer
from src.model.components.decoder import DenoisingNet
from src.model.components.decoder_layers import Linear
from src.utils.all_atom import calc_distogram
from src.utils.model_utils import aggregate_features


def get_positional_embedding(indices, embedding_dim, max_len=2056):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., N_edges] of type integer
        max_len: maximum length.
        embedding_dim: dimension of the embeddings to create

    Returns:
        positional embedding of shape [N, embedding_dim]
    """
    K = torch.arange(embedding_dim//2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embedding_dim))).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embedding_dim))).to(indices.device)
    pos_embedding = torch.cat([
        pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


class FoldEmbedder(nn.Module):
    def __init__(self,
                 encoder_config: dict,
                 latent_config: dict,
                 decoder_config: dict,
                 self_conditioning: bool = True,
                 ):
        super(FoldEmbedder, self).__init__()

        self.encoder = EmbeddingAndSeqformer(encoder_config)
        self.decoder = DenoisingNet(decoder_config)

        self.latent_c = latent_config
        self.condition = self_conditioning

        if self.latent_c['use_pair'] == 'False':
            self.latent_transform = nn.Sequential(
                Linear(self.latent_c['in_dim'], self.latent_c['lat_dim'], bias=False),
                nn.Tanh()
            )
        else:
            # graph-like update single repr.
            self.graph_transform = nn.Sequential(
                Linear(self.latent_c['in_dim'] * 2 + self.latent_c['edge_in_dim'], self.latent_c['lat_dim'], bias=False),
                nn.ReLU(),
            )
            self.latent_transform = nn.Sequential(
                Linear(self.latent_c['lat_dim'], self.latent_c['lat_dim'], bias=False),
                nn.Tanh()
            )

        self.position_embed = partial(
            get_positional_embedding, embedding_dim=self.latent_c['pos_embed_dim']
        )
        self.distogram_embed = partial(
            calc_distogram,
            min_bin=self.latent_c['min_bin'],
            max_bin=self.latent_c['max_bin'],
            num_bins=self.latent_c['num_bins']
        )

        self.latent_decode = nn.Sequential(
            Linear(self.latent_c['lat_dim'], self.latent_c['out_dim'], bias=False),
            nn.ReLU(),
            Linear(self.latent_c['out_dim'], self.latent_c['out_dim']),
            nn.ReLU()
        )

        self.latent_edge_decode = nn.Sequential(
            Linear(self.latent_c['edge_lat_dim'], self.latent_c['edge_out_dim'], bias=False),
            nn.ReLU(),
            Linear(self.latent_c['edge_out_dim'], self.latent_c['edge_out_dim']),
            nn.ReLU()
        )

    def forward(self, batch):
        batch_size, num_node = batch['residue_index'].shape

        s_encode, z_encode = self.encoder(batch)

        if self.latent_c['use_pair'] == 'False':
            s_encode = self.latent_transform(s_encode)
        else:
            s_encode_ = torch.zeros_like(s_encode)
            s_encode = aggregate_features(s_encode, batch['edge_index'], z_encode)
            s_encode = self.graph_transform(s_encode)

            s_encode_.scatter_add_(
                src=s_encode,
                dim=1,
                index=batch['edge_index'].reshape(
                    batch_size, num_node, 1
                ).expand(batch_size, num_node, s_encode.shape[-1])
            )
            s_encode = self.latent_decode(s_encode_)

        # positional embedding on residue index
        s_decode = self.latent_decode(torch.cat([
            self.position_embed(batch['residue_index']),
            s_encode,
        ], dim=-1))

        z_decode = []

        relative_position_embed = (
            batch['residue_index'][:, :, None] - batch['residue_index'][:, None, :]
        ).reshape(
            [batch_size, num_node ** 2]
        )
        z_decode.append(self.position_embed(relative_position_embed))

        z_init = torch.cat([
            torch.tile(s_encode[:, :, None, :], (1, 1, num_node, 1)),
            torch.tile(s_encode[:, None, :, :], (1, num_node, 1, 1))
        ], dim=-1).float().reshape([batch_size, num_node ** 2, -1])
        z_decode.append(z_init)

        if self.condition:
            z_decode.append(
                self.distogram_embed(batch['condition_ca']).reshape([batch_size, num_node ** 2, -1])
            )
        z_decode = self.latent_edge_decode(torch.cat(z_decode, dim=-1))

        output_dict = self.decoder(s_decode, z_decode, batch)
        output_dict.update(
            {'s_encode': s_encode,}
        )

        return output_dict

