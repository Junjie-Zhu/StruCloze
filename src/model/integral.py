from typing import Optional, Union

import torch
import torch.nn as nn

from src.model.components.transformer import (AtomAttentionEncoder,
                                              AtomAttentionDecoder,
                                              DiffusionTransformer)
from src.model.components.primitives import LinearNoBias, LayerNorm
from src.model.components.embedder import EmbeddingModule


class FoldEmbedder(nn.Module):
    def __init__(
        self,
        c_atom: int = 128,
        c_atompair: int = 16,
        c_token: int = 768,
        c_s: int = 384,
        c_z: int = 128,
        n_atom_layers: int = 3,
        n_token_layers: int = 16,
        n_atom_attn_heads: int = 4,
        n_token_attn_heads: int = 8,
        initialization: Optional[dict[str, Union[str, float, bool]]] = None,
    ):
        super(FoldEmbedder, self).__init__()

        self.embedding_module = EmbeddingModule(
            c_s=c_s,
            c_z=c_z,
        )
        self.atom_encoder = AtomAttentionEncoder(
            has_coords=True,
            c_token=c_token,
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_s=c_s,
            c_z=c_z,
            n_blocks=n_atom_layers,
            n_heads=n_atom_attn_heads,
            n_queries=32,
            n_keys=128,  # parameters for local attention, typically not changed
        )
        self.token_transformer = DiffusionTransformer(
            c_a=c_token,
            c_s=c_s,
            c_z=c_z,
            n_blocks=n_token_layers,
            n_heads=n_token_attn_heads,
        )
        self.atom_decoder = AtomAttentionDecoder(
            c_token=c_token,
            c_atom=c_atom,
            c_atompair=c_atompair,
            n_blocks=n_atom_layers,
            n_heads=n_atom_attn_heads,
            n_queries=32,
            n_keys=128,  # parameters for local attention, typically not changed
        )

        # connecting layers
        self.layernorm_s = LayerNorm(c_s)
        self.linear_no_bias_s = LinearNoBias(in_features=c_s, out_features=c_token)
        self.layernorm_a = LayerNorm(c_token)

        # initialize parameters
        self.init_parameters(initialization)

    def init_parameters(self, initialization: dict):
        """
        Initializes the parameters of the diffusion module according to the provided initialization configuration.

        Args:
            initialization (dict): A dictionary containing initialization settings.
        """
        if initialization.get("zero_init_condition_transition", False):
            self.embedding_module.transition_z1.zero_init()
            self.embedding_module.transition_z2.zero_init()
            self.embedding_module.transition_s1.zero_init()
            self.embedding_module.transition_s2.zero_init()

        self.atom_encoder.linear_init(
            zero_init_atom_encoder_residual_linear=initialization.get(
                "zero_init_atom_encoder_residual_linear", False
            ),
            he_normal_init_atom_encoder_small_mlp=initialization.get(
                "he_normal_init_atom_encoder_small_mlp", False
            ),
            he_normal_init_atom_encoder_output=initialization.get(
                "he_normal_init_atom_encoder_output", False
            ),
        )

        if initialization.get("glorot_init_self_attention", False):
            for (
                    block
            ) in (
                    self.atom_encoder.atom_transformer.diffusion_transformer.blocks
            ):
                block.attention_pair_bias.glorot_init()

        for block in self.token_transformer.blocks:
            if initialization.get("zero_init_adaln", False):
                block.attention_pair_bias.layernorm_a.zero_init()
                block.conditioned_transition_block.adaln.zero_init()
            if initialization.get("zero_init_residual_condition_transition", False):
                nn.init.zeros_(
                    block.conditioned_transition_block.linear_nobias_b.weight
                )

        if initialization.get("zero_init_atom_decoder_linear", False):
            nn.init.zeros_(self.atom_decoder.linear_no_bias_a.weight)

        if initialization.get("zero_init_dit_output", False):
            nn.init.zeros_(self.atom_decoder.linear_no_bias_out.weight)

    def forward(
        self,
        initial_positions: torch.Tensor,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
    ) -> torch.Tensor:

        # scaling initial positions to ensure approximately unit variance
        initial_positions = initial_positions / 16.

        # encode token-level features
        s_single, z_pair = self.embedding_module(input_feature_dict)

        # add num_sample dimension
        # initial_positions = initial_positions.unsqueeze(1)
        s_single = s_single.unsqueeze(1)
        z_pair = z_pair.unsqueeze(1)
        
        # encode atom-level features
        a_token, q_skip, c_skip, p_skip = self.atom_encoder(
            input_feature_dict=input_feature_dict,
            r_l=initial_positions,  # structure constructed on CG repr.
            s=s_single,
            z=z_pair,
        )

        # attention on token-level
        a_token = a_token + self.linear_no_bias_s(
            self.layernorm_s(s_single)
        )  # [..., N_sample, N_token, c_token]
        a_token = self.token_transformer(
            a=a_token,
            s=s_single,
            z=z_pair,
        )
        a_token = self.layernorm_a(a_token)

        # decode atom-level features
        r_update = self.atom_decoder(
            input_feature_dict=input_feature_dict,
            a=a_token,
            q_skip=q_skip,
            c_skip=c_skip,
            p_skip=p_skip,
        )
        return r_update * 16.  # rescale back to original scale

