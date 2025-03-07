from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.components.primitives import LinearNoBias, Transition, LayerNorm


class RelativePositionEncoding(nn.Module):
    """
    Implements Algorithm 3 in AF3
    """

    def __init__(self, r_max: int = 32, s_max: int = 2, c_z: int = 128) -> None:
        """
        Args:
            r_max (int, optional): Relative position indices clip value. Defaults to 32.
            s_max (int, optional): Relative chain indices clip value. Defaults to 2.
            c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        """
        super(RelativePositionEncoding, self).__init__()
        self.r_max = r_max
        self.s_max = s_max
        self.c_z = c_z
        self.linear_no_bias = LinearNoBias(
            in_features=(4 * self.r_max + 4), out_features=self.c_z
        )
        self.input_feature = {
            "asym_id": 1,
            "residue_index": 1,
            "entity_id": 1,
            "sym_id": 1,
            "token_index": 1,
        }

    def forward(self, input_feature_dict: dict[str, Any]) -> torch.Tensor:
        """
        Args:
            input_feature_dict (Dict[str, Any]): input meta feature dict.
            asym_id / residue_index / entity_id / sym_id / token_index
                [..., N_tokens]
        Returns:
            torch.Tensor: relative position encoding
                [..., N_token, N_token, c_z]
        """
        b_same_chain = (
                input_feature_dict["chain_index"][..., :, None]
                == input_feature_dict["chain_index"][..., None, :]
        ).long()  # [..., N_token, N_token]
        b_same_residue = (
                input_feature_dict["residue_index"][..., :, None]
                == input_feature_dict["residue_index"][..., None, :]
        ).long()  # [..., N_token, N_token]

        d_residue = torch.clip(
            input=input_feature_dict["residue_index"][..., :, None]
                  - input_feature_dict["residue_index"][..., None, :]
                  + self.r_max,
            min=0,
            max=2 * self.r_max,
        ) * b_same_chain + (1 - b_same_chain) * (
                            2 * self.r_max + 1
                    )  # [..., N_token, N_token]
        a_rel_pos = F.one_hot(d_residue, 2 * (self.r_max + 1))

        d_token = torch.clip(
            input=input_feature_dict["token_index"][..., :, None]
                  - input_feature_dict["token_index"][..., None, :]
                  + self.r_max,
            min=0,
            max=2 * self.r_max,
        ) * b_same_chain * b_same_residue + (1 - b_same_chain * b_same_residue) * (
                          2 * self.r_max + 1
                  )  # [..., N_token, N_token]
        a_rel_token = F.one_hot(d_token, 2 * (self.r_max + 1))

        p = self.linear_no_bias(
            torch.cat(
                [a_rel_pos, a_rel_token],
                dim=-1,
            ).float()
        )  # [..., N_token, N_token, 2 * (self.r_max + 1)+ 2 * (self.r_max + 1)+ 1 + 2 * (self.s_max + 1)] -> [..., N_token, N_token, c_z]
        return p


class EmbeddingModule(nn.Module):
    def __init__(self,
                 c_s: int = 384,
                 c_z: int = 128,
    ):
        super(EmbeddingModule, self).__init__()
        self.linear_no_bias_s = nn.Sequential(
            LinearNoBias(in_features=30, out_features=c_s),
            LayerNorm(c_s),
            LinearNoBias(in_features=c_s, out_features=c_s)
        )
        self.transition_s1 = Transition(c_s, n=2)
        self.transition_s2 = Transition(c_s, n=2)

        self.rel_pos_encode = RelativePositionEncoding(c_z=c_z)
        self.layernorm_z = nn.LayerNorm(c_z)
        self.linear_no_bias_z = LinearNoBias(in_features=c_z, out_features=c_z)
        self.transition_z1 = Transition(c_z, n=2)
        self.transition_z2 = Transition(c_z, n=2)

    def forward(self, input_feature_dict):
        restypes = F.one_hot(input_feature_dict["aatype"], num_classes=30).float()
        s_trunk = self.linear_no_bias_s(restypes)
        s_trunk = s_trunk + self.transition_s1(s_trunk)
        s_trunk = s_trunk + self.transition_s2(s_trunk)

        z_trunk = self.rel_pos_encode(input_feature_dict)
        z_trunk = self.linear_no_bias_z(self.layernorm_z(z_trunk))
        z_trunk = z_trunk + self.transition_z1(z_trunk)
        z_trunk = z_trunk + self.transition_z2(z_trunk)

        # maybe to add a distance embedding here
        return s_trunk, z_trunk

