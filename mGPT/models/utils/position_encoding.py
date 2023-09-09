# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
from typing import List, Optional

import numpy as np
import torch
from torch import Tensor, nn

# from util.misc import NestedTensor


class NestedTensor(object):

    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self,
                 num_pos_feats=64,
                 temperature=10000,
                 normalize=False,
                 scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats,
                             dtype=torch.float32,
                             device=x.device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ],
                        dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(
                            x.shape[0], 1, 1, 1)
        return pos


class PositionEmbeddingSine1D(nn.Module):

    def __init__(self, d_model, max_len=500, batch_first=False):
        super().__init__()
        self.batch_first = batch_first

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        if self.batch_first:
            pos = self.pe.permute(1, 0, 2)[:, :x.shape[1], :]
        else:
            pos = self.pe[:x.shape[0], :]
        return pos


class PositionEmbeddingLearned1D(nn.Module):

    def __init__(self, d_model, max_len=500, batch_first=False):
        super().__init__()
        self.batch_first = batch_first
        # self.dropout = nn.Dropout(p=dropout)

        self.pe = nn.Parameter(torch.zeros(max_len, 1, d_model))
        # self.pe = pe.unsqueeze(0).transpose(0, 1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.pe)

    def forward(self, x):
        # not used in the final model
        if self.batch_first:
            pos = self.pe.permute(1, 0, 2)[:, :x.shape[1], :]
        else:
            x = x + self.pe[:x.shape[0], :]
        return x
        # return self.dropout(x)


def build_position_encoding(N_steps,
                            position_embedding="sine",
                            embedding_dim="1D"):
    # N_steps = hidden_dim // 2
    if embedding_dim == "1D":
        if position_embedding in ('v2', 'sine'):
            position_embedding = PositionEmbeddingSine1D(N_steps)
        elif position_embedding in ('v3', 'learned'):
            position_embedding = PositionEmbeddingLearned1D(N_steps)
        else:
            raise ValueError(f"not supported {position_embedding}")
    elif embedding_dim == "2D":
        if position_embedding in ('v2', 'sine'):
            # TODO find a better way of exposing other arguments
            position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
        elif position_embedding in ('v3', 'learned'):
            position_embedding = PositionEmbeddingLearned(N_steps)
        else:
            raise ValueError(f"not supported {position_embedding}")
    else:
        raise ValueError(f"not supported {embedding_dim}")

    return position_embedding
