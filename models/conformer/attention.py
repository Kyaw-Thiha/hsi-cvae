from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding reused across modules."""

    pe: torch.Tensor

    def __init__(self, dim: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class RelativeMultiheadAttention(nn.Module):
    """Multi-head attention with learnable relative position bias."""

    def __init__(self, d_model: int, n_heads: int, dropout: float, max_len: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.max_rel_distance = max_len - 1
        self.relative_bias = nn.Embedding(2 * self.max_rel_distance + 1, n_heads)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, tgt_len, _ = query.shape
        src_len = key.size(1)

        q = self.q_proj(query).view(batch_size, tgt_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, src_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, src_len, self.n_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        rel_bias = self._relative_bias(tgt_len, src_len, query.device)
        attn_scores = attn_scores + rel_bias

        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, tgt_len, -1)
        return self.out_proj(attn_output)

    def _relative_bias(self, tgt_len: int, src_len: int, device: torch.device) -> torch.Tensor:
        range_q = torch.arange(tgt_len, device=device)
        range_k = torch.arange(src_len, device=device)
        rel_pos = range_q[:, None] - range_k[None, :]
        rel_pos = rel_pos.clamp(-self.max_rel_distance, self.max_rel_distance) + self.max_rel_distance
        bias = self.relative_bias(rel_pos).permute(2, 0, 1)
        return bias.unsqueeze(0)
