from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .attention import PositionalEncoding, RelativeMultiheadAttention


class ConformerFeedForward(nn.Module):
    """Macaron-style feed-forward module with half-step residual scaling."""

    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        inner_dim = expansion * d_model
        self.layer_norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, inner_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.5 * self.net(self.layer_norm(x))


class ConformerConvModule(nn.Module):
    """Depthwise convolution block to capture local spectral structure."""

    def __init__(self, d_model: int, kernel_size: int = 17, dropout: float = 0.1) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_in = nn.Linear(d_model, 2 * d_model)
        self.depthwise_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        x = self.pointwise_in(x)
        x, gate = x.chunk(2, dim=-1)
        x = x * torch.sigmoid(gate)
        x = x.transpose(1, 2)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x).transpose(1, 2)
        x = self.pointwise_out(x)
        x = self.dropout(x)
        return residual + x


class ConformerBlock(nn.Module):
    """Macaron FFN + self-attn + depthwise conv + FFN with residuals."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        ffn_expansion: int,
        conv_kernel_size: int,
        use_relative_pos: bool,
        max_len: int,
    ) -> None:
        super().__init__()
        self.ffn1 = ConformerFeedForward(d_model, expansion=ffn_expansion, dropout=dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        if use_relative_pos:
            self.attn = RelativeMultiheadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout, max_len=max_len)
        else:
            self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.conv_module = ConformerConvModule(d_model=d_model, kernel_size=conv_kernel_size, dropout=dropout)
        self.ffn2 = ConformerFeedForward(d_model, expansion=ffn_expansion, dropout=dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        memory: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.ffn1(x)
        attn_input = self.attn_norm(x)
        if memory is None:
            attn_out = self.attn(attn_input, attn_input, attn_input, attn_mask=attention_mask)
        else:
            attn_out = self.attn(attn_input, memory, memory)
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]
        x = x + self.attn_dropout(attn_out)
        x = self.conv_module(x)
        x = self.ffn2(x)
        return self.final_norm(x)
