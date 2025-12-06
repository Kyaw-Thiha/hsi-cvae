from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding reused for encoder/decoder."""

    pe: torch.Tensor

    def __init__(self, dim: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, dim)

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

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, T, S)
        rel_bias = self._relative_bias(tgt_len, src_len, query.device)
        attn_scores = attn_scores + rel_bias

        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)  # (B, H, T, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, tgt_len, -1)
        return self.out_proj(attn_output)

    def _relative_bias(self, tgt_len: int, src_len: int, device: torch.device) -> torch.Tensor:
        """Compute relative position bias for each head."""

        range_q = torch.arange(tgt_len, device=device)
        range_k = torch.arange(src_len, device=device)
        rel_pos = range_q[:, None] - range_k[None, :]
        rel_pos = rel_pos.clamp(-self.max_rel_distance, self.max_rel_distance) + self.max_rel_distance
        bias = self.relative_bias(rel_pos).permute(2, 0, 1)  # (H, T, S)
        return bias.unsqueeze(0)  # (1, H, T, S)


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
        if isinstance(attn_out, tuple):  # MultiheadAttention returns tuple when using nn.MultiheadAttention
            attn_out = attn_out[0]
        x = x + self.attn_dropout(attn_out)
        x = self.conv_module(x)
        x = self.ffn2(x)
        return self.final_norm(x)


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        seq_len: int,
        cond_dim: int,
        latent_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        ffn_expansion: int,
        conv_kernel_size: int,
        use_relative_pos: bool,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(1, d_model)
        self.condition_projection = nn.Linear(cond_dim, d_model) if cond_dim > 0 else None
        self.positional_encoding = PositionalEncoding(d_model, seq_len + 1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                    ffn_expansion=ffn_expansion,
                    conv_kernel_size=conv_kernel_size,
                    use_relative_pos=use_relative_pos,
                    max_len=seq_len + 1,
                )
                for _ in range(n_layers)
            ]
        )
        self.mu_head = nn.Linear(d_model, latent_dim)
        self.logvar_head = nn.Linear(d_model, latent_dim)

    def forward(self, spectrum: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = spectrum.size(0)
        spectral_tokens = self.input_projection(spectrum.unsqueeze(-1))
        if self.condition_projection is not None:
            spectral_tokens = spectral_tokens + self.condition_projection(cond).unsqueeze(1)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, spectral_tokens], dim=1)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x)
        pooled_cls = x[:, 0]
        token_states = x[:, 1:]
        return self.mu_head(pooled_cls), self.logvar_head(pooled_cls), token_states


class ConformerDecoder(nn.Module):
    def __init__(
        self,
        seq_len: int,
        cond_dim: int,
        latent_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        ffn_expansion: int,
        conv_kernel_size: int,
        use_relative_pos: bool,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.latent_projection = nn.Linear(latent_dim + cond_dim, seq_len * d_model)
        self.target_positional_encoding = PositionalEncoding(d_model, seq_len)
        self.memory_positional_encoding = PositionalEncoding(d_model, seq_len)
        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                    ffn_expansion=ffn_expansion,
                    conv_kernel_size=conv_kernel_size,
                    use_relative_pos=use_relative_pos,
                    max_len=seq_len,
                )
                for _ in range(n_layers)
            ]
        )
        self.output_head = nn.Linear(d_model, 1)

    def forward(self, z: torch.Tensor, cond: torch.Tensor, encoder_memory: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = z.size(0)
        decoder_queries = torch.zeros(batch_size, self.seq_len, self.d_model, device=z.device)
        decoder_queries = self.target_positional_encoding(decoder_queries)
        if encoder_memory is None:
            memory = self._latent_to_memory(z, cond)
        else:
            memory = self.memory_positional_encoding(encoder_memory)
        x = decoder_queries
        for layer in self.layers:
            x = layer(x, memory=memory)
        recon = torch.sigmoid(self.output_head(x)).squeeze(-1)
        return recon

    def _latent_to_memory(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Project latent+condition to a decoder memory sequence for sampling."""

        batch_size = z.size(0)
        latent_condition = torch.cat([z, cond], dim=-1)
        memory = self.latent_projection(latent_condition).view(batch_size, self.seq_len, self.d_model)
        return self.memory_positional_encoding(memory)


class ConformerConditionalVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        latent_dim: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
        ffn_expansion: int = 4,
        conv_kernel_size: int = 17,
        use_relative_pos: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = ConformerEncoder(
            seq_len=input_dim,
            cond_dim=cond_dim,
            latent_dim=latent_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            ffn_expansion=ffn_expansion,
            conv_kernel_size=conv_kernel_size,
            use_relative_pos=use_relative_pos,
        )
        self.decoder = ConformerDecoder(
            seq_len=input_dim,
            cond_dim=cond_dim,
            latent_dim=latent_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            ffn_expansion=ffn_expansion,
            conv_kernel_size=conv_kernel_size,
            use_relative_pos=use_relative_pos,
        )

    def encode(self, spectrum: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.encoder(spectrum, cond)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, cond: torch.Tensor, encoder_memory: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.decoder(z, cond, encoder_memory)

    def forward(self, spectrum: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar, memory = self.encode(spectrum, cond)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, cond, memory)
        return recon, mu, logvar
