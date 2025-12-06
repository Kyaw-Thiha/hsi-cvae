from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .attention import PositionalEncoding
from .conformer_block import ConformerBlock


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
