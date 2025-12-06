from __future__ import annotations


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


class TransformerEncoder(nn.Module):
    """Self-attention encoder that pools a CLS token to obtain mu/logvar."""

    def __init__(
        self,
        seq_len: int,
        cond_dim: int,
        latent_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.input_projection = nn.Linear(1, d_model)
        self.condition_projection = nn.Linear(cond_dim, d_model) if cond_dim > 0 else None
        self.positional_encoding = PositionalEncoding(d_model, seq_len + 1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.mu_head = nn.Linear(d_model, latent_dim)
        self.logvar_head = nn.Linear(d_model, latent_dim)

    def forward(self, spectrum: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = spectrum.size(0)
        spectral_tokens = self.input_projection(spectrum.unsqueeze(-1))
        if self.condition_projection is not None:
            spectral_tokens = spectral_tokens + self.condition_projection(cond).unsqueeze(1)

        cls_token = self.cls_token.expand(batch_size, -1, -1)
        encoder_input = torch.cat([cls_token, spectral_tokens], dim=1)
        encoder_input = self.positional_encoding(encoder_input)
        encoder_output = self.encoder(encoder_input)
        pooled_cls = encoder_output[:, 0]
        token_states = encoder_output[:, 1:]
        return self.mu_head(pooled_cls), self.logvar_head(pooled_cls), token_states


class TransformerDecoder(nn.Module):
    """Decoder attends to encoder memory or latent-derived memory to reconstruct the spectrum."""

    def __init__(
        self,
        seq_len: int,
        cond_dim: int,
        latent_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.latent_projection = nn.Linear(latent_dim + cond_dim, seq_len * d_model)
        self.target_positional_encoding = PositionalEncoding(d_model, seq_len)
        self.memory_positional_encoding = PositionalEncoding(d_model, seq_len)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.output_head = nn.Linear(d_model, 1)

    def forward(self, z: torch.Tensor, cond: torch.Tensor, encoder_memory: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = z.size(0)
        decoder_queries = torch.zeros(batch_size, self.seq_len, self.d_model, device=z.device)
        decoder_queries = self.target_positional_encoding(decoder_queries)

        if encoder_memory is None:
            memory = self.latent_to_memory(z, cond)
        else:
            memory = self.memory_positional_encoding(encoder_memory)

        decoded = self.decoder(tgt=decoder_queries, memory=memory)
        recon = torch.sigmoid(self.output_head(decoded)).squeeze(-1)
        return recon

    def latent_to_memory(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Synthesize a decoder memory sequence purely from the latent sample plus condition.

        Projects the concatenated vector to shape (batch, seq_len, d_model),
        injects positional encoding, and returns it so the decoder’s cross-attention
        can operate even when no encoder token states are available (e.g. during sampling).
        """
        batch_size = z.size(0)
        latent_condition = torch.cat([z, cond], dim=-1)
        memory = self.latent_projection(latent_condition).view(batch_size, self.seq_len, self.d_model)
        return self.memory_positional_encoding(memory)


class TransformerConditionalVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        latent_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = TransformerEncoder(
            seq_len=input_dim,
            cond_dim=cond_dim,
            latent_dim=latent_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.decoder = TransformerDecoder(
            seq_len=input_dim,
            cond_dim=cond_dim,
            latent_dim=latent_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )

    def encode(self, spectrum: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.encoder(spectrum, cond)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, cond: torch.Tensor, encoder_memory: torch.Tensor | None = None) -> torch.Tensor:
        return self.decoder(z, cond, encoder_memory)

    def forward(self, spectrum: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar, memory = self.encode(spectrum, cond)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, cond, memory)
        return recon, mu, logvar
