from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dropout: float = 0.0) -> None:
        super().__init__()
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple wrapper
        return self.block(x)


class ConditionInjector(nn.Module):
    """Projects conditioning vectors into channel maps for convolutional processing."""

    def __init__(self, cond_dim: int, seq_len: int, channels: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.channels = channels
        self.proj = nn.Linear(cond_dim, seq_len * channels) if (cond_dim > 0 and channels > 0) else None

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        if self.channels == 0 or cond.size(1) == 0 or self.proj is None:
            return cond.new_zeros(cond.size(0), self.channels, self.seq_len)
        out = self.proj(cond)
        return out.view(cond.size(0), self.channels, self.seq_len)


class ConvConditionalEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        latent_dim: int,
        conv_channels: Sequence[int],
        cond_channels: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not conv_channels:
            raise ValueError("conv_channels must contain at least one entry.")
        self.seq_len = input_dim
        self.cond_injector = ConditionInjector(cond_dim, input_dim, cond_channels)

        channels = [1 + cond_channels, *conv_channels]
        blocks = [ConvBlock(in_c, out_c, dropout=dropout) for in_c, out_c in zip(channels[:-1], channels[1:])]
        self.net = nn.Sequential(*blocks)

        last_dim = conv_channels[-1] * input_dim
        self.mu_head = nn.Linear(last_dim, latent_dim)
        self.logvar_head = nn.Linear(last_dim, latent_dim)

    def forward(self, spectrum: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = spectrum.unsqueeze(1)  # (B, 1, L)
        cond_map = self.cond_injector(cond)
        h = self.net(torch.cat([x, cond_map], dim=1)).flatten(1)
        return self.mu_head(h), self.logvar_head(h)


class ConvConditionalDecoder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        cond_dim: int,
        latent_dim: int,
        conv_channels: Sequence[int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not conv_channels:
            raise ValueError("conv_channels must contain at least one entry.")
        self.seq_len = output_dim

        first_hidden = conv_channels[0]
        self.fc = nn.Linear(latent_dim + cond_dim, first_hidden * self.seq_len)

        blocks = [ConvBlock(in_c, out_c, dropout=dropout) for in_c, out_c in zip(conv_channels[:-1], conv_channels[1:])]
        self.net = nn.Sequential(*blocks) if blocks else nn.Identity()
        self.head = nn.Conv1d(conv_channels[-1], 1, kernel_size=1)

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.fc(torch.cat([z, cond], dim=-1))
        h = h.view(z.size(0), -1, self.seq_len)
        h = self.net(h)
        return torch.sigmoid(self.head(h)).squeeze(1)


class ConvConditionalVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        latent_dim: int,
        conv_channels: Sequence[int],
        dropout: float = 0.0,
        cond_channels: int = 1,
    ) -> None:
        super().__init__()
        self.encoder = ConvConditionalEncoder(
            input_dim=input_dim,
            cond_dim=cond_dim,
            latent_dim=latent_dim,
            conv_channels=conv_channels,
            cond_channels=cond_channels,
            dropout=dropout,
        )
        decoder_channels = list(reversed(conv_channels))
        self.decoder = ConvConditionalDecoder(
            output_dim=input_dim,
            cond_dim=cond_dim,
            latent_dim=latent_dim,
            conv_channels=decoder_channels,
            dropout=dropout,
        )

    def encode(self, spectrum: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(spectrum, cond)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.decoder(z, cond)

    def forward(self, spectrum: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(spectrum, cond)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, cond)
        return recon, mu, logvar
