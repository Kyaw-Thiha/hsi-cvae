from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


def build_mlp(dims: Sequence[int], dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        layers.extend(
            [
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(inplace=True),
            ]
        )
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class ConditionalEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        latent_dim: int,
        hidden_dims: Sequence[int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dims = [input_dim + cond_dim, *hidden_dims]
        self.net = build_mlp(dims, dropout)

        last_dim = hidden_dims[-1]
        self.mu_head = nn.Linear(last_dim, latent_dim)
        self.logvar_head = nn.Linear(last_dim, latent_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(torch.cat([x, cond], dim=-1))
        return self.mu_head(h), self.logvar_head(h)


class ConditionalDecoder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        cond_dim: int,
        latent_dim: int,
        hidden_dims: Sequence[int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dims = [latent_dim + cond_dim, *hidden_dims, output_dim]
        self.net = build_mlp(dims[:-1], dropout)

        self.output_layer = nn.Linear(dims[-2], dims[-1])

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.net(torch.cat([z, cond], dim=-1))
        return torch.sigmoid(self.output_layer(h))


class ConditionalVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        latent_dim: int,
        hidden_dims: Sequence[int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = ConditionalEncoder(input_dim, cond_dim, latent_dim, hidden_dims, dropout)
        self.decoder = ConditionalDecoder(
            output_dim=input_dim,
            cond_dim=cond_dim,
            latent_dim=latent_dim,
            hidden_dims=list(reversed(hidden_dims)),
            dropout=dropout,
        )

    def encode(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x, cond)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.decoder(z, cond)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, cond)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, cond)
        return recon, mu, logvar
