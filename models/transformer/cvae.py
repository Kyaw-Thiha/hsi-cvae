from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn


@dataclass
class LatentGateConfig:
    start: float = 0.0
    end: float = 1.0
    warmup_steps: int = 0

    @classmethod
    def from_obj(cls, obj: LatentGateConfig | Mapping[str, Any] | None) -> LatentGateConfig:
        if obj is None:
            return cls()
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, Mapping):
            allowed = {"start", "end", "warmup_steps"}
            unknown = set(obj) - allowed
            if unknown:
                raise ValueError(f"latent_gate received unknown keys: {sorted(unknown)}")
            return cls(
                start=float(obj.get("start", cls.start)),
                end=float(obj.get("end", cls.end)),
                warmup_steps=int(obj.get("warmup_steps", cls.warmup_steps)),
            )
        raise TypeError(f"latent_gate must be LatentGateConfig or mapping, got {type(obj).__name__}")


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
        self.target_positional_encoding = PositionalEncoding(d_model, seq_len)
        self.memory_positional_encoding = PositionalEncoding(d_model, seq_len + 1)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.output_head = nn.Linear(d_model, 1)
        self.latent_token_mlp = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        z: torch.Tensor,
        cond: torch.Tensor,
        encoder_memory: torch.Tensor | None = None,
        latent_blend: float | None = None,
    ) -> torch.Tensor:
        """
        Decode by cross-attending to encoder memory augmented with a latent summary token.

        When encoder memory is provided (training), append a latent-derived token to the memory so the
        decoder sees a global summary. During sampling, rely entirely on the latent token.
        """
        batch_size = z.size(0)
        decoder_queries = torch.zeros(batch_size, self.seq_len, self.d_model, device=z.device)
        decoder_queries = self.target_positional_encoding(decoder_queries)

        latent_token = self.latent_to_token(z, cond)
        if encoder_memory is None:
            # No encoder states available (e.g., sampling) so we rely entirely on the latent token.
            memory = latent_token
        else:
            schedule = 0.0 if latent_blend is None else float(latent_blend)
            schedule = torch.tensor(schedule, device=z.device).view(1, 1, 1).clamp_(0.0, 1.0)
            latent_token = latent_token * schedule
            memory = torch.cat([encoder_memory, latent_token], dim=1)

        memory = self.memory_positional_encoding(memory)

        decoded = self.decoder(tgt=decoder_queries, memory=memory)
        recon = torch.sigmoid(self.output_head(decoded)).squeeze(-1)
        return recon

    def latent_to_token(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Project the latent sample plus condition into a single memory token."""
        latent_condition = torch.cat([z, cond], dim=-1)
        token = self.latent_token_mlp(latent_condition)
        return token.unsqueeze(1)


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
        latent_gate: LatentGateConfig | Mapping[str, Any] | None = None,
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
        gate_cfg = LatentGateConfig.from_obj(latent_gate)
        self.latent_gate_start = gate_cfg.start
        self.latent_gate_end = gate_cfg.end
        self.latent_gate_warmup_steps = gate_cfg.warmup_steps
        self.latent_gate_value: torch.Tensor
        self.latent_gate_step: torch.Tensor
        self.register_buffer("latent_gate_value", torch.tensor(self.latent_gate_start, dtype=torch.float32))
        self.register_buffer("latent_gate_step", torch.tensor(0, dtype=torch.long), persistent=False)

    def encode(self, spectrum: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.encoder(spectrum, cond)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _step_latent_gate(self, training: bool) -> float:
        """Advance the latent-memory blend schedule and return the current blend coefficient."""
        # If predicting, rely fully on latent vectors
        if not training:
            return self.latent_gate_end

        if self.latent_gate_warmup_steps <= 0:
            blend = self.latent_gate_end
        else:
            step = int(self.latent_gate_step.item())
            progress = min(step / self.latent_gate_warmup_steps, 1.0)
            blend = self.latent_gate_start + (self.latent_gate_end - self.latent_gate_start) * progress
            self.latent_gate_step += 1

        self.latent_gate_value = torch.tensor(blend, device=self.latent_gate_value.device)
        return blend

    def decode(self, z: torch.Tensor, cond: torch.Tensor, encoder_memory: torch.Tensor | None = None) -> torch.Tensor:
        latent_blend = self._step_latent_gate(self.training) if encoder_memory is not None else None
        return self.decoder(z, cond, encoder_memory, latent_blend=latent_blend)

    def forward(self, spectrum: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar, memory = self.encode(spectrum, cond)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, cond, memory)
        return recon, mu, logvar
