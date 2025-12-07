from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
from torch import nn

from .attention import PositionalEncoding
from .conformer_block import ConformerBlock


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
        self.fusion_gate = nn.Linear(latent_dim + cond_dim, 1)

    def forward(
        self,
        z: torch.Tensor,
        cond: torch.Tensor,
        encoder_memory: Optional[torch.Tensor] = None,
        latent_blend: float | None = None,
    ) -> torch.Tensor:
        """
        Decode by cross-attending over an additive fusion of encoder tokens and latent-projected memory.

        During training we mix encoder memory with latent memory using a learned gate and a scheduled
        scalar so the decoder learns to interpret both. During sampling the encoder path disappears,
        so the decoder falls back entirely to the latent memory.
        """
        batch_size = z.size(0)
        decoder_queries = torch.zeros(batch_size, self.seq_len, self.d_model, device=z.device)
        decoder_queries = self.target_positional_encoding(decoder_queries)
        latent_memory = self._latent_to_memory(z, cond)
        if encoder_memory is None:
            # Sampling path: rely solely on the latent-projected memory.
            memory = latent_memory
        else:
            # Training path: blend encoder states with latent memory via the adaptive gate.
            memory = self.memory_positional_encoding(encoder_memory)
            gate_input = torch.cat([z, cond], dim=-1)
            adaptive_gate = torch.sigmoid(self.fusion_gate(gate_input)).view(batch_size, 1, 1)
            schedule = 0.0 if latent_blend is None else float(latent_blend)
            schedule = torch.tensor(schedule, device=z.device).view(1, 1, 1).clamp_(0.0, 1.0)
            latent_weight = schedule + (1.0 - schedule) * adaptive_gate
            memory = latent_weight * latent_memory + (1.0 - latent_weight) * memory
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
        latent_gate: LatentGateConfig | Mapping[str, Any] | None = None,
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

    def decode(self, z: torch.Tensor, cond: torch.Tensor, encoder_memory: Optional[torch.Tensor] = None) -> torch.Tensor:
        latent_blend = self._step_latent_gate(self.training) if encoder_memory is not None else None
        return self.decoder(z, cond, encoder_memory, latent_blend=latent_blend)

    def forward(self, spectrum: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar, memory = self.encode(spectrum, cond)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, cond, memory)
        return recon, mu, logvar
