from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root-mean-square normalization with learned scale."""

    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


class GatedFiLM(nn.Module):
    """FiLM modulation with a weak learnable scalar gate."""

    def __init__(self, cond_dim: int, d_model: int, gate_init: float = 0.1) -> None:
        super().__init__()
        self.gamma = nn.Linear(cond_dim, d_model)
        self.beta = nn.Linear(cond_dim, d_model)
        init_gate = torch.tensor(float(gate_init), dtype=torch.float32).clamp(1e-6, 1.0 - 1e-6)
        init_logit = torch.log(init_gate / (1.0 - init_gate))
        self.gate = nn.Parameter(init_logit)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma(cond).unsqueeze(1)
        beta = self.beta(cond).unsqueeze(1)
        gate = torch.sigmoid(self.gate).to(device=x.device, dtype=x.dtype)
        return x * (1.0 + gate * gamma) + gate * beta


class EncoderTransformerBlock(nn.Module):
    """Encoder block with self-attn, condition cross-attn, and FFN."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_attn = RMSNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True)
        self.drop_attn = nn.Dropout(dropout)

        self.norm_cross = RMSNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True)
        self.drop_cross = nn.Dropout(dropout)

        self.norm_ffn = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.drop_ffn = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cond_token: torch.Tensor) -> torch.Tensor:
        h = self.norm_attn(x)
        h, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + self.drop_attn(h)

        h = self.norm_cross(x)
        h, _ = self.cross_attn(h, cond_token, cond_token, need_weights=False)
        x = x + self.drop_cross(h)

        h = self.norm_ffn(x)
        h = self.ffn(h)
        x = x + self.drop_ffn(h)
        return x


class DecoderTransformerBlock(nn.Module):
    """Decoder block with self-attn, gated FiLM, and FFN."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        cond_dim: int,
        dropout: float,
        film_gate_init: float,
        use_film: bool,
    ) -> None:
        super().__init__()
        self.norm_attn = RMSNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True)
        self.drop_attn = nn.Dropout(dropout)

        self.film: GatedFiLM | None = None
        if use_film:
            self.film = GatedFiLM(cond_dim=cond_dim, d_model=d_model, gate_init=film_gate_init)

        self.norm_ffn = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.drop_ffn = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm_attn(x)
        h, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + self.drop_attn(h)

        if self.film is not None:
            x = self.film(x, cond)

        h = self.norm_ffn(x)
        h = self.ffn(h)
        x = x + self.drop_ffn(h)
        return x


class DualPathEncoder(nn.Module):
    """Hierarchical encoder producing latent Gaussian parameters."""

    def __init__(
        self,
        seq_len: int,
        latent_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.input_projection = nn.Linear(1, d_model)
        self.layers = nn.ModuleList(
            [EncoderTransformerBlock(d_model=d_model, n_heads=n_heads, dropout=dropout) for _ in range(n_layers)]
        )
        self.mu_head = nn.Linear(d_model, latent_dim)
        self.logvar_head = nn.Linear(d_model, latent_dim)

    def forward(self, spectrum: torch.Tensor, cond_embed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = spectrum.unsqueeze(-1)
        x = self.input_projection(x)
        cond_token = cond_embed.unsqueeze(1)

        for layer in self.layers:
            x = layer(x, cond_token)

        pooled = x.mean(dim=1)
        return self.mu_head(pooled), self.logvar_head(pooled)


class DualPathDecoder(nn.Module):
    """Dual-path decoder with global condition path and local latent path."""

    pos_encoding: torch.Tensor
    latent_fuse_weight: torch.Tensor

    def __init__(
        self,
        seq_len: int,
        latent_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        latent_fuse_weight: float,
        latent_fuse_weight_min: float,
        film_gate_init: float,
        latent_fuse_weight_learnable: bool,
        global_path_dropout: float,
        global_path_hidden_dim: int,
        global_path_warmup_hold_epochs: int,
        global_path_warmup_ramp_epochs: int,
        decoder_logit_gain: float,
        decoder_use_film: bool,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        min_weight = float(latent_fuse_weight_min)
        if not 0.0 <= min_weight < 1.0:
            raise ValueError("latent_fuse_weight_min must be in [0.0, 1.0).")
        self.latent_fuse_weight_min = min_weight

        raw_weight = (float(latent_fuse_weight) - min_weight) / max(1.0 - min_weight, 1e-6)
        init_weight = torch.tensor(raw_weight, dtype=torch.float32).clamp(1e-6, 1.0 - 1e-6)
        init_logit = torch.log(init_weight / (1.0 - init_weight))
        if latent_fuse_weight_learnable:
            self.latent_fuse_weight = nn.Parameter(init_logit.clone())
        else:
            self.register_buffer("latent_fuse_weight", init_logit.clone())
        self.global_path_warmup_hold_epochs = max(int(global_path_warmup_hold_epochs), 0)
        self.global_path_warmup_ramp_epochs = max(int(global_path_warmup_ramp_epochs), 0)
        self._warmup_epoch = 0
        gain = float(decoder_logit_gain)
        if gain <= 0.0:
            raise ValueError("decoder_logit_gain must be > 0.0.")
        self.decoder_logit_gain = gain
        hidden_dim = max(int(global_path_hidden_dim), 1)

        self.global_path = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(global_path_dropout),
            nn.Linear(hidden_dim, seq_len),
        )

        self.latent_projection = nn.Linear(latent_dim, d_model)
        self.register_buffer("pos_encoding", self._create_positional_encoding(seq_len=seq_len, d_model=d_model), persistent=False)
        self.layers = nn.ModuleList(
            [
                DecoderTransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    cond_dim=d_model,
                    dropout=dropout,
                    film_gate_init=film_gate_init,
                    use_film=decoder_use_film,
                )
                for _ in range(n_layers)
            ]
        )
        self.local_out = nn.Linear(d_model, 1)

    @staticmethod
    def _create_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(seq_len, d_model, dtype=torch.float32)
        position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / float(d_model))
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        return pe

    def set_warmup_epoch(self, epoch: int) -> None:
        self._warmup_epoch = max(int(epoch), 0)

    def _global_path_scale(self, apply_schedule: bool) -> float:
        if not apply_schedule:
            return 1.0
        hold = self.global_path_warmup_hold_epochs
        ramp = self.global_path_warmup_ramp_epochs
        epoch = self._warmup_epoch
        if epoch < hold:
            return 0.0
        if ramp <= 0:
            return 1.0
        progress = (epoch - hold + 1) / float(ramp)
        return float(min(max(progress, 0.0), 1.0))

    def effective_local_weight(self) -> float:
        raw_weight = float(torch.sigmoid(self.latent_fuse_weight.detach()).item())
        return self.latent_fuse_weight_min + (1.0 - self.latent_fuse_weight_min) * raw_weight

    def effective_global_scale(self) -> float:
        return self._global_path_scale(apply_schedule=True)

    def forward(self, z: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        global_spectrum = self.global_path(cond_embed)

        local_tokens = self.latent_projection(z).unsqueeze(1).expand(-1, self.seq_len, -1)
        pos_encoding = self.pos_encoding.to(device=local_tokens.device, dtype=local_tokens.dtype)
        local_tokens = local_tokens + torch.unsqueeze(pos_encoding, dim=0)
        for layer in self.layers:
            local_tokens = layer(local_tokens, cond_embed)
        local_spectrum = self.local_out(local_tokens).squeeze(-1)

        raw_weight = torch.sigmoid(self.latent_fuse_weight).to(device=local_spectrum.device, dtype=local_spectrum.dtype)
        weight = self.latent_fuse_weight_min + (1.0 - self.latent_fuse_weight_min) * raw_weight
        global_scale = torch.tensor(
            self._global_path_scale(apply_schedule=self.training),
            device=local_spectrum.device,
            dtype=local_spectrum.dtype,
        )
        fused = global_scale * (1.0 - weight) * global_spectrum + weight * local_spectrum
        fused = self.decoder_logit_gain * fused
        return torch.sigmoid(fused)


class DualPathTransformerConditionalVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        latent_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        encoder_layers: int = 6,
        decoder_layers: int = 2,
        dropout: float = 0.0,
        latent_fuse_weight: float = 0.7,
        latent_fuse_weight_min: float = 0.3,
        gated_film_init: float = 0.1,
        latent_fuse_weight_learnable: bool = True,
        global_path_dropout: float = 0.2,
        global_path_hidden_dim: int = 256,
        global_path_warmup_hold_epochs: int = 5,
        global_path_warmup_ramp_epochs: int = 10,
        decoder_logit_gain: float = 1.0,
        decoder_use_film: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.condition_embedding = (
            nn.Sequential(
                nn.Linear(cond_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            if cond_dim > 0
            else None
        )

        self.encoder = DualPathEncoder(
            seq_len=input_dim,
            latent_dim=latent_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=encoder_layers,
            dropout=dropout,
        )
        self.decoder = DualPathDecoder(
            seq_len=input_dim,
            latent_dim=latent_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=decoder_layers,
            dropout=dropout,
            latent_fuse_weight=latent_fuse_weight,
            latent_fuse_weight_min=latent_fuse_weight_min,
            film_gate_init=gated_film_init,
            latent_fuse_weight_learnable=latent_fuse_weight_learnable,
            global_path_dropout=global_path_dropout,
            global_path_hidden_dim=global_path_hidden_dim,
            global_path_warmup_hold_epochs=global_path_warmup_hold_epochs,
            global_path_warmup_ramp_epochs=global_path_warmup_ramp_epochs,
            decoder_logit_gain=decoder_logit_gain,
            decoder_use_film=decoder_use_film,
        )

    def set_global_path_warmup_epoch(self, epoch: int) -> None:
        self.decoder.set_warmup_epoch(epoch)

    def local_fuse_effective_weight(self) -> float:
        return self.decoder.effective_local_weight()

    def global_path_effective_scale(self) -> float:
        return self.decoder.effective_global_scale()

    def _encode_condition(self, cond: torch.Tensor) -> torch.Tensor:
        if self.condition_embedding is None:
            return torch.zeros(cond.size(0), self.d_model, device=cond.device, dtype=cond.dtype)
        return self.condition_embedding(cond)

    def encode(self, spectrum: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cond_embed = self._encode_condition(cond)
        return self.encoder(spectrum, cond_embed)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        cond_embed = self._encode_condition(cond)
        return self.decoder(z, cond_embed)

    def forward(self, spectrum: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(spectrum, cond)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, cond)
        return recon, mu, logvar
