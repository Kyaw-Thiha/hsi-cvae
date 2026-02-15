from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _build_wavelength_grid(
    input_dim: int,
    start_nm: int,
    end_nm: int,
    step_nm: int,
) -> torch.Tensor:
    if step_nm <= 0:
        raise ValueError("wavelength_step_nm must be positive.")
    if end_nm < start_nm:
        raise ValueError("wavelength_end_nm must be >= wavelength_start_nm.")
    span = end_nm - start_nm
    if span % step_nm != 0:
        raise ValueError("(wavelength_end_nm - wavelength_start_nm) must be divisible by wavelength_step_nm.")
    n_wavelengths = (span // step_nm) + 1
    if n_wavelengths != input_dim:
        raise ValueError(f"Wavelength config implies {n_wavelengths} bands but input_dim is {input_dim}.")
    return torch.arange(start_nm, end_nm + 1, step_nm, dtype=torch.float32)


class RMSNorm(nn.Module):
    """Root-mean-square normalization with learned scale."""

    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


class FiLM(nn.Module):
    """2-layer FiLM MLP: cond -> gamma,beta; applies x * (1 + gamma) + beta."""

    def __init__(self, cond_dim: int, d_model: int) -> None:
        super().__init__()
        self.net = None
        if cond_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(cond_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, 2 * d_model),
            )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if self.net is None:
            return x
        gamma_beta = self.net(cond)  # (B, 2D)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        return x * (1.0 + gamma) + beta


class WavelengthAwarePositionalEncoding(nn.Module):
    """
    Encodes actual wavelength values (in nm) as learnable features.
    The network learns physically meaningful representations of spectral position.
    """

    wavelengths: torch.Tensor

    def __init__(self, d_model: int, wavelengths_nm: torch.Tensor):
        super().__init__()
        self.register_buffer("wavelengths", wavelengths_nm.float())  # (L,)
        self.wavelength_encoder = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )
        self.use_fourier = True
        if self.use_fourier:
            self.freq_bands = nn.Parameter(torch.randn(d_model // 4) * 0.01, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            (B, L, d_model) with wavelength encoding added
        """
        _, L, D = x.shape
        # Keep absolute spectral meaning via fixed min-max scaling to [0, 1].
        wl = self.wavelengths[:L]
        wl_min = self.wavelengths[0]
        wl_max = self.wavelengths[-1]
        wl = (wl - wl_min) / (wl_max - wl_min).clamp_min(1e-6)
        wl = wl.unsqueeze(-1)  # (L, 1)

        if self.use_fourier:
            phase = 2.0 * math.pi * wl * self.freq_bands.unsqueeze(0)  # (L, d_model//4)
            fourier = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)  # (L, d_model//2)
            learned = self.wavelength_encoder(wl)  # (L, d_model)
            if fourier.size(-1) < D:
                fourier = F.pad(fourier, (0, D - fourier.size(-1)))
            encoding = learned + fourier[:, :D]
        else:
            encoding = self.wavelength_encoder(wl)

        return x + encoding.unsqueeze(0)


class RepeatZTransformerBlock(nn.Module):
    """
    Exact order:
    1. RMSNorm
    2. Multi-Head Attention
    3. Dropout
    4. Residual
    5. RMSNorm
    6. FFN (GELU + 4x)
    7. Dropout
    8. Residual

    FiLM (Condition) is injected in both attention and FFN paths.
    """

    def __init__(self, d_model: int, n_heads: int, cond_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm_attn = RMSNorm(d_model)
        self.film_attn = FiLM(cond_dim=cond_dim, d_model=d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True)
        self.drop_attn = nn.Dropout(dropout)

        self.norm_ffn = RMSNorm(d_model)
        self.film_ffn = FiLM(cond_dim=cond_dim, d_model=d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            # nn.GELU(),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.drop_ffn = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm_attn(x)
        h = self.film_attn(h, cond)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop_attn(h)

        h = self.norm_ffn(x)
        h = self.film_ffn(h, cond)
        h = self.ffn(h)
        x = x + self.drop_ffn(h)
        return x


class RepeatZEncoder(nn.Module):
    """Encoder stack with CLS pooling for mu/logvar."""

    def __init__(
        self,
        seq_len: int,
        cond_dim: int,
        latent_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        wavelengths_nm: torch.Tensor,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_enc = WavelengthAwarePositionalEncoding(d_model=d_model, wavelengths_nm=wavelengths_nm)
        self.layers = nn.ModuleList([RepeatZTransformerBlock(d_model, n_heads, cond_dim, dropout) for _ in range(n_layers)])
        self.mu_head = nn.Linear(d_model, latent_dim)
        self.logvar_head = nn.Linear(d_model, latent_dim)

    def forward(self, spectrum: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = spectrum.size(0)
        x = self.input_proj(spectrum.unsqueeze(-1))  # (B, L, D)
        x = self.pos_enc(x)
        cls = (self.cls_token + self.cls_pos).expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, L+1, D)

        for layer in self.layers:
            x = layer(x, cond)

        cls_out = x[:, 0]
        return self.mu_head(cls_out), self.logvar_head(cls_out)


class RepeatZDecoder(nn.Module):
    """Decoder stack starting from repeated latent tokens plus wavelength-aware PE."""

    def __init__(
        self,
        seq_len: int,
        cond_dim: int,
        latent_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        wavelengths_nm: torch.Tensor,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.latent_to_token = nn.Linear(latent_dim, d_model)
        self.pos_enc = WavelengthAwarePositionalEncoding(d_model=d_model, wavelengths_nm=wavelengths_nm)
        self.layers = nn.ModuleList([RepeatZTransformerBlock(d_model, n_heads, cond_dim, dropout) for _ in range(n_layers)])

        self.out_head = nn.Linear(d_model, 1)

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.latent_to_token(z).unsqueeze(1).expand(-1, self.seq_len, -1)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, cond)

        return torch.sigmoid(self.out_head(x)).squeeze(-1)


class TransformerRepeatZConditionalVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        latent_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        encoder_layers: int = 6,
        decoder_layers: int = 4,
        dropout: float = 0.1,
        wavelength_start_nm: int = 400,
        wavelength_end_nm: int = 2490,
        wavelength_step_nm: int = 10,
    ) -> None:
        super().__init__()
        wavelengths_nm = _build_wavelength_grid(
            input_dim=input_dim,
            start_nm=wavelength_start_nm,
            end_nm=wavelength_end_nm,
            step_nm=wavelength_step_nm,
        )

        self.encoder = RepeatZEncoder(
            seq_len=input_dim,
            cond_dim=cond_dim,
            latent_dim=latent_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=encoder_layers,
            dropout=dropout,
            wavelengths_nm=wavelengths_nm,
        )
        self.decoder = RepeatZDecoder(
            seq_len=input_dim,
            cond_dim=cond_dim,
            latent_dim=latent_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=decoder_layers,
            dropout=dropout,
            wavelengths_nm=wavelengths_nm,
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
