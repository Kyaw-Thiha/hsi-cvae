from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

LossFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]],
    tuple[torch.Tensor, dict[str, torch.Tensor]],
]


def _kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Compute per-sample KL divergence against a standard normal prior."""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def _gradient_loss_per_sample(
    recon: torch.Tensor,
    target: torch.Tensor,
    params: dict[str, Any],
) -> torch.Tensor:
    """
    Optional gradient-matching loss on spectral slope.
    Returns per-sample values with shape (B,).
    """
    grad_weight = float(params.get("grad_weight", 0.0))
    if grad_weight <= 0.0:
        return torch.zeros(recon.size(0), device=recon.device, dtype=recon.dtype)

    diff_order = max(1, int(params.get("grad_diff_order", 1)))
    grad_metric = str(params.get("grad_metric", "mse")).lower()

    # If spectra are too short for requested finite-difference order, disable term.
    if recon.size(1) <= diff_order:
        return torch.zeros(recon.size(0), device=recon.device, dtype=recon.dtype)

    recon_grad = torch.diff(recon, dim=1, n=diff_order)
    target_grad = torch.diff(target, dim=1, n=diff_order)

    if grad_metric == "l1":
        grad_loss = F.l1_loss(recon_grad, target_grad, reduction="none").mean(dim=1)
    else:
        grad_loss = F.mse_loss(recon_grad, target_grad, reduction="none").mean(dim=1)

    return grad_loss


def vanilla_vae_loss(
    target: torch.Tensor,
    recon: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    params: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Standard VAE objective with MSE reconstruction and unit KL weight."""
    params = params or {}
    reduction = params.get("reduction", "mean")
    grad_weight = float(params.get("grad_weight", 0.0))

    recon_loss = F.mse_loss(recon, target, reduction="none").mean(dim=1)
    grad_loss = _gradient_loss_per_sample(recon, target, params)
    kld = _kl_divergence(mu, logvar)
    loss = recon_loss + grad_weight * grad_loss + kld
    return (loss.mean() if reduction == "mean" else loss.sum()), {
        "recon_loss": recon_loss.mean(),
        "grad_loss": grad_loss.mean(),
        "kl_loss": kld.mean(),
    }


def beta_vae_loss(
    target: torch.Tensor,
    recon: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    params: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Beta-VAE objective with configurable reconstruction metric."""
    params = params or {}
    beta = params.get("beta", 4.0)
    recon_metric = params.get("recon", "mse")
    grad_weight = float(params.get("grad_weight", 0.0))

    if recon_metric == "l1":
        recon_loss = F.l1_loss(recon, target, reduction="none").mean(dim=1)
    else:
        recon_loss = F.mse_loss(recon, target, reduction="none").mean(dim=1)

    grad_loss = _gradient_loss_per_sample(recon, target, params)
    kld = _kl_divergence(mu, logvar)
    loss = recon_loss + grad_weight * grad_loss + beta * kld
    return loss.mean(), {
        "recon_loss": recon_loss.mean(),
        "grad_loss": grad_loss.mean(),
        "kl_loss": kld.mean(),
    }


def scale_vae_loss(
    target: torch.Tensor,
    recon: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    params: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Scale-VAE loss form; assumes reconstruction used scaled latent means upstream."""
    params = params or {}
    beta = float(params.get("beta", 1.0))
    recon_metric = params.get("recon", "mse")
    reduction = params.get("reduction", "mean")
    grad_weight = float(params.get("grad_weight", 0.0))

    if recon_metric == "l1":
        recon_loss = F.l1_loss(recon, target, reduction="none").mean(dim=1)
    else:
        recon_loss = F.mse_loss(recon, target, reduction="none").mean(dim=1)

    grad_loss = _gradient_loss_per_sample(recon, target, params)
    kld = _kl_divergence(mu, logvar)
    loss = recon_loss + grad_weight * grad_loss + beta * kld
    reduced = loss.mean() if reduction == "mean" else loss.sum()
    return reduced, {
        "recon_loss": recon_loss.mean(),
        "grad_loss": grad_loss.mean(),
        "kl_loss": kld.mean(),
    }


LOSS_REGISTRY: dict[str, LossFn] = {
    "vanilla": vanilla_vae_loss,
    "beta_vae": beta_vae_loss,
    "scale_vae": scale_vae_loss,
}

__all__ = [
    "LossFn",
    "beta_vae_loss",
    "vanilla_vae_loss",
    "scale_vae_loss",
    "LOSS_REGISTRY",
]
