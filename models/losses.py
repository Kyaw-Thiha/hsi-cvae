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
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def vanilla_vae_loss(
    target: torch.Tensor,
    recon: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    params: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    params = params or {}
    reduction = params.get("reduction", "mean")
    recon_loss = F.mse_loss(recon, target, reduction="none").mean(dim=1)
    kld = _kl_divergence(mu, logvar)
    loss = recon_loss + kld
    return (loss.mean() if reduction == "mean" else loss.sum()), {
        "recon_loss": recon_loss.mean(),
        "kl_loss": kld.mean(),
    }


def beta_vae_loss(
    target: torch.Tensor,
    recon: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    params: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    params = params or {}
    beta = params.get("beta", 4.0)
    recon_metric = params.get("recon", "mse")

    if recon_metric == "l1":
        recon_loss = F.l1_loss(recon, target, reduction="none").mean(dim=1)
    else:
        recon_loss = F.mse_loss(recon, target, reduction="none").mean(dim=1)

    kld = _kl_divergence(mu, logvar)
    loss = recon_loss + beta * kld
    return loss.mean(), {
        "recon_loss": recon_loss.mean(),
        "kl_loss": kld.mean(),
    }


LOSS_REGISTRY: dict[str, LossFn] = {
    "vanilla": vanilla_vae_loss,
    "beta_vae": beta_vae_loss,
}

__all__ = [
    "LossFn",
    "beta_vae_loss",
    "vanilla_vae_loss",
    "LOSS_REGISTRY",
]
