from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

LossFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]],
    tuple[torch.Tensor, dict[str, torch.Tensor]],
]


def _build_wavelength_weight_mask(
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
    params: dict[str, Any],
) -> torch.Tensor:
    """
    Build per-band weights in [0, 1].

    Supported params:
      - masked_wavelength_ranges_nm: list[[start_nm, end_nm], ...]
      - masked_wavelength_weight: float, default 0.0
      - wavelength_start_nm: float, default 400.0
      - wavelength_step_nm: float, default 10.0
    """
    mask = torch.ones(seq_len, device=device, dtype=dtype)
    ranges = params.get("masked_wavelength_ranges_nm")
    if not ranges:
        return mask

    if not isinstance(ranges, (list, tuple)):
        raise TypeError("loss_params.masked_wavelength_ranges_nm must be a list of [start_nm, end_nm].")

    start_nm = float(params.get("wavelength_start_nm", 400.0))
    step_nm = float(params.get("wavelength_step_nm", 10.0))
    if step_nm <= 0.0:
        raise ValueError("loss_params.wavelength_step_nm must be > 0.")

    fill = float(params.get("masked_wavelength_weight", 0.0))
    fill = min(max(fill, 0.0), 1.0)

    for pair in ranges:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("Each masked wavelength range must be [start_nm, end_nm].")
        low_nm = float(pair[0])
        high_nm = float(pair[1])
        if high_nm < low_nm:
            low_nm, high_nm = high_nm, low_nm

        # Inclusive wavelength span [low_nm, high_nm].
        start_idx = max(0, int(math.ceil((low_nm - start_nm) / step_nm)))
        end_idx = min(seq_len - 1, int(math.floor((high_nm - start_nm) / step_nm)))
        if start_idx <= end_idx:
            mask[start_idx : end_idx + 1] = fill

    return mask


def _weighted_mean_per_sample(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """values: (B, L), weights: (L,) -> (B,)"""
    denom = weights.sum().clamp_min(1e-6)
    return (values * weights.unsqueeze(0)).sum(dim=1) / denom


def _build_gradient_position_mask(wavelength_mask: torch.Tensor, diff_order: int) -> torch.Tensor:
    """
    Convert an L-length wavelength mask into a mask for finite-diff outputs of length L-diff_order.
    A gradient position is weighted by the minimum weight across its diff stencil points.
    """
    out_len = wavelength_mask.size(0) - diff_order
    if out_len <= 0:
        return torch.zeros(0, device=wavelength_mask.device, dtype=wavelength_mask.dtype)
    stencil = torch.stack(
        [wavelength_mask[offset : offset + out_len] for offset in range(diff_order + 1)],
        dim=0,
    )
    return stencil.amin(dim=0)


def _kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Compute per-sample KL divergence against a standard normal prior."""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def _kl_divergence_with_free_bits(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    free_bits_total: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw KL and free-bits-adjusted KL objective per sample."""
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_raw = torch.sum(kl_per_dim, dim=1)

    free_bits_total = float(free_bits_total)
    if free_bits_total < 0.0:
        raise ValueError("loss_params.free_bits_total must be >= 0.0.")
    if free_bits_total == 0.0:
        return kl_raw, kl_raw

    latent_dims = max(int(kl_per_dim.size(1)), 1)
    free_bits_per_dim = free_bits_total / float(latent_dims)
    kl_objective = torch.clamp(kl_per_dim - free_bits_per_dim, min=0.0).sum(dim=1)
    return kl_raw, kl_objective


def _gradient_loss_per_sample(
    recon: torch.Tensor,
    target: torch.Tensor,
    params: dict[str, Any],
    wavelength_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Optional gradient-matching loss on spectral slope.
    Returns per-sample values with shape (B,).
    """
    grad_weight = float(params.get("grad_weight", 0.0))
    if grad_weight <= 0.0:
        return torch.zeros(recon.size(0), device=recon.device, dtype=recon.dtype)

    diff_orders_raw = params.get("grad_diff_orders")
    if diff_orders_raw is None:
        diff_orders = [max(1, int(params.get("grad_diff_order", 1)))]
    else:
        if not isinstance(diff_orders_raw, (list, tuple)):
            raise TypeError("loss_params.grad_diff_orders must be a list/tuple of positive integers.")
        diff_orders = []
        for item in diff_orders_raw:
            order = int(item)
            if order < 1:
                raise ValueError("loss_params.grad_diff_orders values must be >= 1.")
            if order not in diff_orders:
                diff_orders.append(order)
        if not diff_orders:
            return torch.zeros(recon.size(0), device=recon.device, dtype=recon.dtype)

    order_weights_raw = params.get("grad_order_weights")
    if order_weights_raw is None:
        order_weights = [1.0] * len(diff_orders)
    else:
        if not isinstance(order_weights_raw, (list, tuple)):
            raise TypeError("loss_params.grad_order_weights must be a list/tuple of floats.")
        if len(order_weights_raw) != len(diff_orders):
            raise ValueError("loss_params.grad_order_weights length must match grad_diff_orders length.")
        order_weights = [float(w) for w in order_weights_raw]

    grad_metric = str(params.get("grad_metric", "mse")).lower()
    combined_loss = torch.zeros(recon.size(0), device=recon.device, dtype=recon.dtype)
    used_any_order = False
    for diff_order, order_weight in zip(diff_orders, order_weights):
        if order_weight <= 0.0:
            continue
        # If spectra are too short for requested finite-difference order, skip this term.
        if recon.size(1) <= diff_order:
            continue

        recon_grad = torch.diff(recon, dim=1, n=diff_order)
        target_grad = torch.diff(target, dim=1, n=diff_order)

        if grad_metric == "l1":
            grad_loss_per_pos = F.l1_loss(recon_grad, target_grad, reduction="none")
        else:
            grad_loss_per_pos = F.mse_loss(recon_grad, target_grad, reduction="none")

        if wavelength_mask is None:
            grad_loss = grad_loss_per_pos.mean(dim=1)
        else:
            grad_mask = _build_gradient_position_mask(wavelength_mask, diff_order).to(
                device=grad_loss_per_pos.device,
                dtype=grad_loss_per_pos.dtype,
            )
            if grad_mask.numel() == 0 or float(grad_mask.sum().item()) <= 0.0:
                continue
            grad_loss = _weighted_mean_per_sample(grad_loss_per_pos, grad_mask)

        combined_loss = combined_loss + (float(order_weight) * grad_loss)
        used_any_order = True

    if not used_any_order:
        return torch.zeros(recon.size(0), device=recon.device, dtype=recon.dtype)
    return combined_loss


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

    wavelength_mask = _build_wavelength_weight_mask(
        seq_len=recon.size(1),
        device=recon.device,
        dtype=recon.dtype,
        params=params,
    )

    recon_loss_per_wavelength = F.mse_loss(recon, target, reduction="none")
    recon_loss = _weighted_mean_per_sample(recon_loss_per_wavelength, wavelength_mask)
    grad_loss = _gradient_loss_per_sample(recon, target, params, wavelength_mask=wavelength_mask)
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
    beta = float(params.get("beta", 4.0))
    recon_metric = params.get("recon", "mse")
    grad_weight = float(params.get("grad_weight", 0.0))
    free_bits_total = float(params.get("free_bits_total", 0.0))

    wavelength_mask = _build_wavelength_weight_mask(
        seq_len=recon.size(1),
        device=recon.device,
        dtype=recon.dtype,
        params=params,
    )

    if recon_metric == "l1":
        recon_loss_per_wavelength = F.l1_loss(recon, target, reduction="none")
    else:
        recon_loss_per_wavelength = F.mse_loss(recon, target, reduction="none")
    recon_loss = _weighted_mean_per_sample(recon_loss_per_wavelength, wavelength_mask)

    grad_loss = _gradient_loss_per_sample(recon, target, params, wavelength_mask=wavelength_mask)
    kld_raw, kld_objective = _kl_divergence_with_free_bits(mu, logvar, free_bits_total=free_bits_total)
    loss = recon_loss + grad_weight * grad_loss + beta * kld_objective
    return loss.mean(), {
        "recon_loss": recon_loss.mean(),
        "grad_loss": grad_loss.mean(),
        "kl_loss": kld_raw.mean(),
        "kl_objective_loss": kld_objective.mean(),
    }


def scale_vae_loss(
    target: torch.Tensor,
    recon: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    params: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Scale-VAE objective; assumes reconstruction used scaled latent means upstream."""
    params = params or {}
    recon_metric = params.get("recon", "mse")
    reduction = params.get("reduction", "mean")

    wavelength_mask = _build_wavelength_weight_mask(
        seq_len=recon.size(1),
        device=recon.device,
        dtype=recon.dtype,
        params=params,
    )

    if recon_metric == "l1":
        recon_loss_per_wavelength = F.l1_loss(recon, target, reduction="none")
    else:
        recon_loss_per_wavelength = F.mse_loss(recon, target, reduction="none")
    recon_loss = _weighted_mean_per_sample(recon_loss_per_wavelength, wavelength_mask)

    kld = _kl_divergence(mu, logvar)
    loss = recon_loss + kld
    reduced = loss.mean() if reduction == "mean" else loss.sum()
    return reduced, {
        "recon_loss": recon_loss.mean(),
        "grad_loss": torch.zeros((), device=recon.device, dtype=recon.dtype),
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
