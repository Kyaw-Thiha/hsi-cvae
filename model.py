from __future__ import annotations

from typing import Any, Optional

import lightning as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import optim

from models.base.cvae import ConditionalVAE
from models.base.losses import LOSS_REGISTRY


class CVAELightningModule(L.LightningModule):
    """LightningModule wrapping the Conditional VAE for training/eval."""

    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        latent_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.0,
        loss_name: str = "vanilla",
        loss_params: Optional[dict[str, Any]] = None,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        scheduler_cfg: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["loss_params", "scheduler_cfg"])

        self.model = ConditionalVAE(
            input_dim=input_dim,
            cond_dim=condition_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        if loss_name not in LOSS_REGISTRY:
            raise ValueError(f"Unsupported loss: {loss_name}")
        self.loss_fn = LOSS_REGISTRY[loss_name]
        self.loss_params = loss_params or {}
        self.scheduler_cfg = scheduler_cfg
        self.learning_rate = lr
        self.weight_decay = weight_decay

    def forward(self, spectrum: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        recon, _, _ = self.model(spectrum, condition)
        return recon

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        spectra, cond = batch["spectrum"], batch["condition"]
        recon, mu, logvar = self.model(spectra, cond)
        loss, metrics = self.loss_fn(spectra, recon, mu, logvar, self.loss_params)
        self.log(f"{stage}_loss", loss, prog_bar=True)
        for name, value in metrics.items():
            self.log(f"{stage}_{name}", value, prog_bar=False, logger=True)
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], _) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], _) -> None:
        self._shared_step(batch, "val")

    def test_step(self, batch: dict[str, torch.Tensor], _) -> None:
        self._shared_step(batch, "test")

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        if self.scheduler_cfg is None:
            return optimizer

        sched_name = self.scheduler_cfg.get("name", "cosine")
        if sched_name == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.scheduler_cfg.get("T_max", 200),
            )
        else:
            raise ValueError(f"Unsupported scheduler: {sched_name}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
