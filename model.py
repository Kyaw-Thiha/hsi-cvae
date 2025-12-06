from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Union

import lightning as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import optim

from models.mlp.cvae import ConditionalVAE
from models.cnn.cvae import ConvConditionalVAE
from models.transformer.cvae import TransformerConditionalVAE
from models.conformer.cvae import ConformerConditionalVAE
from models.losses import LOSS_REGISTRY


@dataclass
class LossParams:
    beta: float = 4.0
    recon: str = "mse"


@dataclass
class SchedulerParams:
    name: str = "cosine"
    T_max: int = 200


class CVAELightningModule(L.LightningModule):
    """LightningModule wrapping the Conditional VAE for training/eval."""

    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        latent_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.0,
        architecture: str = "mlp",
        cnn_params: Optional[Mapping[str, Any]] = None,
        transformer_params: Optional[Mapping[str, Any]] = None,
        conformer_params: Optional[Mapping[str, Any]] = None,
        loss_name: str = "vanilla",
        loss_params: Optional[LossParams] = None,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        scheduler_cfg: Optional[SchedulerParams] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["loss_params", "scheduler_cfg"])

        # Model Architecture
        self.architecture = architecture.lower()
        self.cnn_params = dict(cnn_params or {})
        self.transformer_params = dict(transformer_params or {})
        self.conformer_params = dict(conformer_params or {})
        self.model: Union[ConditionalVAE, ConvConditionalVAE, TransformerConditionalVAE, ConformerConditionalVAE] = (
            self._build_model(
                input_dim=input_dim,
                condition_dim=condition_dim,
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                dropout=dropout,
            )
        )

        # Loss Function
        if loss_name not in LOSS_REGISTRY:
            raise ValueError(f"Unsupported loss: {loss_name}")
        self.loss_fn = LOSS_REGISTRY[loss_name]
        if loss_params is None:
            self.loss_params: dict[str, Any] = {}
        elif isinstance(loss_params, LossParams):
            self.loss_params = asdict(loss_params)
        elif isinstance(loss_params, Mapping):
            # For passing in through config.yaml
            self.loss_params = dict(loss_params)
        else:
            raise TypeError("loss_params must be a LossParams dataclass or mapping.")

        # Schedular Config
        if scheduler_cfg is None:
            self.scheduler_cfg: Optional[dict[str, Any]] = None
        elif isinstance(scheduler_cfg, SchedulerParams):
            self.scheduler_cfg = asdict(scheduler_cfg)
        elif isinstance(scheduler_cfg, Mapping):
            # For passing in through config.yaml
            self.scheduler_cfg = dict(scheduler_cfg)
        else:
            raise TypeError("scheduler_cfg must be a SchedulerParams dataclass or mapping.")

        # Model Hyperparams
        self.learning_rate = lr
        self.weight_decay = weight_decay
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.num_classes = condition_dim

    def _build_model(
        self,
        input_dim: int,
        condition_dim: int,
        latent_dim: int,
        hidden_dims: list[int],
        dropout: float,
    ) -> Union[ConditionalVAE, ConvConditionalVAE, TransformerConditionalVAE, ConformerConditionalVAE]:
        if self.architecture == "mlp":
            return ConditionalVAE(
                input_dim=input_dim,
                cond_dim=condition_dim,
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                dropout=dropout,
            )
        if self.architecture == "cnn":
            conv_channels = self.cnn_params.get("conv_channels")
            if not conv_channels:
                raise ValueError("cnn_params.conv_channels must be provided for CNN architecture")
            return ConvConditionalVAE(
                input_dim=input_dim,
                cond_dim=condition_dim,
                latent_dim=latent_dim,
                conv_channels=conv_channels,
                dropout=self.cnn_params.get("dropout", dropout),
                cond_channels=self.cnn_params.get("cond_channels", 1),
            )
        if self.architecture == "transformer":
            transformer_defaults: dict[str, Any] = {
                "d_model": 128,
                "n_heads": 4,
                "n_layers": 4,
                "dropout": dropout,
            }
            transformer_defaults.update(self.transformer_params)
            return TransformerConditionalVAE(
                input_dim=input_dim,
                cond_dim=condition_dim,
                latent_dim=latent_dim,
                **transformer_defaults,
            )
        if self.architecture == "conformer":
            conformer_defaults: dict[str, Any] = {
                "d_model": 256,
                "n_heads": 4,
                "n_layers": 4,
                "dropout": dropout,
                "ffn_expansion": 4,
                "conv_kernel_size": 17,
                "use_relative_pos": True,
            }
            conformer_defaults.update(self.conformer_params)
            return ConformerConditionalVAE(
                input_dim=input_dim,
                cond_dim=condition_dim,
                latent_dim=latent_dim,
                **conformer_defaults,
            )
        raise ValueError(f"Unsupported architecture: {self.architecture}")

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

    def predict_step(
        self,
        batch: dict[str, torch.Tensor],
        _: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        conditions = batch["condition"].to(self.device)
        n = conditions.size(0)
        z = torch.randn(n, self.latent_dim, device=conditions.device)
        spectra = self.model.decode(z, conditions)
        spectra = spectra * 2.0 - 1.0
        return spectra.view(n, 1, 1, -1)

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

    def labels_to_conditions(self, labels: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Map integer labels to conditioning vectors (one-hot rows) for sampling."""
        if self.condition_dim == 0:
            return torch.empty(labels.shape[0], 0, device=device)
        idx = labels.clamp(0, self.num_classes - 1).long()
        prototypes = torch.eye(self.num_classes, device=device, dtype=torch.float32)
        return prototypes[idx]

    @torch.no_grad()
    def sample(
        self,
        n: int,
        y: Optional[torch.Tensor] = None,
        conditions: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        temperature: float = 1.0,
        **_: Any,
    ) -> torch.Tensor:
        target_device = device or self.device
        if not isinstance(target_device, torch.device):
            target_device = torch.device(str(target_device))
        if conditions is not None:
            cond = conditions.to(target_device)
            if cond.ndim != 2 or cond.size(1) != self.condition_dim:
                raise ValueError(f"conditions must have shape (n, {self.condition_dim}) but received {tuple(cond.shape)}")
            n_samples = cond.size(0)
        elif y is not None:
            cond = self.labels_to_conditions(y.to(target_device), target_device)
            n_samples = cond.size(0)
        else:
            n_samples = n
            cond = torch.rand(n_samples, self.condition_dim, device=target_device)
            if self.condition_dim > 0:
                cond = cond / cond.sum(dim=1, keepdim=True).clamp_min(1e-6)
        z = torch.randn(n_samples, self.latent_dim, device=target_device)
        if temperature != 1.0:
            z = z * temperature
        recon = self.model.decode(z, cond)
        recon = recon * 2.0 - 1.0
        return recon.view(n_samples, 1, 1, -1)
