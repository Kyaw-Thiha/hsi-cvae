from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Union, cast

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
    """Loss hyperparameters, including Scale-VAE scaling controls."""
    beta: float = 4.0
    recon: str = "mse"
    des_std: float = 1.0
    f_epo: int = 10
    scale_eps: float = 1e-6


@dataclass
class SchedulerParams:
    name: str = "cosine"
    T_max: int = 200


@dataclass
class WavelengthParams:
    """Model-side wavelength metadata for spectral positional features."""
    start_nm: int = 400
    end_nm: int = 2490
    step_nm: int = 10


class CVAELightningModule(L.LightningModule):
    """LightningModule wrapping the Conditional VAE for training/eval."""
    scale_f_bar: torch.Tensor
    scale_f_accum: torch.Tensor
    scale_f_count: torch.Tensor
    wavelengths_nm: torch.Tensor

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
        wavelength_params: Optional[WavelengthParams] = None,
        temperature: float = 1.0,
        guidance_scale: float = 1.0,
        condition_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["loss_params", "scheduler_cfg", "wavelength_params"])

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
        self.loss_name = loss_name
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

        # Wavelength Config
        if wavelength_params is None:
            self.wavelength_params: dict[str, Any] = {}
        elif isinstance(wavelength_params, WavelengthParams):
            self.wavelength_params = asdict(wavelength_params)
        elif isinstance(wavelength_params, Mapping):
            self.wavelength_params = dict(wavelength_params)
        else:
            raise TypeError("wavelength_params must be a WavelengthParams dataclass or mapping.")

        # Model Hyperparams
        self.learning_rate = lr
        self.weight_decay = weight_decay
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.num_classes = condition_dim
        self.predict_temperature = float(temperature)
        self.predict_guidance_scale = float(guidance_scale)
        self.predict_condition_scale = float(condition_scale)
        self.register_buffer("wavelengths_nm", self._build_wavelengths_nm(input_dim), persistent=False)

        # Scale-VAE state (persist f_bar for inference-time scaling)
        self.register_buffer("scale_f_bar", torch.ones(latent_dim, dtype=torch.float32))
        self.register_buffer("scale_f_accum", torch.zeros(latent_dim, dtype=torch.float32), persistent=False)
        self.register_buffer("scale_f_count", torch.tensor(0, dtype=torch.long), persistent=False)

    def _is_scale_vae(self) -> bool:
        """Return whether the active loss selection is Scale-VAE."""
        return self.loss_name == "scale_vae"

    def _build_wavelengths_nm(self, input_dim: int) -> torch.Tensor:
        """Build fixed wavelength coordinates and validate against input_dim."""
        start = int(self.wavelength_params.get("start_nm", 400))
        end = int(self.wavelength_params.get("end_nm", 2490))
        step = int(self.wavelength_params.get("step_nm", 10))
        if step <= 0:
            raise ValueError("wavelength_params.step_nm must be positive.")
        if end < start:
            raise ValueError("wavelength_params.end_nm must be >= wavelength_params.start_nm.")
        span = end - start
        if span % step != 0:
            raise ValueError("wavelength_params must satisfy (end_nm - start_nm) % step_nm == 0.")
        n_wavelengths = (span // step) + 1
        if n_wavelengths != input_dim:
            raise ValueError(
                f"wavelength_params imply {n_wavelengths} bands but model.input_dim is {input_dim}."
            )
        return torch.arange(start, end + 1, step, dtype=torch.float32)

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

    def _encode_with_optional_memory(
        self, spectrum: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Encode inputs and optionally return token memory for attention-based decoders."""
        if self.architecture in {"transformer", "conformer"}:
            token_model = cast(Union[TransformerConditionalVAE, ConformerConditionalVAE], self.model)
            mu, logvar, memory = token_model.encode(spectrum, condition)
            return mu, logvar, memory
        seq_model = cast(Union[ConditionalVAE, ConvConditionalVAE], self.model)
        mu, logvar = seq_model.encode(spectrum, condition)
        return mu, logvar, None

    def _decode_with_optional_memory(
        self, z: torch.Tensor, condition: torch.Tensor, memory: torch.Tensor | None
    ) -> torch.Tensor:
        """Decode latent samples, passing encoder memory when the decoder supports it."""
        if self.architecture in {"transformer", "conformer"}:
            token_model = cast(Union[TransformerConditionalVAE, ConformerConditionalVAE], self.model)
            return token_model.decode(z, condition, memory)
        seq_model = cast(Union[ConditionalVAE, ConvConditionalVAE], self.model)
        return seq_model.decode(z, condition)

    def _compute_scale_factor(self, mu: torch.Tensor) -> torch.Tensor:
        """Compute per-latent-dimension Scale-VAE factor from batch statistics."""
        des_std = float(self.loss_params.get("des_std", 1.0))
        eps = float(self.loss_params.get("scale_eps", 1e-6))
        mu_std = mu.detach().std(dim=0, unbiased=False).clamp_min(eps)
        factor = torch.full_like(mu_std, des_std) / mu_std
        return factor.to(device=mu.device, dtype=mu.dtype)

    def _select_scale_factor(self, mu: torch.Tensor, stage: str) -> torch.Tensor:
        """Pick batch factor or epoch-average factor based on stage and warmup epoch."""
        f_epo = max(int(self.loss_params.get("f_epo", 0)), 0)

        if stage == "train":
            batch_factor = self._compute_scale_factor(mu)
            scale_f_accum = cast(torch.Tensor, self.scale_f_accum)
            scale_f_count = cast(torch.Tensor, self.scale_f_count)
            scale_f_accum.add_(
                batch_factor.detach().to(device=scale_f_accum.device, dtype=scale_f_accum.dtype)
            )
            scale_f_count.add_(1)
            if (self.current_epoch + 1) <= f_epo:
                return batch_factor
            scale_f_bar = cast(torch.Tensor, self.scale_f_bar)
            return scale_f_bar.to(device=mu.device, dtype=mu.dtype)

        if (self.current_epoch + 1) <= f_epo:
            return self._compute_scale_factor(mu)
        scale_f_bar = cast(torch.Tensor, self.scale_f_bar)
        return scale_f_bar.to(device=mu.device, dtype=mu.dtype)

    def _apply_inference_scale(self, z: torch.Tensor) -> torch.Tensor:
        """Apply learned Scale-VAE latent scaling at inference/prediction time."""
        if not self._is_scale_vae():
            return z
        scale_f_bar = cast(torch.Tensor, self.scale_f_bar)
        return z * scale_f_bar.to(device=z.device, dtype=z.dtype)

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        """Run one train/val/test step with either standard VAE or Scale-VAE flow."""
        spectra, cond = batch["spectrum"], batch["condition"]

        if self._is_scale_vae():
            mu, logvar, memory = self._encode_with_optional_memory(spectra, cond)
            scale_factor = self._select_scale_factor(mu, stage)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu * scale_factor.unsqueeze(0) + eps * std
            recon = self._decode_with_optional_memory(z, cond, memory)
            loss, metrics = self.loss_fn(spectra, recon, mu, logvar, self.loss_params)
            metrics = dict(metrics)
            metrics["scale_factor_mean"] = scale_factor.mean()
            metrics["scale_factor_std"] = scale_factor.std(unbiased=False)
        else:
            recon, mu, logvar = self.model(spectra, cond)
            loss, metrics = self.loss_fn(spectra, recon, mu, logvar, self.loss_params)

        self.log(f"{stage}_loss", loss, prog_bar=True)
        for name, value in metrics.items():
            self.log(f"{stage}_{name}", value, prog_bar=False, logger=True)
        return loss

    def on_train_epoch_start(self) -> None:
        """Reset epoch accumulators used to estimate the next Scale-VAE average factor."""
        if not self._is_scale_vae():
            return
        cast(torch.Tensor, self.scale_f_accum).zero_()
        cast(torch.Tensor, self.scale_f_count).zero_()

    def on_train_epoch_end(self) -> None:
        """Finalize epoch-average latent scaling factor for subsequent epochs."""
        if not self._is_scale_vae():
            return
        scale_f_count = cast(torch.Tensor, self.scale_f_count)
        scale_f_accum = cast(torch.Tensor, self.scale_f_accum)
        scale_f_bar = cast(torch.Tensor, self.scale_f_bar)
        count = int(scale_f_count.item())
        if count <= 0:
            return
        next_factor = scale_f_accum / float(count)
        scale_f_bar.copy_(next_factor.to(device=scale_f_bar.device, dtype=scale_f_bar.dtype))
        self.log("train_scale_factor_mean", scale_f_bar.mean(), prog_bar=False, logger=True)

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
    ) -> dict[str, torch.Tensor]:
        """Generate conditioned spectra for prediction callbacks and exporters."""
        conditions = batch["condition"].to(self.device)
        n = conditions.size(0)

        sample_ids = batch.get("sample_id")
        if sample_ids is not None:
            sample_ids = sample_ids.to(self.device).view(-1)
        else:
            sample_ids = torch.arange(n, device=self.device)

        unique_ids, inverse_indices = torch.unique(sample_ids, sorted=True, return_inverse=True)
        latent_samples = torch.randn(unique_ids.size(0), self.latent_dim, device=conditions.device)
        z = latent_samples[inverse_indices]
        z = z * max(self.predict_temperature, 1e-6)
        z = self._apply_inference_scale(z)

        scaled_conditions = conditions * self.predict_condition_scale

        if math.isclose(self.predict_guidance_scale, 1.0):
            spectra = self.model.decode(z, scaled_conditions)
        else:
            zero_cond = torch.zeros_like(scaled_conditions)
            uncond = self.model.decode(z, zero_cond)
            cond_out = self.model.decode(z, scaled_conditions)
            spectra = uncond + self.predict_guidance_scale * (cond_out - uncond)
        spectra = spectra * 2.0 - 1.0

        payload: dict[str, torch.Tensor] = {
            "spectra": spectra.view(n, 1, 1, -1),
            "sample_id": sample_ids.detach().cpu(),
        }

        cond_idx = batch.get("condition_index")
        if cond_idx is not None:
            payload["condition_index"] = cond_idx.detach().cpu()

        return payload

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
        z = self._apply_inference_scale(z)
        recon = self.model.decode(z, cond)
        recon = recon * 2.0 - 1.0
        return recon.view(n_samples, 1, 1, -1)
