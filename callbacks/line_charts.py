from __future__ import annotations

import os
from typing import Optional, Sequence

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

try:
    import plotly.graph_objects as go
except ModuleNotFoundError as exc:  # pragma: no cover - import guard for optional dependency
    raise ModuleNotFoundError(
        "Plotly is required for the SampleLineCharts callback. "
        "Install it with `pip install plotly` to enable spectral chart logging."
    ) from exc


class SampleLineCharts(Callback):
    """Generate interactive spectral line charts for each condition on validation epochs."""

    def __init__(
        self,
        out_dir: str = "outputs/spectral_charts",
        start_nm: float = 400.0,
        step_nm: float = 10.0,
        num_samples_per_class: int = 1,
        class_names: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        if num_samples_per_class < 1:
            raise ValueError("num_samples_per_class must be >= 1")
        self.out_dir = out_dir
        self.start_nm = start_nm
        self.step_nm = step_nm
        self.num_samples_per_class = num_samples_per_class
        self.class_names = list(class_names) if class_names is not None else None

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Sample spectra per class and write interactive charts as HTML files."""
        if not trainer.logger:
            return
        num_classes = int(getattr(pl_module, "num_classes", 0))
        sample_fn = getattr(pl_module, "sample", None)
        if num_classes <= 0 or sample_fn is None:
            return

        device = pl_module.device
        labels = torch.arange(num_classes, device=device).repeat_interleave(self.num_samples_per_class)
        spectra = sample_fn(n=labels.numel(), y=labels, device=device)
        spectra = spectra.view(labels.numel(), -1).detach().cpu()
        if spectra.numel() == 0:
            return

        wavelengths = (self.start_nm + self.step_nm * torch.arange(spectra.shape[1])).tolist()

        os.makedirs(self.out_dir, exist_ok=True)
        fig = go.Figure()
        label_list = labels.cpu().tolist()
        data = spectra.numpy()

        for idx, label in enumerate(label_list):
            class_name = self._resolve_class_name(label)
            sample_idx = idx % self.num_samples_per_class
            trace_name = class_name if self.num_samples_per_class == 1 else f"{class_name} #{sample_idx + 1}"
            values = (data[idx].clip(-1.0, 1.0) + 1.0) / 2.0
            fig.add_trace(go.Scatter(x=wavelengths, y=values.tolist(), mode="lines", name=trace_name))

        fig.update_layout(
            title=f"Spectral Samples - Epoch {trainer.current_epoch}",
            xaxis_title="Wavelength (nm)",
            yaxis_title="Reflectance",
            hovermode="x unified",
            template="plotly_dark",
        )

        out_path = os.path.join(self.out_dir, f"epoch_{trainer.current_epoch}.html")
        fig.write_html(out_path, auto_open=False, include_plotlyjs="cdn")

    def _resolve_class_name(self, label: int) -> str:
        if self.class_names:
            idx = label % len(self.class_names)
            return str(self.class_names[idx])
        return f"class_{label}"
