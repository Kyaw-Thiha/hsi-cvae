from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import torch
from lightning.pytorch import Callback, LightningModule, Trainer

try:
    import plotly.graph_objects as go
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency guard
    raise ModuleNotFoundError(
        "Plotly is required for the PredictLineCharts callback. "
        "Install it with `pip install plotly` to enable spectral chart logging."
    ) from exc


class PredictLineCharts(Callback):
    """Aggregate spectra during prediction and export a shared chart plus per-sample dumps."""

    def __init__(
        self,
        out_dir: str,
        start_nm: float = 400.0,
        step_nm: float = 10.0,
        class_names: Optional[Sequence[str]] = None,
        custom_conditions: Optional[Sequence[Sequence[float]]] = None,
        normalize_custom_conditions: bool = True,
        save_spectra: bool = True,
        spectra_format: str = "csv",
        spectra_subdir: str = "spectra",
        max_traces: Optional[int] = None,
        condition_key: str = "condition",
    ) -> None:
        super().__init__()
        if spectra_format not in {"csv", "npy"}:
            raise ValueError("spectra_format must be either 'csv' or 'npy'")
        self.out_dir = Path(out_dir)
        self.start_nm = float(start_nm)
        self.step_nm = float(step_nm)
        self.class_names = list(class_names) if class_names is not None else None
        self.normalize_custom_conditions = normalize_custom_conditions
        self.save_spectra = save_spectra
        self.spectra_format = spectra_format
        self.spectra_subdir = Path(spectra_subdir)
        self.max_traces = max_traces
        self.condition_key = condition_key

        if custom_conditions is not None:
            cond_tensor = torch.tensor(custom_conditions, dtype=torch.float32)
            if normalize_custom_conditions and cond_tensor.numel() > 0:
                cond_tensor = self._normalize(cond_tensor)
            self.custom_conditions = cond_tensor
        else:
            self.custom_conditions = None

        self._spectra: list[torch.Tensor] = []
        self._conditions: list[torch.Tensor] = []
        self._collect_conditions = True

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._spectra.clear()
        self._conditions.clear()
        self._collect_conditions = True
        if trainer.is_global_zero:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            if self.save_spectra:
                (self.out_dir / self.spectra_subdir).mkdir(parents=True, exist_ok=True)

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if outputs is None:
            return
        if isinstance(outputs, (list, tuple)):
            if not outputs:
                return
            outputs = outputs[0]
        if not isinstance(outputs, torch.Tensor):
            outputs = torch.as_tensor(outputs)

        spectra = outputs.detach().cpu()
        if spectra.ndim <= 1:
            return
        if spectra.ndim > 2:
            spectra = spectra.view(spectra.size(0), -1)
        self._spectra.append(spectra.float())

        if self._collect_conditions and isinstance(batch, Mapping) and self.condition_key in batch:
            cond_tensor = torch.as_tensor(batch[self.condition_key]).detach().cpu()
            if cond_tensor.ndim == 1:
                cond_tensor = cond_tensor.unsqueeze(-1)
            elif cond_tensor.ndim > 2:
                cond_tensor = cond_tensor.view(cond_tensor.size(0), -1)
            self._conditions.append(cond_tensor.float())
        else:
            self._collect_conditions = False
            self._conditions.clear()

    def on_predict_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not trainer.is_global_zero or not self._spectra:
            return

        spectra = torch.cat(self._spectra, dim=0)
        num_samples, seq_len = spectra.shape
        wavelengths = (self.start_nm + self.step_nm * torch.arange(seq_len)).tolist()
        values = ((spectra.clamp(-1.0, 1.0) + 1.0) / 2.0)

        conditions = None
        if self._collect_conditions and self._conditions:
            conditions = torch.cat(self._conditions, dim=0)
            if self.normalize_custom_conditions and conditions.size(1) > 0:
                conditions = self._normalize(conditions)

        self._write_chart(wavelengths, values, conditions)

        if self.save_spectra:
            self._export_spectra(wavelengths, values, conditions)

    def _write_chart(
        self,
        wavelengths: list[float],
        spectra: torch.Tensor,
        conditions: Optional[torch.Tensor],
    ) -> None:
        limit = self.max_traces or spectra.size(0)
        limit = min(limit, spectra.size(0))
        fig = go.Figure()
        data = spectra[:limit].cpu().numpy()
        cond_list = conditions[:limit] if conditions is not None else None

        for idx in range(limit):
            cond_vec = cond_list[idx] if cond_list is not None else None
            trace_name = self._resolve_trace_name(idx, cond_vec)
            fig.add_trace(go.Scatter(x=wavelengths, y=data[idx].tolist(), mode="lines", name=trace_name))

        fig.update_layout(
            title="Predicted Spectra",
            xaxis_title="Wavelength (nm)",
            yaxis_title="Reflectance",
            hovermode="x unified",
            template="plotly_dark",
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = self.out_dir / f"predict_{timestamp}.html"
        fig.write_html(str(out_path), auto_open=False, include_plotlyjs="cdn")

    def _export_spectra(
        self,
        wavelengths: list[float],
        spectra: torch.Tensor,
        conditions: Optional[torch.Tensor],
    ) -> None:
        spectra_dir = self.out_dir / self.spectra_subdir
        cond_np = conditions.cpu().numpy() if conditions is not None else None

        for idx in range(spectra.size(0)):
            cond_vec = None
            if cond_np is not None:
                cond_vec = cond_np[idx]
            base = self._slugify(self._resolve_trace_name(idx, cond_vec))
            base_name = f"{idx:05d}_{base}" if base else f"{idx:05d}"
            if self.spectra_format == "csv":
                path = spectra_dir / f"{base_name}.csv"
                self._save_csv(path, wavelengths, spectra[idx], cond_vec)
            else:
                path = spectra_dir / f"{base_name}.npy"
                self._save_npy(path, wavelengths, spectra[idx], cond_vec)

    def _resolve_trace_name(self, index: int, condition: Optional[torch.Tensor | np.ndarray]) -> str:
        cond_tensor: Optional[torch.Tensor]
        if isinstance(condition, np.ndarray):
            cond_tensor = torch.from_numpy(condition) if condition.size else None
        else:
            cond_tensor = condition

        if cond_tensor is not None and self.custom_conditions is not None and cond_tensor.numel() > 0:
            diffs = torch.abs(self.custom_conditions - cond_tensor.unsqueeze(0))
            matches = torch.all(diffs < 1e-4, dim=1)
            if matches.any():
                match_idx = int(matches.nonzero(as_tuple=True)[0][0])
                if self.class_names and match_idx < len(self.class_names):
                    return str(self.class_names[match_idx])
                return f"condition_{match_idx}"

        if self.class_names:
            return str(self.class_names[index % len(self.class_names)])
        return f"spectrum_{index}"

    def _save_csv(
        self,
        path: Path,
        wavelengths: list[float],
        spectrum: torch.Tensor,
        condition: Optional[np.ndarray],
    ) -> None:
        values = spectrum.cpu().numpy().tolist()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["wavelength_nm", *[f"{w:.2f}" for w in wavelengths]])
            writer.writerow(["reflectance", *[f"{v:.6f}" for v in values]])
            if condition is not None and condition.size > 0:
                cond_vals = [f"{float(v):.6f}" for v in condition.tolist()]
                writer.writerow(["condition", *cond_vals])

    def _save_npy(
        self,
        path: Path,
        wavelengths: list[float],
        spectrum: torch.Tensor,
        condition: Optional[np.ndarray],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "wavelengths": np.asarray(wavelengths, dtype=np.float32),
            "reflectance": spectrum.cpu().numpy(),
        }
        if condition is not None and condition.size > 0:
            data["condition"] = condition.astype(np.float32)
        np.save(path, data, allow_pickle=True)

    @staticmethod
    def _normalize(tensor: torch.Tensor) -> torch.Tensor:
        denom = tensor.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return tensor / denom

    @staticmethod
    def _slugify(value: str) -> str:
        return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value).strip("_")
