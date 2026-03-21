from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import torch
from lightning.pytorch import Callback, LightningModule, Trainer

from .path_utils import resolve_callback_out_dir

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

        self.custom_conditions = None
        self._set_custom_conditions(custom_conditions)

        self._groups: dict[int, list[dict[str, torch.Tensor | Optional[int]]]] = {}
        self._resolved_out_dir = self.out_dir

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        self._groups.clear()
        if self.custom_conditions is None:
            datamodule = getattr(trainer, "datamodule", None)
            resolved = getattr(datamodule, "resolved_predict_conditions", None)
            if resolved:
                self._set_custom_conditions(resolved)
        self._resolved_out_dir = resolve_callback_out_dir(trainer.default_root_dir, self.out_dir)
        if trainer.is_global_zero:
            self._resolved_out_dir.mkdir(parents=True, exist_ok=True)

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor] | None,
        batch: Mapping[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        parsed = self._parse_outputs(outputs)
        if parsed is None:
            return

        spectra = parsed["spectra"].detach().cpu()
        if spectra.ndim <= 1:
            return
        if spectra.ndim > 2:
            spectra = spectra.view(spectra.size(0), -1)

        batch_conditions = batch.get(self.condition_key)
        cond_tensor: Optional[torch.Tensor] = None
        if batch_conditions is not None:
            cond_tensor = torch.as_tensor(batch_conditions).detach().cpu()
            if cond_tensor.ndim == 1:
                cond_tensor = cond_tensor.unsqueeze(-1)
            elif cond_tensor.ndim > 2:
                cond_tensor = cond_tensor.view(cond_tensor.size(0), -1)

        sample_ids = parsed.get("sample_id")
        if sample_ids is not None:
            sample_ids = torch.as_tensor(sample_ids).detach().cpu().view(-1)
        else:
            sample_ids = torch.arange(spectra.size(0))

        condition_indices = parsed.get("condition_index")
        cond_idx_tensor: Optional[torch.Tensor] = None
        if condition_indices is not None:
            cond_idx_tensor = torch.as_tensor(condition_indices).detach().cpu().view(-1)
        elif "condition_index" in batch:
            cond_idx_tensor = torch.as_tensor(batch["condition_index"]).detach().cpu().view(-1)

        for row in range(spectra.size(0)):
            sample_key = int(sample_ids[row])
            entry: dict[str, torch.Tensor | Optional[int]] = {
                "spectrum": spectra[row].float(),
            }
            if cond_tensor is not None:
                entry["condition"] = cond_tensor[row].float()
            if cond_idx_tensor is not None:
                entry["condition_index"] = int(cond_idx_tensor[row])
            self._groups.setdefault(sample_key, []).append(entry)

    def on_predict_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        if not trainer.is_global_zero or not self._groups:
            return

        # Determine sequence length from first stored spectrum
        first_entry = None
        for entries in self._groups.values():
            if entries:
                first_entry = entries[0]
                break
        if first_entry is None:
            return
        seq_len = first_entry["spectrum"].numel()
        wavelengths = (self.start_nm + self.step_nm * torch.arange(seq_len)).tolist()

        for sample_id in sorted(self._groups.keys()):
            entries = self._groups[sample_id]
            if not entries:
                continue
            self._write_sample_outputs(sample_id, wavelengths, entries)

    def _write_chart(
        self,
        out_path: Path,
        wavelengths: list[float],
        spectra: torch.Tensor,
        conditions: Optional[torch.Tensor],
        entries: list[dict[str, torch.Tensor | Optional[int]]],
    ) -> None:
        limit = self.max_traces or spectra.size(0)
        limit = min(limit, spectra.size(0))
        fig = go.Figure()
        data = spectra[:limit].cpu().numpy()
        cond_list = conditions[:limit] if conditions is not None else None

        for idx in range(limit):
            cond_vec = cond_list[idx] if cond_list is not None else entries[idx].get("condition")
            condition_index = entries[idx].get("condition_index")
            trace_name = self._resolve_trace_name(idx, cond_vec, condition_index)
            fig.add_trace(go.Scatter(x=wavelengths, y=data[idx].tolist(), mode="lines", name=trace_name))

        fig.update_layout(
            title="Predicted Spectra",
            xaxis_title="Wavelength (nm)",
            yaxis_title="Reflectance",
            hovermode="x unified",
            template="plotly_dark",
        )

        fig.write_html(str(out_path), auto_open=False, include_plotlyjs="cdn")

    def _export_spectra(
        self,
        spectra_dir: Path,
        wavelengths: list[float],
        spectra: torch.Tensor,
        conditions: Optional[torch.Tensor],
        entries: list[dict[str, torch.Tensor | Optional[int]]],
    ) -> None:
        cond_np = conditions.cpu().numpy() if conditions is not None else None

        for idx in range(spectra.size(0)):
            cond_vec = None
            if cond_np is not None:
                cond_vec = cond_np[idx]
            elif "condition" in entries[idx]:
                cond_vec = torch.as_tensor(entries[idx]["condition"]).numpy()

            trace_name = self._resolve_trace_name(idx, cond_vec, entries[idx].get("condition_index"))
            base = self._slugify(trace_name)
            base_name = f"{idx:03d}_{base}" if base else f"{idx:03d}"
            if self.spectra_format == "csv":
                path = spectra_dir / f"{base_name}.csv"
                self._save_csv(path, wavelengths, spectra[idx], cond_vec)
            else:
                path = spectra_dir / f"{base_name}.npy"
                self._save_npy(path, wavelengths, spectra[idx], cond_vec)

    def _resolve_trace_name(
        self,
        index: int,
        condition: Optional[torch.Tensor | np.ndarray],
        condition_index: Optional[int] = None,
    ) -> str:
        cond_tensor: Optional[torch.Tensor]
        if isinstance(condition, np.ndarray):
            cond_tensor = torch.from_numpy(condition) if condition.size else None
        else:
            cond_tensor = condition

        if condition_index is not None and self.class_names and 0 <= condition_index < len(self.class_names):
            return str(self.class_names[condition_index])

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

    def _set_custom_conditions(self, custom_conditions: Optional[Sequence[Sequence[float]]]) -> None:
        if custom_conditions is None:
            self.custom_conditions = None
            return

        cond_tensor = torch.tensor(custom_conditions, dtype=torch.float32)
        if self.normalize_custom_conditions and cond_tensor.numel() > 0:
            cond_tensor = self._normalize(cond_tensor)
        self.custom_conditions = cond_tensor

    def _parse_outputs(
        self, outputs: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor] | None
    ) -> Optional[dict[str, torch.Tensor]]:
        if outputs is None:
            return None
        if isinstance(outputs, dict):
            spectra = outputs.get("spectra")
            if spectra is None:
                spectra = next(iter(outputs.values()), None)
            if spectra is None:
                return None
            result: dict[str, torch.Tensor] = {"spectra": torch.as_tensor(spectra)}
            if "sample_id" in outputs:
                result["sample_id"] = torch.as_tensor(outputs["sample_id"])
            if "condition_index" in outputs:
                result["condition_index"] = torch.as_tensor(outputs["condition_index"])
            return result
        if isinstance(outputs, (list, tuple)):
            if not outputs:
                return None
            outputs = outputs[0]
        return {"spectra": torch.as_tensor(outputs)}

    def _write_sample_outputs(
        self,
        sample_id: int,
        wavelengths: list[float],
        entries: list[dict[str, torch.Tensor | Optional[int]]],
    ) -> None:
        entries_sorted = sorted(
            entries,
            key=lambda entry: (
                entry.get("condition_index")
                if isinstance(entry.get("condition_index"), int)
                else len(entries)
            ),
        )

        spectra = torch.stack([torch.as_tensor(entry["spectrum"]) for entry in entries_sorted], dim=0)
        has_conditions = all("condition" in entry for entry in entries_sorted)
        conditions = (
            torch.stack([torch.as_tensor(entry["condition"]) for entry in entries_sorted], dim=0) if has_conditions else None
        )
        if conditions is not None and self.normalize_custom_conditions and conditions.size(1) > 0:
            conditions = self._normalize(conditions)

        values = (spectra.clamp(-1.0, 1.0) + 1.0) / 2.0

        sample_dir = self._resolved_out_dir / f"sample_{sample_id:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        chart_path = sample_dir / f"sample_{sample_id:04d}.html"
        self._write_chart(chart_path, wavelengths, values, conditions, entries_sorted)

        if self.save_spectra:
            spectra_dir = sample_dir / self.spectra_subdir
            spectra_dir.mkdir(parents=True, exist_ok=True)
            self._export_spectra(spectra_dir, wavelengths, values, conditions, entries_sorted)

    def _save_csv(
        self,
        path: Path,
        wavelengths: list[float],
        spectrum: torch.Tensor,
        condition: Optional[np.ndarray],
    ) -> None:
        values = spectrum.cpu().numpy().tolist()
        cond_headers: list[str] = []
        cond_values: list[str] = []
        if condition is not None and condition.size > 0:
            cond_headers = [f"condition_{idx}" for idx in range(condition.size)]
            cond_values = [f"{float(v):.6f}" for v in condition.tolist()]

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([*cond_headers, "wavelength_nm", *[f"{w:.2f}" for w in wavelengths]])
            writer.writerow([*cond_values, "reflectance", *[f"{v:.6f}" for v in values]])

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
