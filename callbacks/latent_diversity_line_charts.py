from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from lightning.pytorch import Callback, LightningModule, Trainer

from .path_utils import resolve_callback_out_dir

try:
    import plotly.graph_objects as go
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency guard
    raise ModuleNotFoundError(
        "Plotly is required for LatentDiversityLineCharts. "
        "Install it with `pip install plotly` to enable spectral chart logging."
    ) from exc


class LatentDiversityLineCharts(Callback):
    """Plot per-condition generated diversity against up to two matching original pure spectra."""

    def __init__(
        self,
        csv_path: str = "data/simpler_data_rwc.csv",
        out_dir: str = "outputs/line_charts/latent_diversity",
        start_nm: int = 400,
        end_nm: int = 2490,
        step_nm: int = 10,
        condition_columns: Optional[Sequence[str]] = ("gv_fraction", "npv_fraction", "soil_fraction"),
        spectrum_id_column: str = "Spectra",
        normalization: str = "dataset",
        max_original_samples: int = 2,
        random_seed: int = 42,
        run_every_n_epochs: int = 1,
        samples_per_condition: int = 16,
        sample_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.csv_path = Path(csv_path)
        self.out_dir = Path(out_dir)
        self.start_nm = int(start_nm)
        self.end_nm = int(end_nm)
        self.step_nm = int(step_nm)
        self.condition_columns = list(condition_columns) if condition_columns else []
        self.spectrum_id_column = spectrum_id_column
        self.normalization = self._validate_normalization(normalization)
        self.max_original_samples = min(max(int(max_original_samples), 0), 2)
        self.random_seed = int(random_seed)
        self.run_every_n_epochs = max(int(run_every_n_epochs), 1)
        self.samples_per_condition = max(int(samples_per_condition), 1)
        self.sample_temperature = max(float(sample_temperature), 1e-6)

        self._wavelengths: list[int] = []
        self._pure_values_by_material: dict[str, np.ndarray] = {}
        self._pure_names_by_material: dict[str, list[str]] = {}

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del trainer, pl_module
        self._prepare_original_data()

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if (trainer.current_epoch % self.run_every_n_epochs) != 0:
            return

        sample_fn = getattr(pl_module, "sample", None)
        condition_dim = int(getattr(pl_module, "condition_dim", 0))
        if sample_fn is None or condition_dim <= 0:
            return

        if not self._wavelengths or not self._pure_values_by_material:
            self._prepare_original_data()

        out_dir = resolve_callback_out_dir(trainer.default_root_dir, self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        device = pl_module.device

        for material in ("gv", "npv", "soil"):
            condition = self._pure_condition(material=material, condition_dim=condition_dim, device=device)
            if condition is None:
                continue

            cond_batch = condition.repeat(self.samples_per_condition, 1)
            predicted = sample_fn(
                n=self.samples_per_condition,
                conditions=cond_batch,
                device=device,
                temperature=self.sample_temperature,
            )
            pred_lines = predicted.view(self.samples_per_condition, -1).detach().cpu().numpy()
            pred_lines = np.clip((pred_lines + 1.0) / 2.0, 0.0, 1.0)

            orig_values, orig_names = self._sample_original(material=material, epoch=trainer.current_epoch)
            self._write_chart(
                epoch=trainer.current_epoch,
                material=material,
                pred_lines=pred_lines,
                orig_values=orig_values,
                orig_names=orig_names,
                out_dir=out_dir,
            )

    def _prepare_original_data(self) -> None:
        df = pd.read_csv(self.csv_path)
        spectral_columns = self._infer_spectral_columns(df.columns)
        if not spectral_columns:
            raise ValueError("No spectral columns detected in provided CSV.")

        self._wavelengths = [int(col) for col in spectral_columns]
        values = df[spectral_columns].to_numpy(dtype=np.float32)
        values = self._apply_normalization(values)

        for material in ("gv", "npv", "soil"):
            idx = self._find_condition_index(material)
            if idx is None or idx >= len(self.condition_columns):
                self._pure_values_by_material[material] = np.empty((0, values.shape[1]), dtype=np.float32)
                self._pure_names_by_material[material] = []
                continue

            pure = np.zeros(len(self.condition_columns), dtype=np.float32)
            pure[idx] = 1.0
            cond_matrix = df[self.condition_columns].to_numpy(dtype=np.float32)
            mask = np.all(np.isclose(cond_matrix, pure, atol=1e-6), axis=1)

            pure_values = values[mask]
            pure_df = df.loc[mask]
            pure_names = self._trace_names(pure_df)

            self._pure_values_by_material[material] = pure_values
            self._pure_names_by_material[material] = pure_names

    def _sample_original(self, material: str, epoch: int) -> tuple[np.ndarray, list[str]]:
        values = self._pure_values_by_material.get(material)
        names = self._pure_names_by_material.get(material, [])
        if values is None or values.size == 0 or self.max_original_samples == 0:
            return np.empty((0, len(self._wavelengths)), dtype=np.float32), []

        n = min(values.shape[0], self.max_original_samples)
        rng = np.random.default_rng(self.random_seed + epoch + self._material_offset(material))
        picked = np.sort(rng.choice(values.shape[0], size=n, replace=False))
        return values[picked], [names[int(i)] for i in picked]

    def _write_chart(
        self,
        epoch: int,
        material: str,
        pred_lines: np.ndarray,
        orig_values: np.ndarray,
        orig_names: Sequence[str],
        out_dir: Path,
    ) -> None:
        fig = go.Figure()

        mean_line = pred_lines.mean(axis=0)
        std_line = pred_lines.std(axis=0)
        upper = np.clip(mean_line + std_line, 0.0, 1.0)
        lower = np.clip(mean_line - std_line, 0.0, 1.0)

        fig.add_trace(
            go.Scatter(
                x=self._wavelengths,
                y=upper.tolist(),
                mode="lines",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self._wavelengths,
                y=lower.tolist(),
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor="rgba(100, 149, 237, 0.20)",
                name="Generated mean ± std",
                hoverinfo="skip",
            )
        )

        for idx in range(pred_lines.shape[0]):
            fig.add_trace(
                go.Scatter(
                    x=self._wavelengths,
                    y=pred_lines[idx].tolist(),
                    mode="lines",
                    name=f"Generated #{idx + 1}",
                    line={"width": 1},
                    opacity=0.30,
                    showlegend=(idx == 0),
                )
            )

        fig.add_trace(
            go.Scatter(
                x=self._wavelengths,
                y=mean_line.tolist(),
                mode="lines",
                name="Generated mean",
                line={"width": 3},
            )
        )

        for idx in range(orig_values.shape[0]):
            fig.add_trace(
                go.Scatter(
                    x=self._wavelengths,
                    y=orig_values[idx].tolist(),
                    mode="lines",
                    name=f"Original pure {material} #{idx + 1} ({orig_names[idx]})",
                    line={"dash": "dot", "width": 2},
                )
            )

        fig.update_layout(
            title=f"Latent Diversity - {material.upper()} - Epoch {epoch}",
            xaxis_title="Wavelength (nm)",
            yaxis_title="Reflectance",
            hovermode="x unified",
            template="plotly_dark",
        )
        out_path = out_dir / f"epoch_{epoch}_{material}.html"
        fig.write_html(str(out_path), auto_open=False, include_plotlyjs="cdn")

    def _pure_condition(self, material: str, condition_dim: int, device: torch.device) -> Optional[torch.Tensor]:
        idx = self._find_condition_index(material)
        if idx is None or idx >= condition_dim:
            return None

        cond = torch.zeros(1, condition_dim, dtype=torch.float32, device=device)
        cond[0, idx] = 1.0
        return cond

    def _find_condition_index(self, material: str) -> Optional[int]:
        material_lower = material.lower()
        for idx, col in enumerate(self.condition_columns):
            col_lower = col.lower()
            if material_lower in col_lower:
                return idx
        return None

    def _trace_names(self, df: pd.DataFrame) -> list[str]:
        if self.spectrum_id_column in df.columns:
            return [str(v) for v in df[self.spectrum_id_column].tolist()]
        return [f"row_{int(i)}" for i in df.index.to_list()]

    def _infer_spectral_columns(self, columns: Iterable[str]) -> list[str]:
        spectral_columns: list[str] = []
        for col in columns:
            try:
                value = int(col)
            except ValueError:
                continue
            if self.start_nm <= value <= self.end_nm and ((value - self.start_nm) % self.step_nm == 0):
                spectral_columns.append(col)
        spectral_columns.sort(key=int)
        return spectral_columns

    @staticmethod
    def _validate_normalization(value: str) -> str:
        mode = str(value).lower()
        if mode not in {"dataset", "row", "none"}:
            raise ValueError("normalization must be one of: dataset, row, none.")
        return mode

    def _apply_normalization(self, values: np.ndarray) -> np.ndarray:
        if self.normalization == "none":
            return values
        if self.normalization == "row":
            return self._normalize_rows(values)
        return self._normalize_dataset(values)

    @staticmethod
    def _normalize_dataset(values: np.ndarray) -> np.ndarray:
        min_value = float(values.min())
        max_value = float(values.max())
        value_range = max(max_value - min_value, 1e-6)
        return (values - min_value) / value_range

    @staticmethod
    def _normalize_rows(values: np.ndarray) -> np.ndarray:
        mins = values.min(axis=1, keepdims=True)
        maxs = values.max(axis=1, keepdims=True)
        ranges = np.clip(maxs - mins, a_min=1e-6, a_max=None)
        return (values - mins) / ranges

    @staticmethod
    def _material_offset(material: str) -> int:
        offsets = {"npv": 17, "soil": 31, "gv": 47}
        return offsets.get(material, 0)
