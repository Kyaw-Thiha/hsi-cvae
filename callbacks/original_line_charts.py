from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from lightning.pytorch import Callback, LightningModule, Trainer

try:
    import plotly.graph_objects as go
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency guard
    raise ModuleNotFoundError(
        "Plotly is required for OriginalLineCharts. "
        "Install it with `pip install plotly` to enable spectral chart logging."
    ) from exc


class OriginalLineCharts(Callback):
    """Render original CSV spectra as interactive Plotly line charts."""

    def __init__(
        self,
        csv_path: str = "data/simpler_data_rwc.csv",
        out_dir: str = "outputs/original/charts",
        start_nm: int = 400,
        end_nm: int = 2490,
        step_nm: int = 10,
        use_column: str = "use",
        spectrum_id_column: str = "Spectra",
        condition_columns: Optional[Sequence[str]] = ("gv_fraction", "npv_fraction", "soil_fraction"),
        normalization: str = "dataset",
        normalize_rows: Optional[bool] = None,
        max_traces: int = 250,
        random_seed: int = 42,
        run_on_stage: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.csv_path = Path(csv_path)
        self.out_dir = Path(out_dir)
        self.start_nm = int(start_nm)
        self.end_nm = int(end_nm)
        self.step_nm = int(step_nm)
        self.use_column = use_column
        self.spectrum_id_column = spectrum_id_column
        self.condition_columns = list(condition_columns) if condition_columns else []
        # Backward-compatible alias: old configs set normalize_rows.
        # New behavior defaults to dataset-level normalization to match training targets.
        if normalize_rows is not None:
            normalization = "row" if bool(normalize_rows) else "none"
        self.normalization = self._validate_normalization(normalization)
        self.max_traces = max(int(max_traces), 1)
        self.random_seed = int(random_seed)
        self.run_on_stage = run_on_stage

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del trainer, pl_module
        if self.run_on_stage == "fit":
            self.export()

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del trainer, pl_module
        if self.run_on_stage == "predict":
            self.export()

    def export(self) -> None:
        df = pd.read_csv(self.csv_path)
        spectral_columns = self._infer_spectral_columns(df.columns)
        if not spectral_columns:
            raise ValueError("No spectral columns detected in provided CSV.")

        self.out_dir.mkdir(parents=True, exist_ok=True)
        wavelengths = [int(col) for col in spectral_columns]
        values = df[spectral_columns].to_numpy(dtype=np.float32)
        values = self._apply_normalization(values)

        self._write_chart(
            out_path=self.out_dir / "all_spectra.html",
            wavelengths=wavelengths,
            values=values,
            trace_names=self._trace_names(df),
            title="Original Spectra - All Rows",
        )

        if self.use_column in df.columns:
            for split_name, split_df in df.groupby(self.use_column, dropna=False):
                split_name_str = str(split_name)
                split_idx = split_df.index.to_numpy()
                split_values = values[split_idx]
                self._write_chart(
                    out_path=self.out_dir / f"use_{self._slugify(split_name_str)}.html",
                    wavelengths=wavelengths,
                    values=split_values,
                    trace_names=self._trace_names(split_df),
                    title=f"Original Spectra - {self.use_column}={split_name_str}",
                )

        if self.condition_columns and all(col in df.columns for col in self.condition_columns):
            pure_conditions = np.eye(len(self.condition_columns), dtype=np.float32)
            cond_matrix = df[self.condition_columns].to_numpy(dtype=np.float32)
            for idx, cond in enumerate(pure_conditions):
                mask = np.all(np.isclose(cond_matrix, cond, atol=1e-6), axis=1)
                if not np.any(mask):
                    continue
                pure_df = df.loc[mask]
                pure_values = values[mask]
                name = self.condition_columns[idx]
                self._write_chart(
                    out_path=self.out_dir / f"pure_{self._slugify(name)}.html",
                    wavelengths=wavelengths,
                    values=pure_values,
                    trace_names=self._trace_names(pure_df),
                    title=f"Original Spectra - Pure {name}",
                )

    def _write_chart(
        self,
        out_path: Path,
        wavelengths: list[int],
        values: np.ndarray,
        trace_names: Sequence[str],
        title: str,
    ) -> None:
        if values.size == 0:
            return

        subset_values, subset_names = self._sample_rows(values, trace_names)
        fig = go.Figure()
        for idx in range(subset_values.shape[0]):
            fig.add_trace(
                go.Scatter(
                    x=wavelengths,
                    y=subset_values[idx].tolist(),
                    mode="lines",
                    name=subset_names[idx],
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title="Wavelength (nm)",
            yaxis_title="Reflectance",
            hovermode="x unified",
            template="plotly_dark",
        )
        fig.write_html(str(out_path), auto_open=False, include_plotlyjs="cdn")

    def _sample_rows(self, values: np.ndarray, trace_names: Sequence[str]) -> tuple[np.ndarray, list[str]]:
        if values.shape[0] <= self.max_traces:
            return values, list(trace_names)
        rng = np.random.default_rng(self.random_seed)
        picked = np.sort(rng.choice(values.shape[0], size=self.max_traces, replace=False))
        return values[picked], [trace_names[int(i)] for i in picked]

    def _trace_names(self, df: pd.DataFrame) -> list[str]:
        if self.spectrum_id_column in df.columns:
            return [str(value) for value in df[self.spectrum_id_column].tolist()]
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
    def _slugify(value: str) -> str:
        return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value).strip("_")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick original-dataset spectral chart generator.")
    parser.add_argument("--csv-path", type=str, default="data/simpler_data_rwc.csv")
    parser.add_argument("--out-dir", type=str, default="outputs/original/charts")
    parser.add_argument("--start-nm", type=int, default=400)
    parser.add_argument("--end-nm", type=int, default=2490)
    parser.add_argument("--step-nm", type=int, default=10)
    parser.add_argument("--use-column", type=str, default="use")
    parser.add_argument("--spectrum-id-column", type=str, default="Spectra")
    parser.add_argument("--max-traces", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--normalization",
        type=str,
        choices=["dataset", "row", "none"],
        default="dataset",
    )
    parser.add_argument("--no-normalize-rows", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    normalization = "none" if args.no_normalize_rows else args.normalization
    callback = OriginalLineCharts(
        csv_path=args.csv_path,
        out_dir=args.out_dir,
        start_nm=args.start_nm,
        end_nm=args.end_nm,
        step_nm=args.step_nm,
        use_column=args.use_column,
        spectrum_id_column=args.spectrum_id_column,
        normalization=normalization,
        max_traces=args.max_traces,
        random_seed=args.seed,
    )
    callback.export()
    print(f"Saved original-data charts in: {Path(args.out_dir)}")


if __name__ == "__main__":
    main()
