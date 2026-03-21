from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

Batch = dict[str, torch.Tensor]


class HyperspectralDataset(Dataset[Batch]):
    """Loads spectra plus conditioning fractions from the CSV file."""

    def __init__(
        self,
        csv_path: str | Path,
        condition_columns: Sequence[str],
        spectral_range: tuple[int, int, int] = (400, 2490, 10),
        dtype: torch.dtype = torch.float32,
        cache_dataframe: bool = False,
    ) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(self.csv_path)

        df = pd.read_csv(self.csv_path)
        self.condition_columns = list(condition_columns)
        self.spectral_columns = self._infer_spectral_columns(df.columns, spectral_range)
        self.row_indices = np.arange(len(df), dtype=np.int64)
        self.spectrum_id_column = "Spectra"
        if self.spectrum_id_column in df.columns:
            self.spectrum_ids = df[self.spectrum_id_column].astype(str).tolist()
            self.has_source_spectrum_ids = True
        else:
            self.spectrum_ids = [f"row_{idx}" for idx in self.row_indices]
            self.has_source_spectrum_ids = False

        raw_spectra = df[self.spectral_columns].to_numpy(dtype=np.float32)
        self.reflectance_min = float(raw_spectra.min())
        self.reflectance_max = float(raw_spectra.max())
        self.reflectance_range = max(self.reflectance_max - self.reflectance_min, 1e-6)
        spectra = self.normalize_reflectance(
            raw_spectra,
            min_value=self.reflectance_min,
            max_value=self.reflectance_max,
        )
        conditions = df[self.condition_columns].to_numpy(dtype=np.float32)

        self.raw_spectra = torch.tensor(raw_spectra, dtype=dtype)
        self.reflectance_mean = float(self.raw_spectra.mean().item())
        self.reflectance_std = float(self.raw_spectra.std(unbiased=True).item())
        self.spectra = torch.tensor(spectra, dtype=dtype)
        self.conditions = torch.tensor(conditions, dtype=dtype)
        self._df = df if cache_dataframe else None

    @staticmethod
    def normalize_reflectance(
        values: np.ndarray,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> np.ndarray:
        """Min-max normalize using a single dataset-wide range to keep absolute brightness information."""
        if min_value is None:
            min_value = float(values.min())
        if max_value is None:
            max_value = float(values.max())
        value_range = max(max_value - min_value, 1e-6)
        return (values - min_value) / value_range

    @staticmethod
    def _infer_spectral_columns(columns: Iterable[str], spectral_range: tuple[int, int, int]) -> list[str]:
        start, end, step = spectral_range
        spectral_cols = []
        for col in columns:
            try:
                value = int(col)
            except ValueError:
                continue
            if start <= value <= end and ((value - start) % step == 0):
                spectral_cols.append(col)
        spectral_cols.sort(key=lambda name: int(name))

        if not spectral_cols:
            raise ValueError("No spectral columns detected in provided CSV.")
        return spectral_cols

    def __len__(self) -> int:
        return len(self.spectra)

    def __getitem__(self, idx: int) -> Batch:
        return {
            "spectrum": self.spectra[idx],
            "condition": self.conditions[idx],
        }

    @property
    def input_dim(self) -> int:
        return len(self.spectral_columns)

    @property
    def condition_dim(self) -> int:
        return len(self.condition_columns)


class PredictConditionDataset(Dataset[Batch]):
    """Dataset that yields only conditions for generation-time sampling."""

    def __init__(
        self,
        condition_dim: int,
        conditions: Optional[Sequence[Sequence[float]]] = None,
        samples_per_condition: int = 1,
        fallback_condition: Optional[Sequence[float]] = None,
        source_conditions: Optional[torch.Tensor] = None,
    ) -> None:
        if conditions:
            base = torch.tensor(conditions, dtype=torch.float32)
        elif fallback_condition is not None:
            base = torch.tensor([fallback_condition], dtype=torch.float32)
        elif source_conditions is not None and source_conditions.size(0) > 0:
            base = source_conditions[:1].clone()
        else:
            raise ValueError("Must provide either conditions, fallback_condition, or source_conditions for predict dataset")

        if base.ndim != 2 or base.size(1) != condition_dim:
            raise ValueError(f"PredictConditionDataset expected shape (N, {condition_dim}), got {tuple(base.shape)}")

        self.base_conditions = base
        self.num_conditions = base.size(0)
        self.samples_per_condition = max(int(samples_per_condition), 1)

        sample_ids = torch.arange(self.samples_per_condition).repeat_interleave(self.num_conditions)
        condition_indices = torch.arange(self.num_conditions).repeat(self.samples_per_condition)

        self.sample_ids = sample_ids
        self.condition_indices = condition_indices
        self.conditions = self.base_conditions[self.condition_indices]

    def __len__(self) -> int:
        return int(self.conditions.size(0))

    def __getitem__(self, idx: int) -> Batch:
        return {
            "condition": self.conditions[idx],
            "condition_index": torch.tensor(int(self.condition_indices[idx]), dtype=torch.long),
            "sample_id": torch.tensor(int(self.sample_ids[idx]), dtype=torch.long),
        }
