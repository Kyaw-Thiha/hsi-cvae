from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

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

        spectra = df[self.spectral_columns].to_numpy(dtype=np.float32)
        spectra = self.normalize_reflectance(spectra)
        conditions = df[self.condition_columns].to_numpy(dtype=np.float32)

        self.spectra = torch.tensor(spectra, dtype=dtype)
        self.conditions = torch.tensor(conditions, dtype=dtype)
        self._df = df if cache_dataframe else None

    @staticmethod
    def normalize_reflectance(values: np.ndarray) -> np.ndarray:
        """Min-max normalize each spectrum row so reflectance stays in [0, 1]."""
        mins = values.min(axis=1, keepdims=True)
        maxs = values.max(axis=1, keepdims=True)
        ranges = np.clip(maxs - mins, a_min=1e-6, a_max=None)
        return (values - mins) / ranges

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
