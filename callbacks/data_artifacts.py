from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from lightning.pytorch.callbacks import Callback


class SaveDataArtifacts(Callback):
    """Export run-local data artifacts needed for reporting and downstream use."""

    def __init__(
        self,
        out_dir: str = "outputs/artifacts",
        save_npy: bool = True,
    ) -> None:
        super().__init__()
        self.out_dir = out_dir
        self.save_npy = bool(save_npy)

    def on_fit_start(self, trainer, pl_module) -> None:  # type: ignore[override]
        del pl_module
        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is None:
            return

        if getattr(datamodule, "dataset", None) is None:
            datamodule.setup("fit")

        dataset = getattr(datamodule, "dataset", None)
        train_set = getattr(datamodule, "train_set", None)
        val_set = getattr(datamodule, "val_set", None)
        test_set = getattr(datamodule, "test_set", None)
        if dataset is None or train_set is None or val_set is None or test_set is None:
            return

        train_idx = self._extract_indices(train_set)
        val_idx = self._extract_indices(val_set)
        test_idx = self._extract_indices(test_set)
        if train_idx is None or val_idx is None or test_idx is None:
            return

        default_root_dir = str(getattr(trainer, "default_root_dir", "outputs"))
        out_dir = self._resolve_callback_out_dir(default_root_dir, self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        condition_columns = list(getattr(datamodule, "condition_columns", []))
        if len(condition_columns) != int(getattr(dataset, "condition_dim", 0)):
            condition_columns = [f"condition_{idx}" for idx in range(int(dataset.condition_dim))]

        conditions = dataset.conditions.detach().cpu().numpy().astype(np.float32)
        psi1_train_conditions = self._normalize_rows(conditions[train_idx])
        psi2_test_val_conditions = self._normalize_rows(np.concatenate([conditions[test_idx], conditions[val_idx]], axis=0))

        self._write_matrix_csv(out_dir / "psi1_train_conditions_normalized.csv", psi1_train_conditions, condition_columns)
        self._write_matrix_csv(out_dir / "psi2_test_val_conditions_normalized.csv", psi2_test_val_conditions, condition_columns)
        if self.save_npy:
            np.save(out_dir / "psi1_train_conditions_normalized.npy", psi1_train_conditions)
            np.save(out_dir / "psi2_test_val_conditions_normalized.npy", psi2_test_val_conditions)

        spectra_norm = dataset.spectra.detach().cpu().numpy().astype(np.float32)
        raw_min = float(getattr(dataset, "reflectance_min", 0.0))
        raw_max = float(getattr(dataset, "reflectance_max", 1.0))
        raw_range = max(float(getattr(dataset, "reflectance_range", raw_max - raw_min)), 1e-6)
        spectra_raw = (spectra_norm * raw_range) + raw_min

        spectral_columns = list(getattr(dataset, "spectral_columns", []))
        if len(spectral_columns) != int(spectra_norm.shape[1]):
            spectral_columns = [str(idx) for idx in range(int(spectra_norm.shape[1]))]

        psi1_train_spectra = spectra_norm[train_idx]
        psi2_test_val_spectra = np.concatenate([spectra_norm[test_idx], spectra_norm[val_idx]], axis=0)

        # Main psi artifacts include both normalized conditions and normalized spectra.
        psi_columns = condition_columns + spectral_columns
        psi1_train = np.concatenate([psi1_train_conditions, psi1_train_spectra], axis=1)
        psi2_test_val = np.concatenate([psi2_test_val_conditions, psi2_test_val_spectra], axis=1)
        self._write_matrix_csv(out_dir / "psi1_train_normalized.csv", psi1_train, psi_columns)
        self._write_matrix_csv(out_dir / "psi2_test_val_normalized.csv", psi2_test_val, psi_columns)
        if self.save_npy:
            np.save(out_dir / "psi1_train_normalized.npy", psi1_train)
            np.save(out_dir / "psi2_test_val_normalized.npy", psi2_test_val)

        # Keep spectra-only exports as explicit auxiliary files.
        self._write_matrix_csv(out_dir / "psi1_train_spectra_normalized.csv", psi1_train_spectra, spectral_columns)
        self._write_matrix_csv(out_dir / "psi2_test_val_spectra_normalized.csv", psi2_test_val_spectra, spectral_columns)
        if self.save_npy:
            np.save(out_dir / "psi1_train_spectra_normalized.npy", psi1_train_spectra)
            np.save(out_dir / "psi2_test_val_spectra_normalized.npy", psi2_test_val_spectra)

        minmax_payload = {
            "normalization_policy": "dataset_wide_minmax",
            "csv_path": str(getattr(datamodule, "csv_path", "")),
            "spectral_range": [int(v) for v in getattr(datamodule, "spectral_range", (400, 2490, 10))],
            "seed": int(getattr(datamodule, "seed", 0)),
            "splits": [float(v) for v in getattr(datamodule, "splits", (0.8, 0.1, 0.1))],
            "spectra_raw_min": raw_min,
            "spectra_raw_max": raw_max,
            "raw_reference": {
                "min": raw_min,
                "max": raw_max,
            },
            "normalized_reference": {
                "min": float(spectra_norm.min()),
                "max": float(spectra_norm.max()),
            },
            "train": self._split_minmax(spectra_norm[train_idx], spectra_raw[train_idx]),
            "val": self._split_minmax(spectra_norm[val_idx], spectra_raw[val_idx]),
            "test": self._split_minmax(spectra_norm[test_idx], spectra_raw[test_idx]),
            "test_val": self._split_minmax(
                np.concatenate([spectra_norm[test_idx], spectra_norm[val_idx]], axis=0),
                np.concatenate([spectra_raw[test_idx], spectra_raw[val_idx]], axis=0),
            ),
        }
        (out_dir / "minmax_stats.json").write_text(json.dumps(minmax_payload, indent=2), encoding="utf-8")

        metadata = {
            "psi1_file": "psi1_train_normalized.csv",
            "psi2_file": "psi2_test_val_normalized.csv",
            "psi1_definition": "train split with normalized conditions + normalized spectra",
            "psi2_definition": "test+val split with normalized conditions + normalized spectra",
            "psi_column_order": "condition_columns first, then spectral_columns",
            "psi1_conditions_file": "psi1_train_conditions_normalized.csv",
            "psi2_conditions_file": "psi2_test_val_conditions_normalized.csv",
            "psi1_spectra_file": "psi1_train_spectra_normalized.csv",
            "psi2_spectra_file": "psi2_test_val_spectra_normalized.csv",
            "train_size": int(train_idx.size),
            "val_size": int(val_idx.size),
            "test_size": int(test_idx.size),
            "test_val_size": int(val_idx.size + test_idx.size),
            "spectral_columns": spectral_columns,
            "condition_columns": condition_columns,
            "split_index_source": "outputs/checkpoints/data_<seed>.json",
        }
        (out_dir / "artifact_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @staticmethod
    def _extract_indices(split) -> Optional[np.ndarray]:
        indices = getattr(split, "indices", None)
        if indices is None:
            return None
        return np.asarray(indices, dtype=np.int64)

    @staticmethod
    def _resolve_callback_out_dir(default_root_dir: str, out_dir: str | Path) -> Path:
        configured = Path(out_dir)
        if configured.is_absolute():
            return configured

        root = Path(default_root_dir)
        parts = configured.parts
        if parts and parts[0] == "outputs":
            configured = Path(*parts[1:]) if len(parts) > 1 else Path(".")
        return root / configured

    @staticmethod
    def _normalize_rows(values: np.ndarray) -> np.ndarray:
        denom = np.clip(values.sum(axis=1, keepdims=True), a_min=1e-6, a_max=None)
        return values / denom

    @staticmethod
    def _write_matrix_csv(path: Path, values: np.ndarray, columns: list[str]) -> None:
        header = ",".join(columns)
        np.savetxt(path, values, delimiter=",", header=header, comments="", fmt="%.8f")

    @staticmethod
    def _split_minmax(values_norm: np.ndarray, values_raw: np.ndarray) -> dict[str, float]:
        return {
            "normalized_min": float(values_norm.min()),
            "normalized_max": float(values_norm.max()),
            "raw_min": float(values_raw.min()),
            "raw_max": float(values_raw.max()),
        }
