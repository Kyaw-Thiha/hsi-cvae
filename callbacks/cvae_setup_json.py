from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml
from lightning.pytorch.callbacks import Callback, ModelCheckpoint


class SaveCVAESetupJSON(Callback):
    """Export a diffusion-style CVAE setup JSON for downstream consumers."""

    def __init__(
        self,
        out_dir: str = "outputs/artifacts",
        filename: str = "cfg_tcvae_setup.json",
    ) -> None:
        super().__init__()
        self.out_dir = out_dir
        self.filename = filename

    def on_fit_end(self, trainer, pl_module) -> None:  # type: ignore[override]
        del pl_module
        default_root_dir = str(getattr(trainer, "default_root_dir", "outputs"))
        run_dir = Path(default_root_dir)
        out_dir = self._resolve_callback_out_dir(default_root_dir, self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        version_dir = self._latest_version_dir(run_dir)
        if version_dir is None:
            return

        config_path = version_dir / "config.yaml"
        if not config_path.exists():
            return

        with config_path.open("r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}

        payload = self._build_payload(
            cfg=cfg,
            run_dir=run_dir,
            version_dir=version_dir,
            out_dir=out_dir,
            trainer=trainer,
        )
        out_path = out_dir / self.filename
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _latest_version_dir(run_dir: Path) -> Optional[Path]:
        logs_dir = run_dir / "lightning_logs"
        if not logs_dir.exists():
            return None
        version_dirs = [p for p in logs_dir.glob("version_*") if p.is_dir() and (p / "config.yaml").exists()]
        if not version_dirs:
            return None
        return max(version_dirs, key=lambda p: p.stat().st_mtime)

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

    def _build_payload(
        self,
        cfg: dict[str, Any],
        run_dir: Path,
        version_dir: Path,
        out_dir: Path,
        trainer,
    ) -> dict[str, Any]:
        model_cfg = dict(cfg.get("model") or {})
        data_cfg = dict(cfg.get("data") or {})
        trainer_cfg = dict(cfg.get("trainer") or {})

        best_ckpt_path, interval_latest_ckpt_path, checkpoint_dirs = self._checkpoint_paths_from_trainer(trainer)
        artifact_paths = self._artifact_paths(out_dir)
        model_block = self._model_block(model_cfg)
        tcvae_block = {
            "model": model_block,
            "loss": {
                "loss_name": model_cfg.get("loss_name"),
                "loss_params": model_cfg.get("loss_params"),
            },
            "optimizer": {
                "lr": model_cfg.get("lr"),
                "weight_decay": model_cfg.get("weight_decay"),
            },
            "scheduler": model_cfg.get("scheduler_cfg"),
            "data": {
                "csv_path": data_cfg.get("csv_path"),
                "condition_columns": data_cfg.get("condition_columns"),
                "spectral_range": data_cfg.get("spectral_range"),
                "splits": data_cfg.get("splits"),
                "seed": data_cfg.get("seed"),
                "batch_size": data_cfg.get("batch_size"),
                "num_workers": data_cfg.get("num_workers"),
                "pin_memory": data_cfg.get("pin_memory"),
            },
            "sampling": {
                "temperature": model_cfg.get("temperature"),
                "guidance_scale": model_cfg.get("guidance_scale"),
                "condition_scale": model_cfg.get("condition_scale"),
                "condition_dropout": model_cfg.get("condition_dropout"),
            },
            "artifacts": {
                "run_dir": str(run_dir),
                "config_yaml": str(version_dir / "config.yaml"),
                "hparams_yaml": str(version_dir / "hparams.yaml"),
                "resume_from_ckpt_path": cfg.get("ckpt_path"),
                "checkpoint_dirs": checkpoint_dirs,
                "best_ckpt_path": best_ckpt_path,
                "interval_latest_ckpt_path": interval_latest_ckpt_path,
                **artifact_paths,
            },
            "provenance": {
                "subcommand": cfg.get("subcommand"),
                "seed_everything": cfg.get("seed_everything"),
                "trainer": {
                    "max_epochs": trainer_cfg.get("max_epochs"),
                    "precision": trainer_cfg.get("precision"),
                    "default_root_dir": trainer_cfg.get("default_root_dir"),
                },
                "exported_at_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit": self._git_commit(),
            },
        }

        return {
            "cvae_type": model_block.get("model_type", "ConditionalVAE"),
            "cfg_tcvae": tcvae_block,
            "cfg_cvae": tcvae_block,
        }

    @staticmethod
    def _model_block(model_cfg: dict[str, Any]) -> dict[str, Any]:
        block: dict[str, Any] = {
            "model_type": "ConditionalVAE",
            "architecture": model_cfg.get("architecture"),
            "input_dim": model_cfg.get("input_dim"),
            "condition_dim": model_cfg.get("condition_dim"),
            "latent_dim": model_cfg.get("latent_dim"),
            "hidden_dims": model_cfg.get("hidden_dims"),
            "dropout": model_cfg.get("dropout"),
            "wavelength_params": model_cfg.get("wavelength_params"),
        }
        for key in (
            "cnn_params",
            "transformer_params",
            "transformer_repeatz_params",
            "dual_path_transformer_params",
            "conformer_params",
        ):
            value = model_cfg.get(key)
            if value is not None:
                block[key] = value
        return block

    @staticmethod
    def _artifact_paths(artifact_dir: Path) -> dict[str, Optional[str]]:
        file_keys = {
            "norm_dict_path": "norm_dict.json",
            "minmax_stats_path": "minmax_stats.json",
            "psi1_path": "psi1_gdstreamline.csv",
            "psi2_path": "psi2_gdstreamline.csv",
            "psi1_normalized_path": "psi1_train_normalized.csv",
            "psi2_normalized_path": "psi2_test_val_normalized.csv",
            "psi1_conditions_path": "psi1_train_conditions_normalized.csv",
            "psi2_conditions_path": "psi2_test_val_conditions_normalized.csv",
            "psi1_spectra_path": "psi1_train_spectra_normalized.csv",
            "psi2_spectra_path": "psi2_test_val_spectra_normalized.csv",
            "artifact_manifest_path": "artifact_manifest.json",
        }
        result: dict[str, Optional[str]] = {}
        for key, filename in file_keys.items():
            path = artifact_dir / filename
            result[key] = str(path) if path.exists() else None
        return result

    @staticmethod
    def _checkpoint_paths_from_trainer(trainer) -> tuple[Optional[str], Optional[str], dict[str, Optional[str]]]:
        best_ckpt_path: Optional[str] = None
        best_ckpt_dir: Optional[str] = None
        interval_dir: Optional[str] = None

        for callback in getattr(trainer, "callbacks", []):
            if not isinstance(callback, ModelCheckpoint):
                continue
            dirpath = str(callback.dirpath) if callback.dirpath is not None else None
            monitor = getattr(callback, "monitor", None)
            if monitor:
                best_ckpt_dir = dirpath
                candidate = getattr(callback, "best_model_path", None) or None
                if candidate:
                    best_ckpt_path = str(candidate)
            else:
                interval_dir = dirpath

        interval_latest = SaveCVAESetupJSON._latest_checkpoint_in_dir(interval_dir)
        checkpoint_dirs = {
            "best": best_ckpt_dir,
            "interval": interval_dir,
        }
        return best_ckpt_path, interval_latest, checkpoint_dirs

    @staticmethod
    def _latest_checkpoint_in_dir(path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        dirpath = Path(path)
        if not dirpath.exists():
            return None
        checkpoints = [p for p in dirpath.glob("*.ckpt") if p.is_file()]
        if not checkpoints:
            return None
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return str(latest)

    @staticmethod
    def _git_commit() -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            return None
        return result.stdout.strip() or None
